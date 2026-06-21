"""Hybrid model: Linear Regression (LR-EW) baseline + LSTM residual.

By design the methodology uses Linear Regression (LR-EW) as the tabular baseline so
the model keeps a strong linear component, while the LSTM captures the remaining
complex relationships by learning the baseline residual. The baseline is configured
per circuit via the YAML key ``hybrid_baseline_model`` (never from the holdout). The
LSTM is trained to predict the residual ``LapTime_seconds - baseline_prediction`` and
the final prediction is ``hybrid_prediction = baseline_prediction + lstm_residual_prediction``.

Leakage control (see baseline_utils):
  - Validation: tabular trained on train_laps predicts val_laps (OOS); train-target
    residuals use an out-of-fold baseline within train_laps.
  - Holdout: tabular trained on the whole modeling block predicts the holdout; the
    final LSTM training targets use an out-of-fold baseline over the modeling block.

This script reuses the LSTM core from model_lstm.py unchanged (network, sequences,
Optuna, epoch calibration) by forcing ``lstm_target_mode = 'residual_from_tabular'`` and
supplying the tabular baseline series through the existing ``baseline_*`` parameters.
The pure LSTM, LR-EW and XGBoost-EW scripts are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from modeling_utils import (
    build_sequential_split,
    calc_holdout_ci,
    load_cleaned_data,
    log_mlflow_run,
    prepare_raw_features,
    resolve_repo_path,
    safe_gp_name,
    select_modeling_columns,
    summarize_cos,
)
from baseline_utils import (
    BASELINE_MODEL_KINDS,
    generate_block_baseline,
    generate_oof_baseline,
    load_baseline_predictions,
    resolve_baseline_window_ratio,
    resolve_xgb_ew_hparams,
)
from model_lstm_baseline import (
    LSTM_SEARCH_SPACE_VERSION,
    LSTM_TUNING_STRATEGY,
    build_split_indices,
    fit_final_lstm,
    fit_predict_lstm,
    lstm_config,
    metric_values,
    resolve_lstm_features,
    tune_lstm_hyperparams,
)

HYBRID_TARGET_MODE = "residual_from_tabular"


def build_hybrid_paths(repo_root, config, safe_name):
    # hybrid_* keys live in the raw config (they don't carry the lstm_ prefix that
    # lstm_config() filters on), so read them from config directly.
    results_dir = resolve_repo_path(repo_root, str(config["results_dir"]))
    models_dir = results_dir / str(config.get("hybrid_lstm_models_subdir", "lstm_hybrid/models"))
    params_dir = results_dir / str(config.get("hybrid_lstm_params_subdir", "lstm_hybrid/params"))
    model_path = models_dir / str(
        config.get("hybrid_lstm_model_filename_template", "{safe_gp_name}_lstm_hybrid_model.keras")
    ).format(safe_gp_name=safe_name)
    metadata_path = models_dir / str(
        config.get(
            "hybrid_lstm_model_metadata_filename_template",
            "{safe_gp_name}_lstm_hybrid_model_metadata.json",
        )
    ).format(safe_gp_name=safe_name)
    params_path = params_dir / str(
        config.get("hybrid_lstm_params_filename_template", "{safe_gp_name}_lstm_hybrid_params.json")
    ).format(safe_gp_name=safe_name)
    trials_path = params_dir / str(
        config.get(
            "hybrid_lstm_trials_filename_template", "{safe_gp_name}_lstm_hybrid_optuna_trials.csv"
        )
    ).format(safe_gp_name=safe_name)
    return model_path, metadata_path, params_path, trials_path


def resolve_window_ratio_sweep(config):
    """Return the ordered list of lstm_window_ratio values the hybrid should sweep.

    Uses ``lstm_window_ratio_sweep`` (a YAML array) when present; otherwise falls
    back to the single ``lstm_window_ratio`` scalar so behaviour is unchanged when the
    sweep key is absent. The pure LSTM script keeps reading the scalar ``lstm_window_ratio``.
    """
    sweep = config.get("lstm_window_ratio_sweep")
    if isinstance(sweep, (list, tuple)) and len(sweep) > 0:
        ratios = [float(r) for r in sweep]
    else:
        scalar = config.get(
            "lstm_window_ratio", config.get("lstm_ew_window_ratio", config["window_ratio"])
        )
        ratios = [float(scalar)]
    # Deduplicate while preserving order.
    seen = set()
    ordered = []
    for r in ratios:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered


def main():
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()
    lstm_cfg = lstm_config(config)
    lstm_cfg["lstm_target_mode"] = HYBRID_TARGET_MODE  # hybrid always learns the residual

    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    seed = int(config["random_seed"])

    model_kind = str(config.get("hybrid_baseline_model", "")).lower()
    if model_kind not in BASELINE_MODEL_KINDS:
        raise ValueError(
            f"Config key 'hybrid_baseline_model' must be one of {BASELINE_MODEL_KINDS}; got "
            f"{model_kind!r}. Set it per circuit from validation metrics (never the holdout)."
        )
    baseline_window_ratio = resolve_baseline_window_ratio(config, model_kind)

    # Tabular features (full set, e.g. with LapTime_prev) for the baseline.
    num_cols_tab, cat_cols_tab = select_modeling_columns(df_base, config)
    # LSTM features (possibly reduced by the feature mode) for the network input.
    # The hybrid uses its own feature mode (hybrid_lstm_feature_mode) so it does not
    # disturb the pure LSTM's per-circuit selection. Default full_embedding: in the
    # hybrid the baseline is the tabular model (not LapTime_prev), so there is no reason
    # to drop LapTime_prev from the network input as the auxiliary modes do.
    feature_mode = str(
        config.get("hybrid_lstm_feature_mode", lstm_cfg.get("lstm_feature_mode", "full_embedding"))
    ).lower()
    lstm_cfg["lstm_feature_mode"] = feature_mode
    num_cols_lstm, cat_cols_lstm = resolve_lstm_features(feature_mode, num_cols_tab, cat_cols_tab, target_col)

    X_lstm_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols_lstm, cat_cols_lstm, target_col)
    # Same valid_indices (target-driven) -> tabular X aligns row-for-row with the LSTM X.
    X_tab_raw, _, _ = prepare_raw_features(df_base, num_cols_tab, cat_cols_tab, target_col)

    group_cols = [col for col in list(lstm_cfg["lstm_group_cols"]) if col in df_base.columns]

    print("--- HYBRID: BEST TABULAR BASELINE + LSTM RESIDUAL ---")
    print(f"Grand Prix: {target_gp_name}")
    print(f"Baseline tabular model: {model_kind} | window_ratio={baseline_window_ratio}")
    print(f"LSTM feature mode: {feature_mode}")
    print(f"Tabular features: num={num_cols_tab} | cat={cat_cols_tab}")
    print(f"LSTM features:    num={num_cols_lstm} | cat={cat_cols_lstm}")

    (
        lap_series, lap_min, lap_max,
        model_idx, holdout_idx,
        holdout_start_lap, model_end_lap, total_laps,
    ) = build_sequential_split(df_base, valid_indices, float(config["holdout_ratio"]), lap_col)

    model_laps = lap_series.loc[model_idx]
    model_order_idx = model_laps.sort_values(kind="mergesort").index
    holdout_laps = lap_series.loc[holdout_idx]
    holdout_order_idx = holdout_laps.sort_values(kind="mergesort").index

    X_model_raw = X_lstm_raw.loc[model_order_idx].reset_index(drop=True)
    X_model_tab = X_tab_raw.loc[model_order_idx].reset_index(drop=True)
    y_model = y_raw.loc[model_order_idx].reset_index(drop=True)
    lap_model_sorted = model_laps.loc[model_order_idx].reset_index(drop=True)
    group_model = df_base.loc[model_order_idx, group_cols].reset_index(drop=True)

    X_holdout_raw = X_lstm_raw.loc[holdout_order_idx].reset_index(drop=True)
    X_holdout_tab = X_tab_raw.loc[holdout_order_idx].reset_index(drop=True)
    y_holdout = y_raw.loc[holdout_order_idx].reset_index(drop=True)
    lap_holdout_sorted = holdout_laps.loc[holdout_order_idx].reset_index(drop=True)
    group_holdout = df_base.loc[holdout_order_idx, group_cols].reset_index(drop=True)

    unique_laps = np.sort(pd.to_numeric(lap_model_sorted, errors="coerce").dropna().unique())
    n_model_laps = len(unique_laps)

    # Sequence length depends on lstm_window_ratio; the hybrid sweeps a list of ratios
    # (see resolve_window_ratio_sweep) and keeps the best by validation RMSE. n_race_laps
    # is ratio-independent so it is computed once here.
    if "Year" in df_base.columns:
        n_race_laps = len(pd.to_numeric(df_base.loc[model_order_idx, lap_col], errors="coerce").dropna().unique())
    else:
        n_race_laps = n_model_laps

    # --- Tabular baseline (ratio-independent): reuse the persisted predictions when available ---
    # The standalone LR-EW/XGB-EW scripts export per-row OOF (modeling) and holdout
    # predictions. Reusing them makes the hybrid baseline identical to the reported
    # tabular model. If absent, regenerate them deterministically (same params/data).
    # A single OOF series over the modeling block is leakage-free for both the LSTM
    # validation split and the final training targets: every OOF value is predicted by a
    # tabular model trained only on earlier laps (expanding window).
    loaded_oof, loaded_holdout = load_baseline_predictions(repo_root, config, model_kind)
    if (
        loaded_oof is not None
        and set(model_order_idx).issubset(loaded_oof.index)
        and set(holdout_order_idx).issubset(loaded_holdout.index)
    ):
        baseline_model = loaded_oof.loc[model_order_idx].reset_index(drop=True)
        baseline_holdout = loaded_holdout.loc[holdout_order_idx].reset_index(drop=True)
        baseline_source = "loaded_from_saved_predictions"
        print("\n--- Reusing saved tabular baseline predictions ---")
    else:
        print("\n--- Saved baseline predictions absent/incomplete; regenerating ---")
        xgb_params, xgb_best_n = (None, None)
        if model_kind == "xgb_ew":
            xgb_params, xgb_best_n = resolve_xgb_ew_hparams(
                repo_root, config, unique_laps, lap_model_sorted, X_model_tab, y_model, cat_cols_tab
            )
        baseline_model = generate_oof_baseline(
            model_kind, X_model_tab, y_model, lap_model_sorted, cat_cols_tab, config,
            baseline_window_ratio, xgb_params, xgb_best_n,
        )
        baseline_holdout = generate_block_baseline(
            model_kind, X_model_tab, y_model, X_holdout_tab, cat_cols_tab, xgb_params, xgb_best_n
        )
        baseline_source = "regenerated"

    print(
        f"Baseline RMSE ({baseline_source}) | OOF modeling="
        f"{np.sqrt(mean_squared_error(y_model, baseline_model)):.4f} | "
        f"holdout={np.sqrt(mean_squared_error(y_holdout, baseline_holdout)):.4f}"
    )

    print("\n--- Sequential split ---")
    print(f"Total laps: {total_laps} (LapNumber {lap_min}-{lap_max})")
    print(f"Modeling block: records={len(X_model_raw)} | unique_laps={n_model_laps}")
    print(f"Holdout block:  records={len(X_holdout_raw)}")

    safe_name = f"{safe_gp_name(target_gp_name)}_{feature_mode}_{model_kind}"
    model_path, metadata_path, params_path, trials_path = build_hybrid_paths(
        repo_root, config, safe_name
    )

    ratios = resolve_window_ratio_sweep(config)
    print(
        f"\n--- LSTM window-ratio sweep: {ratios} | "
        f"selecting the best by VALIDATION RMSE (never the holdout) ---"
    )

    def evaluate_ratio(lstm_window_ratio):
        """Train+evaluate the hybrid for one lstm_window_ratio. Returns a result dict
        (or None when the ratio leaves an empty validation split)."""
        # Fresh config copy per ratio: tuning mutates it in place.
        cfg = dict(lstm_cfg)
        sequence_length = max(1, int(np.ceil(n_race_laps * lstm_window_ratio)))
        cfg["lstm_sequence_length"] = sequence_length
        cfg["lstm_sequence_length_source"] = "lstm_window_ratio_times_race_laps"

        n_train_laps = max(
            sequence_length + 1, int(np.floor(n_model_laps * float(config["window_train_ratio"])))
        )
        train_laps = unique_laps[:n_train_laps]
        val_laps = unique_laps[n_train_laps:]
        if len(val_laps) == 0:
            print(
                f"\n=== ratio={lstm_window_ratio} | sequence_length={sequence_length}: "
                f"validation split empty; skipping ==="
            )
            return None

        # Per-ratio Optuna params/trials files so the sweep does not clobber itself.
        ratio_tag = f"{lstm_window_ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        r_params_path = params_path.with_name(
            f"{params_path.stem}_wr{ratio_tag}{params_path.suffix}"
        )
        r_trials_path = trials_path.with_name(
            f"{trials_path.stem}_wr{ratio_tag}{trials_path.suffix}"
        )

        print(
            f"\n=== ratio={lstm_window_ratio} | sequence_length={sequence_length} | "
            f"train_laps={len(train_laps)} | val_laps={len(val_laps)} ==="
        )
        if bool(cfg.get("use_saved_lstm_params", False)):
            print(f"  Saved-parameter candidate: {r_params_path}")

        if bool(cfg["lstm_tuning_enabled"]) or bool(cfg.get("use_saved_lstm_params", False)):
            cfg, optuna_best_epoch, optuna_summary = tune_lstm_hyperparams(
                X_model_raw, y_model, lap_model_sorted, group_model,
                train_laps, val_laps, cat_cols_lstm, cfg, seed=seed,
                params_path=r_params_path, trials_path=r_trials_path,
                baseline_model=baseline_model,
            )
            cfg["lstm_sequence_length"] = sequence_length
            cfg["lstm_sequence_length_source"] = "lstm_window_ratio_times_race_laps"
        else:
            optuna_best_epoch = int(cfg["lstm_epochs"])
            optuna_summary = None
            print("LSTM Optuna tuning disabled; using YAML hyperparameters.")

        final_epoch_count = max(optuna_best_epoch, int(cfg["lstm_min_final_epochs"]))

        # --- Validation split evaluation ---
        context_mask, train_mask, val_mask, train_idx, val_idx = build_split_indices(
            X_model_raw, lap_model_sorted, train_laps, val_laps
        )
        preds_val, y_val_seq, val_seq_laps, *_, _ = fit_predict_lstm(
            X_model_raw.loc[train_mask],
            X_model_raw.loc[context_mask],
            y_model.loc[context_mask],
            lap_model_sorted.loc[context_mask],
            group_model.loc[context_mask],
            train_idx, val_idx,
            cat_cols_lstm, cfg, seed=seed,
            baseline_context=baseline_model.loc[context_mask],
        )
        val_metrics = metric_values(y_val_seq, preds_val)
        val_ci = calc_holdout_ci(y_val_seq, preds_val, seed=seed)
        print(
            f"  Val sequences: {len(y_val_seq)} | "
            f"RMSE={val_metrics['rmse']:.4f} "
            f"[{val_ci['rmse'][0]:.4f}, {val_ci['rmse'][1]:.4f}] | "
            f"MAE={val_metrics['mae']:.4f} "
            f"[{val_ci['mae'][0]:.4f}, {val_ci['mae'][1]:.4f}] | "
            f"R2={val_metrics['r2']:.4f} "
            f"[{val_ci['r2'][0]:.4f}, {val_ci['r2'][1]:.4f}]"
        )

        # --- Final model + holdout ---
        (
            preds_holdout, y_holdout_seq, holdout_seq_laps,
            final_model, _, _, _, feature_names, final_epoch_count, feature_meta,
        ) = fit_final_lstm(
            X_model_raw, y_model, lap_model_sorted, group_model,
            X_holdout_raw, y_holdout, lap_holdout_sorted, group_holdout,
            cat_cols_lstm, cfg, seed=seed, final_epoch_count=final_epoch_count,
            baseline_model_series=baseline_model, baseline_holdout_series=baseline_holdout,
        )
        holdout_ci = calc_holdout_ci(y_holdout_seq, preds_holdout, seed=seed)
        holdout_metrics = metric_values(y_holdout_seq, preds_holdout)
        print(
            f"  Holdout RMSE={holdout_metrics['rmse']:.4f} "
            f"[{holdout_ci['rmse'][0]:.4f}, {holdout_ci['rmse'][1]:.4f}] | "
            f"MAE={holdout_metrics['mae']:.4f} "
            f"[{holdout_ci['mae'][0]:.4f}, {holdout_ci['mae'][1]:.4f}] | "
            f"R2={holdout_metrics['r2']:.4f} "
            f"[{holdout_ci['r2'][0]:.4f}, {holdout_ci['r2'][1]:.4f}]"
        )

        return {
            "lstm_window_ratio": float(lstm_window_ratio),
            "sequence_length": int(sequence_length),
            "train_laps": train_laps,
            "val_laps": val_laps,
            "cfg": cfg,
            "final_epoch_count": int(final_epoch_count),
            "optuna_summary": optuna_summary,
            "val_metrics": val_metrics,
            "val_ci": val_ci,
            "holdout_metrics": holdout_metrics,
            "holdout_ci": holdout_ci,
            "y_val_seq": y_val_seq,
            "y_holdout_seq": y_holdout_seq,
            "final_model": final_model,
            "feature_names": feature_names,
            "params_path": r_params_path,
            "trials_path": r_trials_path,
        }

    sweep_rows = []
    best = None
    for lstm_window_ratio in ratios:
        result = evaluate_ratio(lstm_window_ratio)
        if result is None:
            continue
        sweep_rows.append(
            {
                "lstm_window_ratio": result["lstm_window_ratio"],
                "sequence_length": result["sequence_length"],
                "val_rmse": result["val_metrics"]["rmse"],
                "val_mae": result["val_metrics"]["mae"],
                "val_r2": result["val_metrics"]["r2"],
                "holdout_rmse": result["holdout_metrics"]["rmse"],
                "holdout_mae": result["holdout_metrics"]["mae"],
                "holdout_r2": result["holdout_metrics"]["r2"],
            }
        )
        if best is None or result["val_metrics"]["rmse"] < best["val_metrics"]["rmse"]:
            best = result

    if best is None:
        raise ValueError(
            "No lstm_window_ratio in the sweep produced a non-empty validation split. "
            "Lower the ratios or reduce window_train_ratio."
        )

    # Unpack the winning configuration (selected purely on validation RMSE).
    lstm_window_ratio = best["lstm_window_ratio"]
    sequence_length = best["sequence_length"]
    train_laps = best["train_laps"]
    val_laps = best["val_laps"]
    lstm_cfg = best["cfg"]
    final_epoch_count = best["final_epoch_count"]
    optuna_summary = best["optuna_summary"]
    val_metrics = best["val_metrics"]
    val_ci = best["val_ci"]
    holdout_metrics = best["holdout_metrics"]
    holdout_ci = best["holdout_ci"]
    y_val_seq = best["y_val_seq"]
    y_holdout_seq = best["y_holdout_seq"]
    final_model = best["final_model"]
    feature_names = best["feature_names"]
    params_path = best["params_path"]
    trials_path = best["trials_path"]

    print(
        f"\n--- Selected best lstm_window_ratio={lstm_window_ratio} "
        f"(val RMSE={val_metrics['rmse']:.4f}, sequence_length={sequence_length}, "
        f"train_laps={len(train_laps)}, val_laps={len(val_laps)}) ---"
    )

    # Persist the full sweep table so the overnight run leaves an auditable record.
    sweep_df = pd.DataFrame(sweep_rows).sort_values("val_rmse").reset_index(drop=True)
    sweep_path = params_path.parent / f"{safe_name}_window_ratio_sweep.csv"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved window-ratio sweep table to: {sweep_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(model_path)

    # Baseline-only holdout metrics over the full holdout block (the tabular model's own
    # holdout performance, matching the standalone LR-EW/XGB-EW scripts). Computed on all
    # holdout records rather than the LSTM sequence subset, which is grouped by
    # (Year, Driver) and cannot be aligned positionally to baseline_holdout.
    baseline_holdout_metrics = metric_values(y_holdout.to_numpy(), baseline_holdout.to_numpy())

    results_for_cos = {
        "window": [1],
        "rmse": [val_metrics["rmse"]],
        "mae": [val_metrics["mae"]],
        "r2": [val_metrics["r2"]],
        "std": [val_metrics["std"]],
    }
    cos = summarize_cos(
        results_for_cos,
        val_metrics["mae"], val_metrics["rmse"],
        holdout_metrics["mae"], holdout_metrics["rmse"],
        val_metrics["std"], holdout_metrics["std"],
        float(config["alpha_cos"]), float(config["beta_cos"]),
        r2_m=val_metrics["r2"], r2_holdout=holdout_metrics["r2"],
    )

    metadata = {
        "target_gp_name": target_gp_name,
        "model": "lstm_hybrid",
        "hybrid_baseline_model": model_kind,
        "baseline_window_ratio": baseline_window_ratio,
        "validation_protocol": "single_sequential_split",
        "tuning_strategy": LSTM_TUNING_STRATEGY,
        "search_space_version": LSTM_SEARCH_SPACE_VERSION,
        "model_path": str(model_path),
        "target_col": target_col,
        "lap_col": lap_col,
        "feature_mode": feature_mode,
        "tabular_numerical_features": num_cols_tab,
        "tabular_categorical_features": cat_cols_tab,
        "lstm_numerical_features": num_cols_lstm,
        "lstm_categorical_features": cat_cols_lstm,
        "encoded_feature_names": list(feature_names),
        "sequence_length": int(sequence_length),
        "target_mode": HYBRID_TARGET_MODE,
        "lstm_window_ratio": lstm_window_ratio,
        "lstm_window_ratio_sweep": ratios,
        "lstm_window_ratio_selection": "best validation RMSE (never holdout)",
        "lstm_window_ratio_sweep_results": sweep_df.to_dict(orient="records"),
        "window_train_ratio": float(config["window_train_ratio"]),
        "modeling_lap_count": int(n_model_laps),
        "train_laps": int(len(train_laps)),
        "val_laps": int(len(val_laps)),
        "sequence_groups": group_cols,
        "baseline_source": baseline_source,
        "baseline_generation": {
            "modeling": "out-of-fold (expanding-window) tabular predictions; used for both the "
            "LSTM validation split and the final training residual targets",
            "holdout": "tabular trained on the whole modeling block predicting the holdout",
            "reuse": "loaded from the standalone LR-EW/XGB-EW per-row prediction export when present",
            "leakage_note": "holdout never used to train the tabular baseline or to select it",
        },
        "final_epoch_count": int(final_epoch_count),
        "val_metrics": val_metrics,
        "val_ci": val_ci,
        "holdout_metrics": holdout_metrics,
        "baseline_full_holdout_metrics": baseline_holdout_metrics,
        "baseline_full_holdout_note": "tabular baseline over all holdout records (not the LSTM sequence subset)",
        "optuna_summary": optuna_summary,
        "lstm_config": {k: v for k, v in lstm_cfg.items() if not callable(v)},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Saved final hybrid model to: {model_path}")
    print(f"Saved metadata to: {metadata_path}")

    # Persist baseline predictions for the non-leakage audit.
    baseline_subdir = str(config.get("hybrid_baseline_predictions_subdir", "lstm_hybrid/baseline"))
    baseline_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / baseline_subdir
    baseline_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"lap": lap_model_sorted, "y": y_model, "baseline_oof": baseline_model}
    ).to_csv(baseline_dir / f"{safe_name}_hybrid_baseline_modeling.csv", index=False)
    pd.DataFrame({"y": y_holdout, "baseline_holdout": baseline_holdout}).to_csv(
        baseline_dir / f"{safe_name}_hybrid_baseline_holdout.csv", index=False
    )

    split_info = {
        "total_laps": total_laps,
        "model_records": len(X_model_raw),
        "modeling_lap_count": int(n_model_laps),
        "train_laps": int(len(train_laps)),
        "val_laps": int(len(val_laps)),
        "val_sequences": int(len(y_val_seq)),
        "sequence_length": int(sequence_length),
        "holdout_records": len(X_holdout_raw),
        "holdout_sequences": int(len(y_holdout_seq)),
    }
    summary_metrics = {
        "val_rmse": val_metrics["rmse"],
        "val_rmse_ci": val_ci["rmse"],
        "val_mae": val_metrics["mae"],
        "val_mae_ci": val_ci["mae"],
        "val_r2": val_metrics["r2"],
        "val_r2_ci": val_ci["r2"],
        "val_residual_std": val_metrics["std"],
        "holdout_rmse": holdout_metrics["rmse"],
        "holdout_rmse_ci": holdout_ci["rmse"],
        "holdout_mae": holdout_metrics["mae"],
        "holdout_mae_ci": holdout_ci["mae"],
        "holdout_r2": holdout_metrics["r2"],
        "holdout_r2_ci": holdout_ci["r2"],
        "holdout_residual_std": holdout_metrics["std"],
        "baseline_full_holdout_rmse": baseline_holdout_metrics["rmse"],
        "baseline_full_holdout_mae": baseline_holdout_metrics["mae"],
        "baseline_full_holdout_r2": baseline_holdout_metrics["r2"],
        "cos_mae": cos["cos_mae"],
        "cos_mae_ci": cos["cos_mae_ci"],
        "cos_rmse": cos["cos_rmse"],
        "cos_rmse_ci": cos["cos_rmse_ci"],
        "cos_r2": cos["cos_r2"],
        "cos_r2_ci": cos["cos_r2_ci"],
    }
    log_mlflow_run(
        repo_root, config, "lstm_hybrid", num_cols_lstm, cat_cols_lstm,
        split_info, results_for_cos, summary_metrics,
        extra_params={
            "hybrid_baseline_model": model_kind,
            "baseline_window_ratio": baseline_window_ratio,
            "feature_mode": feature_mode,
            "validation_protocol": "single_sequential_split",
            "target_mode": HYBRID_TARGET_MODE,
            "sequence_length": int(sequence_length),
            "lstm_window_ratio": lstm_window_ratio,
            "window_train_ratio": float(config["window_train_ratio"]),
            "lstm_final_epoch_count": int(final_epoch_count),
        },
        artifacts=[model_path, metadata_path, *(p for p in [params_path, trials_path] if p.exists())],
        validation_mode="single_split",
    )

    print("\n--- Validation split (hybrid) ---")
    print(f"Hybrid  RMSE: {val_metrics['rmse']:.4f} | 95% CI: "
          f"[{val_ci['rmse'][0]:.4f}, {val_ci['rmse'][1]:.4f}]")
    print(f"Hybrid  MAE:  {val_metrics['mae']:.4f} | 95% CI: "
          f"[{val_ci['mae'][0]:.4f}, {val_ci['mae'][1]:.4f}]")
    print(f"Hybrid  R2:   {val_metrics['r2']:.4f} | 95% CI: "
          f"[{val_ci['r2'][0]:.4f}, {val_ci['r2'][1]:.4f}]")

    print("\n--- Sequential holdout (hybrid) ---")
    print(f"Hybrid  RMSE: {holdout_metrics['rmse']:.4f} | 95% CI: "
          f"[{holdout_ci['rmse'][0]:.4f}, {holdout_ci['rmse'][1]:.4f}]")
    print(f"Baseline RMSE (tabular only, full holdout): {baseline_holdout_metrics['rmse']:.4f}")
    print(f"Hybrid  MAE:  {holdout_metrics['mae']:.4f} | 95% CI: "
          f"[{holdout_ci['mae'][0]:.4f}, {holdout_ci['mae'][1]:.4f}]")
    print(f"Hybrid  R2:   {holdout_metrics['r2']:.4f} | 95% CI: "
          f"[{holdout_ci['r2'][0]:.4f}, {holdout_ci['r2'][1]:.4f}]")
    print(f"COS_RMSE: {cos['cos_rmse']:.4f} | COS_MAE: {cos['cos_mae']:.4f} | COS_R2: {cos['cos_r2']:.4f}")


if __name__ == "__main__":
    main()
