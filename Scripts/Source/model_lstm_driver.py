"""Driver-filtered standalone LSTM predicting LapTime_seconds directly (absolute target).

Complementary methodological comparison (NOT a replacement for model_lstm_baseline.py or
model_lstm_hybrid.py). The dataset is filtered to a single ``Driver`` (chosen with
``--driver``) before the temporal split, and the LSTM is trained with
``lstm_target_mode = "absolute"`` so it forecasts the raw lap time directly — no residual,
no tabular baseline. This closes the per-driver comparison with the driver-filtered tabular
baselines (model_lr_ew_driver.py / model_xgb_ew_driver.py).

It reuses the LSTM core from model_lstm_baseline.py unchanged (sequence building, Optuna/TPE
tuning, epoch calibration, embeddings, metrics, COS). Differences vs. the pure LSTM runner:
  - filters the dataset to one driver before everything;
  - forces the absolute target and the ``full_embedding`` feature mode (all features incl.
    ``LapTime_prev``; ``Driver`` is dropped because it is constant after the filter, so
    ``Team`` is the only embedded categorical);
  - writes artifacts/params/trials to dedicated ``lstm_driver`` paths keyed by circuit AND
    driver, with a per-driver Optuna parameter cache (never the full-circuit LSTM params).

Sequences are grouped by ``[Year, Driver]``; with a constant driver this collapses to one
trajectory per year. The protocol is the project's single sequential split (the LSTM never
uses expanding/sliding windows), so the fair comparison with the tabular driver baselines is
mainly on the sequential holdout and the COS metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
from model_lstm_baseline import (
    LSTM_SEARCH_SPACE_VERSION,
    LSTM_TUNING_STRATEGY,
    _embedding_dim,
    build_split_indices,
    fit_final_lstm,
    fit_predict_lstm,
    lstm_config,
    metric_values,
    resolve_embedding_cols,
    resolve_lstm_features,
    tune_lstm_hyperparams,
)

DRIVER_TARGET_MODE = "absolute"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--driver", required=True, help="Driver code to filter on (e.g. VER, NOR, HAM).")
    return parser.parse_args()


def build_lstm_driver_model_paths(repo_root, config, safe_driver):
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    model_filename = str(
        config.get("lstm_driver_model_filename_template", "{safe_gp_name}_{safe_driver}_lstm_model_driver.keras")
    ).format(target_gp_name=target_gp_name, safe_gp_name=safe_name, safe_driver=safe_driver)
    metadata_filename = str(
        config.get(
            "lstm_driver_model_metadata_filename_template",
            "{safe_gp_name}_{safe_driver}_lstm_model_driver_metadata.json",
        )
    ).format(target_gp_name=target_gp_name, safe_gp_name=safe_name, safe_driver=safe_driver)
    subdir = str(config.get("lstm_driver_models_subdir", "lstm_driver/models"))
    model_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / subdir
    return model_dir / model_filename, model_dir / metadata_filename


def build_lstm_driver_params_paths(repo_root, config, safe_driver):
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    params_filename = str(
        config.get("lstm_driver_params_filename_template", "{safe_gp_name}_{safe_driver}_lstm_params_driver.json")
    ).format(target_gp_name=target_gp_name, safe_gp_name=safe_name, safe_driver=safe_driver)
    trials_filename = str(
        config.get(
            "lstm_driver_trials_filename_template",
            "{safe_gp_name}_{safe_driver}_lstm_optuna_trials_driver.csv",
        )
    ).format(target_gp_name=target_gp_name, safe_gp_name=safe_name, safe_driver=safe_driver)
    subdir = str(config.get("lstm_driver_params_subdir", "lstm_driver/params"))
    params_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / subdir
    return params_dir / params_filename, params_dir / trials_filename


def main():
    args = parse_args()
    driver = args.driver.strip().upper()
    safe_driver = driver.lower()

    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()
    lstm_cfg = lstm_config(config)
    # Standalone, non-residual: forecast the raw lap time directly.
    lstm_cfg["lstm_target_mode"] = DRIVER_TARGET_MODE
    feature_mode = str(config.get("lstm_driver_feature_mode", "full_embedding")).lower()
    lstm_cfg["lstm_feature_mode"] = feature_mode
    # Per-driver Optuna is the agreed strategy: the circuit YAML may disable tuning because
    # the full-circuit LSTM loads fixed saved params, but the filtered dataset is different,
    # so always tune per driver. use_saved_lstm_params still lets a matching per-driver cache
    # be reused on reruns (the params path carries the driver name).
    lstm_cfg["lstm_tuning_enabled"] = True

    if "Driver" not in df_base.columns:
        print("SKIP: cleaned dataset has no 'Driver' column; cannot run the driver-filtered LSTM.")
        sys.exit(0)

    available_drivers = sorted(df_base["Driver"].dropna().astype(str).str.upper().unique())
    df_base = df_base[df_base["Driver"].astype(str).str.upper() == driver].reset_index(drop=True)
    filtered_record_count = len(df_base)
    if filtered_record_count == 0:
        print(
            f"SKIP: driver={driver!r} has no rows in {target_gp_name}. "
            f"Available drivers: {', '.join(available_drivers)}. No artifacts generated."
        )
        sys.exit(0)

    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    seed = int(config["random_seed"])

    num_cols, cat_cols = select_modeling_columns(df_base, config)
    num_cols, cat_cols = resolve_lstm_features(feature_mode, num_cols, cat_cols, target_col)
    # Driver is constant after the filter -> drop it. resolve_embedding_cols only embeds
    # columns still present in cat_cols, so this also removes Driver from the embeddings
    # (Team remains). Documented in the metadata.
    dropped_constant_features = [c for c in cat_cols if c == "Driver"]
    cat_cols = [c for c in cat_cols if c != "Driver"]

    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    # Absolute target -> no residual baseline series.
    baseline_model = None
    baseline_holdout = None

    group_cols = [col for col in list(lstm_cfg["lstm_group_cols"]) if col in df_base.columns]

    print("--- LSTM (DRIVER-FILTERED, ABSOLUTE TARGET): SINGLE SEQUENTIAL SPLIT + HOLDOUT ---")
    print(f"Grand Prix: {target_gp_name} | Driver: {driver}")
    print(f"Filtered records: {filtered_record_count}")
    print(f"Feature mode: {feature_mode} | target mode: {DRIVER_TARGET_MODE}")
    print(f"Numerical features: {num_cols}")
    print(f"Categorical features: {cat_cols} (dropped constant: {dropped_constant_features})")
    embed_cols, embed_max_dim = resolve_embedding_cols(lstm_cfg, cat_cols)
    if embed_cols:
        onehot_cols = [c for c in cat_cols if c not in embed_cols]
        dims = {
            c: _embedding_dim(int(df_base[c].fillna("Missing").astype(str).nunique()), embed_max_dim)
            for c in embed_cols
        }
        print(f"Embedding (cat -> dim aprox.): {dims} | One-hot: {onehot_cols}")
    print(f"LSTM sequence groups: {group_cols if group_cols else 'none (flat sequences)'}")

    try:
        (
            lap_series, lap_min, lap_max,
            model_idx, holdout_idx,
            holdout_start_lap, model_end_lap, total_laps,
        ) = build_sequential_split(df_base, valid_indices, float(config["holdout_ratio"]), lap_col)
    except ValueError as exc:
        print(
            f"SKIP: driver={driver!r} in {target_gp_name} cannot form a sequential split "
            f"(filtered_records={filtered_record_count}): {exc}. No artifacts generated."
        )
        sys.exit(0)

    model_laps = lap_series.loc[model_idx]
    model_order_idx = model_laps.sort_values(kind="mergesort").index
    holdout_laps = lap_series.loc[holdout_idx]
    holdout_order_idx = holdout_laps.sort_values(kind="mergesort").index

    X_model_raw = X_raw.loc[model_order_idx].reset_index(drop=True)
    y_model = y_raw.loc[model_order_idx].reset_index(drop=True)
    lap_model_sorted = model_laps.loc[model_order_idx].reset_index(drop=True)
    group_model = df_base.loc[model_order_idx, group_cols].reset_index(drop=True)

    X_holdout_raw = X_raw.loc[holdout_order_idx].reset_index(drop=True)
    y_holdout = y_raw.loc[holdout_order_idx].reset_index(drop=True)
    lap_holdout_sorted = holdout_laps.loc[holdout_order_idx].reset_index(drop=True)
    group_holdout = df_base.loc[holdout_order_idx, group_cols].reset_index(drop=True)

    unique_laps = np.sort(pd.to_numeric(lap_model_sorted, errors="coerce").dropna().unique())
    n_model_laps = len(unique_laps)
    n_unique_temporal_laps = n_model_laps

    lstm_window_ratio = float(
        config.get("lstm_window_ratio", config.get("lstm_ew_window_ratio", config["window_ratio"]))
    )
    if "Year" in df_base.columns:
        n_race_laps = len(
            pd.to_numeric(df_base.loc[model_order_idx, lap_col], errors="coerce").dropna().unique()
        )
    else:
        n_race_laps = n_model_laps
    sequence_length = max(1, int(np.ceil(n_race_laps * lstm_window_ratio)))
    lstm_cfg["lstm_sequence_length"] = sequence_length
    lstm_cfg["lstm_sequence_length_source"] = "lstm_window_ratio_times_race_laps"

    n_train_laps = max(sequence_length + 1, int(np.floor(n_model_laps * float(config["window_train_ratio"]))))
    train_laps = unique_laps[:n_train_laps]
    val_laps = unique_laps[n_train_laps:]
    if len(val_laps) == 0:
        print(
            f"SKIP: driver={driver!r} in {target_gp_name} leaves an empty validation split "
            f"(modeling_laps={n_model_laps}, sequence_length={sequence_length}). No artifacts generated."
        )
        sys.exit(0)

    print("\n--- Sequential split ---")
    print(f"Total laps: {total_laps} (step {lap_min}-{lap_max})")
    print(f"Modeling block: records={len(X_model_raw)} | unique_laps={n_model_laps}")
    print(f"Holdout block:  records={len(X_holdout_raw)}")
    print(
        f"Train split: {len(train_laps)} laps | Val split: {len(val_laps)} laps | "
        f"sequence_length={sequence_length} | lstm_window_ratio={lstm_window_ratio}"
    )

    lstm_params_path, lstm_trials_path = build_lstm_driver_params_paths(repo_root, config, safe_driver)

    try:
        if bool(lstm_cfg["lstm_tuning_enabled"]) or bool(lstm_cfg.get("use_saved_lstm_params", False)):
            lstm_cfg, optuna_best_epoch, optuna_summary = tune_lstm_hyperparams(
                X_model_raw, y_model, lap_model_sorted, group_model,
                train_laps, val_laps, cat_cols, lstm_cfg, seed=seed,
                params_path=lstm_params_path, trials_path=lstm_trials_path,
                baseline_model=baseline_model,
            )
            lstm_cfg["lstm_sequence_length"] = sequence_length
            lstm_cfg["lstm_sequence_length_source"] = "lstm_window_ratio_times_race_laps"
            tuned_seq_len = int(lstm_cfg["lstm_sequence_length"])
            if len(train_laps) <= tuned_seq_len:
                n_train_laps = max(tuned_seq_len + 1, int(np.floor(n_model_laps * float(config["window_train_ratio"]))))
                train_laps = unique_laps[:n_train_laps]
                val_laps = unique_laps[n_train_laps:]
                if len(val_laps) == 0:
                    print(
                        f"SKIP: driver={driver!r} leaves an empty validation split after "
                        f"sequence_length={tuned_seq_len} adjustment. No artifacts generated."
                    )
                    sys.exit(0)
        else:
            optuna_best_epoch = int(lstm_cfg["lstm_epochs"])
            optuna_summary = None
            print("LSTM Optuna tuning disabled; using YAML hyperparameters.")

        final_epoch_count = max(optuna_best_epoch, int(lstm_cfg["lstm_min_final_epochs"]))
        print(
            f"\nSelected LSTM config: "
            f"sequence_length={lstm_cfg['lstm_sequence_length']} | units={lstm_cfg['lstm_units']} | "
            f"dense_units={lstm_cfg['lstm_dense_units']} | dropout={lstm_cfg['lstm_dropout']:.3f} | "
            f"lr={lstm_cfg['lstm_learning_rate']:.5f} | batch={lstm_cfg['lstm_batch_size']} | "
            f"final_epochs={final_epoch_count}"
        )

        print("\n--- Validation split evaluation ---")
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
            cat_cols, lstm_cfg, seed=seed,
            baseline_context=None,
        )
        val_metrics = metric_values(y_val_seq, preds_val)
        print(
            f"Val sequences: {len(y_val_seq)} | "
            f"RMSE={val_metrics['rmse']:.4f} | MAE={val_metrics['mae']:.4f} | R2={val_metrics['r2']:.4f}"
        )

        print("\n--- Training final LSTM model ---")
        (
            preds_holdout, y_holdout_seq, holdout_seq_laps,
            final_model, _, _, _, feature_names, final_epoch_count, feature_meta, _,
        ) = fit_final_lstm(
            X_model_raw, y_model, lap_model_sorted, group_model,
            X_holdout_raw, y_holdout, lap_holdout_sorted, group_holdout,
            cat_cols, lstm_cfg, seed=seed, final_epoch_count=final_epoch_count,
            baseline_model_series=baseline_model, baseline_holdout_series=baseline_holdout,
        )
    except ValueError as exc:
        print(
            f"SKIP: driver={driver!r} in {target_gp_name} could not build LSTM sequences "
            f"(sequence_length={sequence_length}, modeling_records={len(X_model_raw)}): {exc}. "
            "No artifacts generated."
        )
        sys.exit(0)

    lstm_model_path, lstm_model_metadata_path = build_lstm_driver_model_paths(repo_root, config, safe_driver)
    lstm_model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(lstm_model_path)

    holdout_ci = calc_holdout_ci(y_holdout_seq, preds_holdout, seed=seed)
    holdout_metrics = metric_values(y_holdout_seq, preds_holdout)

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

    lstm_model_metadata = {
        "target_gp_name": target_gp_name,
        "target_driver": driver,
        "model": "lstm_driver",
        "validation_protocol": "single_sequential_split_driver_filtered",
        "baseline_note": (
            "driver-filtered standalone LSTM (absolute target); complementary to the main "
            "models; not a replacement"
        ),
        "tuning_strategy": LSTM_TUNING_STRATEGY,
        "search_space_version": LSTM_SEARCH_SPACE_VERSION,
        "model_path": str(lstm_model_path),
        "target_col": target_col,
        "lap_col": lap_col,
        "feature_mode": feature_mode,
        "target_mode": DRIVER_TARGET_MODE,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "dropped_constant_features": dropped_constant_features,
        "embedding_cols": embed_cols,
        "onehot_cols": [c for c in cat_cols if c not in embed_cols],
        "embedding_spec": feature_meta.get("embed_spec", []),
        "encoded_feature_names": list(feature_names),
        "sequence_length": int(lstm_cfg["lstm_sequence_length"]),
        "sequence_length_source": lstm_cfg["lstm_sequence_length_source"],
        "lstm_window_ratio": lstm_window_ratio,
        "window_train_ratio": float(config["window_train_ratio"]),
        "filtered_record_count": int(filtered_record_count),
        "modeling_record_count": int(len(X_model_raw)),
        "holdout_record_count": int(len(X_holdout_raw)),
        "n_unique_temporal_laps": int(n_unique_temporal_laps),
        "modeling_lap_count": int(n_model_laps),
        "train_laps": len(train_laps),
        "val_laps": len(val_laps),
        "sequence_groups": group_cols,
        "training_block": "first_sequential_modeling_block",
        "holdout_usage": "holdout laps are forecast targets only; never used for training, tuning, or early stopping",
        "preprocessing": "median_imputer_standard_scaler_one_hot_full_rank_or_embeddings",
        "final_epoch_count": int(final_epoch_count),
        "val_metrics": val_metrics,
        "optuna_summary": optuna_summary,
    }
    lstm_model_metadata_path.write_text(json.dumps(lstm_model_metadata, indent=2, default=str), encoding="utf-8")
    print(f"Saved final driver-filtered LSTM model to: {lstm_model_path}")
    print(f"Saved metadata to: {lstm_model_metadata_path}")

    split_info = {
        "target_driver": driver,
        "filtered_record_count": int(filtered_record_count),
        "total_laps": total_laps,
        "model_records": len(X_model_raw),
        "modeling_lap_count": int(n_model_laps),
        "n_unique_temporal_laps": int(n_unique_temporal_laps),
        "train_laps": int(len(train_laps)),
        "val_laps": int(len(val_laps)),
        "val_sequences": int(len(y_val_seq)),
        "sequence_length": int(lstm_cfg["lstm_sequence_length"]),
        "holdout_records": len(X_holdout_raw),
        "holdout_sequences": int(len(y_holdout_seq)),
    }
    summary_metrics = {
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_r2": val_metrics["r2"],
        "val_residual_std": val_metrics["std"],
        "holdout_rmse": holdout_metrics["rmse"],
        "holdout_rmse_ci": holdout_ci["rmse"],
        "holdout_mae": holdout_metrics["mae"],
        "holdout_mae_ci": holdout_ci["mae"],
        "holdout_r2": holdout_metrics["r2"],
        "holdout_r2_ci": holdout_ci["r2"],
        "holdout_residual_std": holdout_metrics["std"],
        "cos_mae": cos["cos_mae"],
        "cos_mae_ci": cos["cos_mae_ci"],
        "cos_rmse": cos["cos_rmse"],
        "cos_rmse_ci": cos["cos_rmse_ci"],
        "cos_r2": cos["cos_r2"],
        "cos_r2_ci": cos["cos_r2_ci"],
    }
    log_mlflow_run(
        repo_root, config, "lstm_driver", num_cols, cat_cols,
        split_info, results_for_cos, summary_metrics,
        extra_params={
            "target_driver": driver,
            "validation_protocol": "single_sequential_split_driver_filtered",
            "feature_mode": feature_mode,
            "target_mode": DRIVER_TARGET_MODE,
            "tuning_strategy": LSTM_TUNING_STRATEGY,
            "search_space_version": LSTM_SEARCH_SPACE_VERSION,
            "sequence_length": int(lstm_cfg["lstm_sequence_length"]),
            "lstm_window_ratio": lstm_window_ratio,
            "window_train_ratio": float(config["window_train_ratio"]),
            "sequence_groups": ", ".join(group_cols),
            "lstm_tuning_enabled": bool(lstm_cfg["lstm_tuning_enabled"]),
            "lstm_optuna_trials": int(lstm_cfg["lstm_optuna_trials"]),
            "lstm_units": int(lstm_cfg["lstm_units"]),
            "lstm_dense_units": int(lstm_cfg["lstm_dense_units"]),
            "lstm_learning_rate": float(lstm_cfg["lstm_learning_rate"]),
            "lstm_batch_size": int(lstm_cfg["lstm_batch_size"]),
            "lstm_final_epoch_count": int(final_epoch_count),
        },
        artifacts=[
            lstm_model_path,
            lstm_model_metadata_path,
            *(p for p in [lstm_params_path, lstm_trials_path] if p.exists()),
        ],
        validation_mode="single_split_driver",
    )

    print("\n--- Validation split ---")
    print(f"Val sequences: {len(y_val_seq)} | RMSE: {val_metrics['rmse']:.4f} | "
          f"MAE: {val_metrics['mae']:.4f} | R2: {val_metrics['r2']:.4f}")

    print("\n--- Sequential holdout ---")
    print(f"Holdout sequences: {len(y_holdout_seq)}")
    print(f"RMSE: {holdout_metrics['rmse']:.4f} | 95% CI: [{holdout_ci['rmse'][0]:.4f}, {holdout_ci['rmse'][1]:.4f}]")
    print(f"MAE:  {holdout_metrics['mae']:.4f} | 95% CI: [{holdout_ci['mae'][0]:.4f}, {holdout_ci['mae'][1]:.4f}]")
    print(f"R2:   {holdout_metrics['r2']:.4f} | 95% CI: [{holdout_ci['r2'][0]:.4f}, {holdout_ci['r2'][1]:.4f}]")
    print(f"COS_MAE:  {cos['cos_mae']:.4f} | 95% CI: [{cos['cos_mae_ci'][0]:.4f}, {cos['cos_mae_ci'][1]:.4f}]")
    print(f"COS_RMSE: {cos['cos_rmse']:.4f} | 95% CI: [{cos['cos_rmse_ci'][0]:.4f}, {cos['cos_rmse_ci'][1]:.4f}]")
    print(f"COS_R2:   {cos['cos_r2']:.4f} | 95% CI: [{cos['cos_r2_ci'][0]:.4f}, {cos['cos_r2_ci'][1]:.4f}]")

    # Driver-filtered standalone LSTM does NOT export any hybrid baseline predictions:
    # the LSTM Hybrid keeps consuming the full-circuit LR-EW/XGB-EW predictions only.


if __name__ == "__main__":
    main()
