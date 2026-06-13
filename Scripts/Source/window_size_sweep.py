"""Window size sensitivity sweep across SW and EW validation for LR and XGBoost.

For each window size in the configured range, runs Optuna hyperparameter tuning
for XGBoost using the windows of that size and scheme (SW or EW). Each combination
of (window_size, scheme) gets its own tuned hyperparameters, saved to a JSON file
and used for both CV evaluation and the final holdout model.

LR has no hyperparameters — a fresh OLS model is fitted per window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modeling_utils import (
    build_expanding_windows,
    build_sequential_split,
    build_sliding_windows,
    calc_holdout_ci,
    calc_stats,
    fit_predict_linear_regression,
    json_ready,
    load_cleaned_data,
    prepare_raw_features,
    resolve_repo_path,
    safe_gp_name,
    select_modeling_columns,
    summarize_cos,
)
from xgb_utils import (
    BASE_XGB_PARAMS,
    XGB_SEARCH_SPACE_VERSION,
    aggregate_window_params,
    build_optuna_sampler,
    build_xgb_matrix,
    build_xgb_search_space,
    median_n_estimators,
    suggest_xgb_params,
    window_train_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Objective factory (avoids closure-over-loop-variable bug)
# ---------------------------------------------------------------------------

def _make_objective(X_train, y_train, X_val, y_val, cat_cols, config, search_space):
    def objective(trial):
        params = {
            **BASE_XGB_PARAMS,
            "seed": int(config["random_seed"]),
            **suggest_xgb_params(trial, search_space),
        }
        dtrain, dval, _, _ = build_xgb_matrix(X_train, X_val, y_train, y_val, cat_cols)
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=5000,
            evals=[(dval, "validation")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        best_iter = booster.best_iteration + 1
        preds = booster.predict(dval, iteration_range=(0, best_iter))
        trial.set_user_attr("n_estimators", best_iter)
        return float(np.sqrt(mean_squared_error(y_val, preds)))
    return objective


# ---------------------------------------------------------------------------
# Per-scheme Optuna tuning
# ---------------------------------------------------------------------------

def _tune_xgb(windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols, config, label):
    """Tune XGBoost hyperparameters on a set of windows (SW or EW).

    Returns (train_params, best_n, window_summaries) or (None, None, None) if
    all windows are empty.
    """
    search_space = build_xgb_search_space(config)
    optuna_trials = int(config["optuna_trials"])

    window_summaries = []
    for fold_id, (start, split, end) in enumerate(windows, start=1):
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        X_train = X_model_raw.loc[train_mask]
        y_train = y_model.loc[train_mask]
        X_val = X_model_raw.loc[val_mask]
        y_val = y_model.loc[val_mask]
        if len(X_train) == 0 or len(X_val) == 0:
            continue

        study = optuna.create_study(
            direction="minimize", sampler=build_optuna_sampler(config)
        )
        study.optimize(
            _make_objective(X_train, y_train, X_val, y_val, cat_cols, config, search_space),
            n_trials=optuna_trials,
            show_progress_bar=False,
        )
        best = study.best_trial
        window_summaries.append({
            "window": fold_id,
            "n_estimators": int(best.user_attrs["n_estimators"]),
            "params": {k: json_ready(v) for k, v in best.params.items()},
            "rmse": float(best.value),
        })

    if not window_summaries:
        return None, None, None

    aggregated_params, aggregated_source_values = aggregate_window_params(window_summaries, search_space)
    best_n = median_n_estimators(window_summaries)
    agg_summary = {"window": "aggregated", "params": aggregated_params}
    train_params = window_train_params(agg_summary, config)
    return train_params, best_n, window_summaries


def _save_xgb_params(params_path: Path, train_params, best_n, window_summaries, search_space, config, scheme, window_ratio):
    payload = {
        **{k: v for k, v in train_params.items() if k not in BASE_XGB_PARAMS},
        "n_estimators": best_n,
        "search_space_version": XGB_SEARCH_SPACE_VERSION,
        "tuning_strategy": f"window_sweep_{scheme}_per_fold_median_v1",
        "window_ratio": window_ratio,
        "scheme": scheme,
        "search_space": build_xgb_search_space(config),
        "optuna_sampler": str(config.get("xgb_optuna_sampler", "tpe")).lower(),
        "per_fold_params": window_summaries,
    }
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4)


# ---------------------------------------------------------------------------
# Window evaluation helpers
# ---------------------------------------------------------------------------

def _run_lr_windows(windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols):
    results = {"rmse": [], "mae": [], "r2": [], "std": []}
    for start, split, end in windows:
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        X_train, y_train = X_model_raw.loc[train_mask], y_model.loc[train_mask]
        X_val, y_val = X_model_raw.loc[val_mask], y_model.loc[val_mask]
        if len(X_train) == 0 or len(X_val) == 0:
            continue
        preds, *_ = fit_predict_linear_regression(X_train, y_train, X_val, cat_cols)
        residuals = np.asarray(y_val) - np.asarray(preds)
        results["rmse"].append(float(np.sqrt(mean_squared_error(y_val, preds))))
        results["mae"].append(float(mean_absolute_error(y_val, preds)))
        results["r2"].append(float(r2_score(y_val, preds)))
        results["std"].append(float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0)
    return results


def _run_xgb_windows(windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols, train_params, best_n):
    results = {"rmse": [], "mae": [], "r2": [], "std": []}
    for start, split, end in windows:
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        X_train, y_train = X_model_raw.loc[train_mask], y_model.loc[train_mask]
        X_val, y_val = X_model_raw.loc[val_mask], y_model.loc[val_mask]
        if len(X_train) == 0 or len(X_val) == 0:
            continue
        dtrain, dval, _, _ = build_xgb_matrix(X_train, X_val, y_train, y_val, cat_cols)
        booster = xgb.train(params=train_params, dtrain=dtrain, num_boost_round=best_n, verbose_eval=False)
        preds = booster.predict(dval)
        residuals = np.asarray(y_val) - np.asarray(preds)
        results["rmse"].append(float(np.sqrt(mean_squared_error(y_val, preds))))
        results["mae"].append(float(mean_absolute_error(y_val, preds)))
        results["r2"].append(float(r2_score(y_val, preds)))
        results["std"].append(float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0)
    return results


def _aggregate_results(
    window_results, y_holdout, preds_holdout, seed, alpha_cos, beta_cos, label, window_ratio,
    params_path=None, best_n=None,
):
    if not window_results["rmse"]:
        return None

    rmse_m, rmse_l, rmse_u = calc_stats(window_results["rmse"])
    mae_m, mae_l, mae_u = calc_stats(window_results["mae"])
    r2_m, r2_l, r2_u = calc_stats(window_results["r2"])
    std_m, _, _ = calc_stats(window_results["std"])
    sample_std_rmse = float(np.std(window_results["rmse"], ddof=1)) if len(window_results["rmse"]) > 1 else 0.0
    sample_std_mae = float(np.std(window_results["mae"], ddof=1)) if len(window_results["mae"]) > 1 else 0.0

    holdout_ci = calc_holdout_ci(np.asarray(y_holdout), np.asarray(preds_holdout), seed=seed)
    rmse_holdout = float(np.sqrt(mean_squared_error(y_holdout, preds_holdout)))
    mae_holdout = float(mean_absolute_error(y_holdout, preds_holdout))
    r2_holdout = float(r2_score(y_holdout, preds_holdout))
    std_holdout = float(np.std(np.asarray(y_holdout) - np.asarray(preds_holdout), ddof=1)) if len(y_holdout) > 1 else 0.0

    cos = summarize_cos(
        window_results, mae_m, rmse_m, mae_holdout, rmse_holdout, std_m, std_holdout, alpha_cos, beta_cos
    )

    return {
        "window_ratio": window_ratio,
        "validation": label,
        "n_folds": len(window_results["rmse"]),
        "xgb_n_estimators": best_n,
        "tuned_params_path": str(params_path) if params_path else None,
        "ew_or_sw_rmse_mean": rmse_m,
        "ew_or_sw_rmse_ci_lower": rmse_l,
        "ew_or_sw_rmse_ci_upper": rmse_u,
        "ew_or_sw_mae_mean": mae_m,
        "ew_or_sw_mae_ci_lower": mae_l,
        "ew_or_sw_mae_ci_upper": mae_u,
        "ew_or_sw_r2_mean": r2_m,
        "ew_or_sw_r2_ci_lower": r2_l,
        "ew_or_sw_r2_ci_upper": r2_u,
        "ew_or_sw_residual_std_mean": std_m,
        "sample_std_rmse": sample_std_rmse,
        "sample_std_mae": sample_std_mae,
        "holdout_rmse": rmse_holdout,
        "holdout_rmse_ci_lower": holdout_ci["rmse"][0],
        "holdout_rmse_ci_upper": holdout_ci["rmse"][1],
        "holdout_mae": mae_holdout,
        "holdout_mae_ci_lower": holdout_ci["mae"][0],
        "holdout_mae_ci_upper": holdout_ci["mae"][1],
        "holdout_r2": r2_holdout,
        "holdout_r2_ci_lower": holdout_ci["r2"][0],
        "holdout_r2_ci_upper": holdout_ci["r2"][1],
        "holdout_residual_std": std_holdout,
        "cos_mae": cos["cos_mae"],
        "cos_mae_ci_lower": cos["cos_mae_ci"][0],
        "cos_mae_ci_upper": cos["cos_mae_ci"][1],
        "cos_rmse": cos["cos_rmse"],
        "cos_rmse_ci_lower": cos["cos_rmse_ci"][0],
        "cos_rmse_ci_upper": cos["cos_rmse_ci"][1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()

    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    num_cols, cat_cols = select_modeling_columns(df_base, config)
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    sweep_min = float(config.get("window_size_sweep_min", 0.05))
    sweep_max = float(config.get("window_size_sweep_max", 0.55))
    sweep_step = float(config.get("window_size_sweep_step", 0.05))
    alpha_cos = float(config["alpha_cos"])
    beta_cos = float(config["beta_cos"])
    seed = int(config["random_seed"])

    safe_name = safe_gp_name(target_gp_name)
    sweep_results_dir = resolve_repo_path(
        repo_root, str(config.get("window_sweep_results_dir", "Scripts/Results/window_sweep"))
    )
    params_dir = resolve_repo_path(
        repo_root, str(config.get("window_sweep_params_dir", "Scripts/Results/window_sweep/params"))
    )
    filename_template = str(
        config.get("window_sweep_results_filename_template", "{safe_gp_name}_window_sweep_results.csv")
    )
    sweep_output_path = sweep_results_dir / filename_template.format(
        target_gp_name=target_gp_name, safe_gp_name=safe_name
    )

    print("--- WINDOW SIZE SWEEP (with per-combo Optuna tuning) ---")
    print(f"Grand Prix: {target_gp_name}")
    print(f"Sweep range: {sweep_min:.0%} to {sweep_max:.0%} in steps of {sweep_step:.0%}")
    print(f"Output: {sweep_output_path}")
    print(f"Params dir: {params_dir}")

    if sweep_output_path.exists():
        print(f"WARNING: Output file already exists: {sweep_output_path}")
        print("Results will be overwritten.")

    (
        lap_series, lap_min, lap_max,
        model_idx, holdout_idx,
        holdout_start_lap, model_end_lap, total_laps,
    ) = build_sequential_split(df_base, valid_indices, float(config["holdout_ratio"]), lap_col)

    X_model_raw = X_raw.loc[model_idx].copy()
    y_model = y_raw.loc[model_idx].copy()
    X_holdout_raw = X_raw.loc[holdout_idx].copy()
    y_holdout = y_raw.loc[holdout_idx].copy()

    model_laps = lap_series.loc[model_idx]
    model_order_idx = model_laps.sort_values(kind="mergesort").index
    X_model_raw = X_model_raw.loc[model_order_idx].reset_index(drop=True)
    y_model = y_model.loc[model_order_idx].reset_index(drop=True)
    lap_model_sorted = model_laps.loc[model_order_idx].reset_index(drop=True)
    unique_laps = np.sort(pd.to_numeric(lap_model_sorted, errors="coerce").dropna().unique())

    # LR holdout is fixed (same model regardless of window size)
    preds_holdout_lr, *_ = fit_predict_linear_regression(
        X_model_raw, y_model, X_holdout_raw, cat_cols
    )

    n_steps = round((sweep_max - sweep_min) / sweep_step)
    window_ratios = [round(sweep_min + i * sweep_step, 10) for i in range(n_steps + 1)]
    window_ratios = [r for r in window_ratios if r <= sweep_max + 1e-9]

    train_ratio = float(config["window_train_ratio"])
    step_ratio = float(config["window_step_ratio"])
    all_rows = []

    for window_ratio in window_ratios:
        print(f"\n--- window_ratio={window_ratio:.2%} ---")

        try:
            sw_windows, _, _, _, _ = build_sliding_windows(
                len(unique_laps), window_ratio, train_ratio, step_ratio
            )
        except ValueError as err:
            print(f"  SW skipped: {err}")
            sw_windows = []
        try:
            ew_windows, _, _, _, _ = build_expanding_windows(
                len(unique_laps), window_ratio, train_ratio, step_ratio
            )
        except ValueError as err:
            print(f"  EW skipped: {err}")
            ew_windows = []

        window_pct = int(round(window_ratio * 100))

        # ---- LR (no tuning) ------------------------------------------------
        for scheme, windows in [("sw", sw_windows), ("ew", ew_windows)]:
            if not windows:
                continue
            label = f"lr_{scheme}"
            print(f"  {label}: {len(windows)} folds", end=" | ", flush=True)
            w_results = _run_lr_windows(
                windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols
            )
            row = _aggregate_results(
                w_results, y_holdout, preds_holdout_lr, seed,
                alpha_cos, beta_cos, label, window_ratio,
            )
            if row:
                row["model"] = "lr"
                all_rows.append(row)
                print(
                    f"RMSE_folds={row['ew_or_sw_rmse_mean']:.4f} | "
                    f"RMSE_holdout={row['holdout_rmse']:.4f} | "
                    f"COS_RMSE={row['cos_rmse']:.4f}"
                )

        # ---- XGB (Optuna tuning per scheme) --------------------------------
        for scheme, windows in [("sw", sw_windows), ("ew", ew_windows)]:
            if not windows:
                continue
            label = f"xgb_{scheme}"
            params_path = params_dir / f"{safe_name}_xgb_{scheme}_{window_pct}pct_params.json"

            print(
                f"  {label}: {len(windows)} folds | tuning with {config['optuna_trials']} Optuna trials ...",
                flush=True,
            )
            train_params, best_n, window_summaries = _tune_xgb(
                windows, unique_laps, lap_model_sorted,
                X_model_raw, y_model, cat_cols, config, label,
            )
            if train_params is None:
                print(f"  {label}: skipped (no valid folds after tuning)")
                continue

            _save_xgb_params(
                params_path, train_params, best_n, window_summaries,
                build_xgb_search_space(config), config, scheme, window_ratio,
            )

            # Evaluate CV with tuned params
            w_results = _run_xgb_windows(
                windows, unique_laps, lap_model_sorted,
                X_model_raw, y_model, cat_cols, train_params, best_n,
            )

            # Holdout with tuned params (trained on full modeling block)
            dmodel_full, dholdout, _, _ = build_xgb_matrix(
                X_model_raw, X_holdout_raw, y_model, y_holdout, cat_cols
            )
            final_model = xgb.train(
                params=train_params, dtrain=dmodel_full,
                num_boost_round=best_n, verbose_eval=False,
            )
            preds_holdout_xgb = final_model.predict(dholdout)

            row = _aggregate_results(
                w_results, y_holdout, preds_holdout_xgb, seed,
                alpha_cos, beta_cos, label, window_ratio,
                params_path=params_path, best_n=best_n,
            )
            if row:
                row["model"] = "xgb"
                all_rows.append(row)
                print(
                    f"  {label}: RMSE_folds={row['ew_or_sw_rmse_mean']:.4f} | "
                    f"RMSE_holdout={row['holdout_rmse']:.4f} | "
                    f"COS_RMSE={row['cos_rmse']:.4f} | "
                    f"n_est={best_n}"
                )

    if not all_rows:
        print("No results generated. Check window size sweep configuration.")
        return

    sweep_results_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_rows)
    col_order = (
        ["window_ratio", "validation", "model", "n_folds", "xgb_n_estimators", "tuned_params_path"]
        + [c for c in results_df.columns if c not in {
            "window_ratio", "validation", "model", "n_folds", "xgb_n_estimators", "tuned_params_path"
        }]
    )
    col_order = [c for c in col_order if c in results_df.columns]
    results_df = results_df[col_order]
    results_df.to_csv(sweep_output_path, index=False)
    print(f"\nSaved sweep results to: {sweep_output_path}")
    print(f"Total rows: {len(results_df)} ({len(window_ratios)} window sizes × up to 4 combos)")


if __name__ == "__main__":
    main()
