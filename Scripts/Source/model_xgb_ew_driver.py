"""Driver-filtered XGBoost with Optuna, expanding-window validation, and sequential holdout.

Complementary methodological baseline (NOT a replacement for model_xgb_ew.py). The dataset is
filtered to a single ``Driver`` (chosen with ``--driver``) before the temporal split, bringing
the learning unit closer to the per-driver trajectory logic of the LSTM Hybrid. Everything else
mirrors the main XGB-EW protocol: ordering Year -> LapNumber, per-fold Optuna/TPE tuning inside
the first 80% modeling block with median aggregation across folds, sequential 20% holdout, the
circuit-specific search space, and the same metrics.

It reuses ``tune_or_load_params_ew`` from model_xgb_ew.py unchanged, so the tuning strategy is
identical. Only the artifact paths differ: params/trials/model/metadata go to dedicated
``ew_driver`` subdirectories keyed by both circuit AND driver, so the filtered run never
overwrites the main XGB-EW artifacts and never reuses the full-circuit saved parameters (the
filtered dataset is different). After the filter ``Driver`` is constant, so it is dropped from
the categorical features. This script deliberately does NOT export hybrid baseline predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from modeling_utils import (
    build_expanding_windows,
    build_sequential_split,
    calc_holdout_ci,
    calc_stats,
    decode_step_key,
    log_mlflow_run,
    load_cleaned_data,
    prepare_raw_features,
    resolve_repo_path,
    safe_gp_name,
    select_modeling_columns,
    summarize_cos,
)
from xgb_utils import (
    build_holdout_block_diagnostics,
    build_xgb_matrix,
    window_train_params,
)
from model_xgb_ew import build_xgb_ew_trials_path, tune_or_load_params_ew
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--driver", required=True, help="Driver code to filter on (e.g. VER, NOR, HAM).")
    return parser.parse_args()


def build_xgb_ew_driver_params_path(repo_root, config, safe_driver):
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    filename = str(
        config.get("ew_driver_xgb_params_filename_template", "{safe_gp_name}_{safe_driver}_xgb_params_ew_driver.json")
    ).format(target_gp_name=target_gp_name, safe_gp_name=safe_name, safe_driver=safe_driver)
    subdir = str(config.get("ew_driver_xgb_params_subdir", "xgboost/ew_driver/params"))
    return resolve_repo_path(repo_root, str(config["results_dir"])) / subdir / filename


def build_xgb_ew_driver_model_paths(repo_root, config, safe_driver):
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    model_filename = str(
        config.get("ew_driver_xgb_model_filename_template", "{safe_gp_name}_{safe_driver}_xgb_model_ew_driver.json")
    ).format(target_gp_name=target_gp_name, safe_gp_name=safe_name, safe_driver=safe_driver)
    metadata_filename = str(
        config.get(
            "ew_driver_xgb_model_metadata_filename_template",
            "{safe_gp_name}_{safe_driver}_xgb_model_ew_driver_metadata.json",
        )
    ).format(target_gp_name=target_gp_name, safe_gp_name=safe_name, safe_driver=safe_driver)
    subdir = str(config.get("ew_driver_xgb_models_subdir", "xgboost/ew_driver/models"))
    model_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / subdir
    return model_dir / model_filename, model_dir / metadata_filename


def main():
    args = parse_args()
    driver = args.driver.strip().upper()
    safe_driver = driver.lower()

    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()

    if "Driver" not in df_base.columns:
        print("SKIP: cleaned dataset has no 'Driver' column; cannot run the driver-filtered baseline.")
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
    num_cols, cat_cols = select_modeling_columns(df_base, config)
    # Driver is constant after the filter -> drop it (zero information, documented in metadata).
    dropped_constant_features = [c for c in cat_cols if c == "Driver"]
    cat_cols = [c for c in cat_cols if c != "Driver"]
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    window_ratio = float(config.get("xgb_ew_window_ratio", config["window_ratio"]))

    print("--- XGBOOST (DRIVER-FILTERED): OPTUNA + EXPANDING WINDOW + SEQUENTIAL HOLDOUT ---")
    print(f"Grand Prix: {target_gp_name} | Driver: {driver}")
    print(f"Filtered records: {filtered_record_count}")
    print(
        "Config: "
        f"holdout={config['holdout_ratio']} | window={window_ratio} | "
        f"window_train={config['window_train_ratio']} | step={config['window_step_ratio']} | "
        f"alpha_cos={config['alpha_cos']} | beta_cos={config['beta_cos']} | "
        f"optuna_trials={config['optuna_trials']}"
    )
    print(f"Numerical features: {num_cols}")
    print(f"Categorical features: {cat_cols} (dropped constant: {dropped_constant_features})")

    try:
        (
            lap_series,
            lap_min,
            lap_max,
            model_idx,
            holdout_idx,
            holdout_start_lap,
            model_end_lap,
            total_laps,
        ) = build_sequential_split(df_base, valid_indices, float(config["holdout_ratio"]), lap_col)
    except ValueError as exc:
        print(
            f"SKIP: driver={driver!r} in {target_gp_name} cannot form a sequential split "
            f"(filtered_records={filtered_record_count}): {exc}. No artifacts generated."
        )
        sys.exit(0)

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
    n_unique_temporal_laps = len(unique_laps)

    try:
        windows, window_size, train_size, val_size, step_size = build_expanding_windows(
            n_unique_temporal_laps,
            window_ratio,
            float(config["window_train_ratio"]),
            float(config["window_step_ratio"]),
        )
    except ValueError as exc:
        print(
            f"SKIP: driver={driver!r} in {target_gp_name} has only {n_unique_temporal_laps} unique "
            f"temporal steps in the modeling block (filtered_records={filtered_record_count}); "
            f"insufficient to build an EW fold with window_ratio={window_ratio}: {exc}. "
            "No artifacts generated."
        )
        sys.exit(0)

    print("\n--- Sequential split ---")
    print(f"Total temporal steps: {total_laps} ({decode_step_key(lap_min)} → {decode_step_key(lap_max)})")
    print(f"Modeling block: {decode_step_key(lap_min)} – {decode_step_key(model_end_lap)} | records={len(X_model_raw)}")
    print(f"Holdout block: {decode_step_key(holdout_start_lap)} – {decode_step_key(lap_max)} | records={len(X_holdout_raw)}")
    print(
        f"Expanding folds: {len(windows)} | initial_train={train_size} | "
        f"val_chunk={val_size} | step={step_size}"
    )

    # Driver-specific params/trials paths: never reuse the full-circuit XGB-EW saved params,
    # because the filtered dataset is different. tune_or_load_params_ew is reused unchanged.
    params_path = build_xgb_ew_driver_params_path(repo_root, config, safe_driver)
    trials_path = build_xgb_ew_trials_path(params_path)
    train_params, best_n, final_params = tune_or_load_params_ew(
        params_path, windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols, config
    )

    print("\n--- Training final driver-filtered XGBoost EW model ---")
    selected_fold = final_params.get("selected_fold", {})
    if selected_fold:
        print(
            "Best individual validation fold by RMSE: "
            f"fold {selected_fold['fold']:02d} "
            f"(val {decode_step_key(selected_fold['val_lap_start'])} → {decode_step_key(selected_fold['val_lap_end'])}, "
            f"RMSE={selected_fold['rmse']:.4f}, "
            f"MAE={selected_fold['mae']:.4f})."
        )
    if final_params.get("aggregation_folds"):
        print(f"Final hyperparameters aggregated from folds: {final_params['aggregation_folds']}")

    dmodel_full, dholdout, X_model_enc, X_holdout_enc = build_xgb_matrix(
        X_model_raw, X_holdout_raw, y_model, y_holdout, cat_cols
    )
    final_model = xgb.train(params=train_params, dtrain=dmodel_full, num_boost_round=best_n, verbose_eval=False)
    print(f"Selected n_estimators: {best_n}")
    print(f"n_estimators source: {final_params.get('n_estimators_source')}")

    model_path, model_metadata_path = build_xgb_ew_driver_model_paths(repo_root, config, safe_driver)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    model_metadata = {
        "target_gp_name": target_gp_name,
        "target_driver": driver,
        "model": "xgboost",
        "validation_protocol": "expanding_window_driver_filtered",
        "baseline_note": (
            "driver-filtered methodological baseline; complementary to the main "
            "LR-EW/XGB-EW models; not a replacement"
        ),
        "model_path": str(model_path),
        "params_path": str(params_path),
        "target_col": target_col,
        "lap_col": lap_col,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "dropped_constant_features": dropped_constant_features,
        "encoded_feature_names": list(X_model_enc.columns),
        "median_imputation_values": {key: float(value) for key, value in X_model_enc.median(numeric_only=True).items()},
        "window_ratio": window_ratio,
        "filtered_record_count": int(filtered_record_count),
        "modeling_record_count": int(len(X_model_raw)),
        "holdout_record_count": int(len(X_holdout_raw)),
        "n_unique_temporal_laps": int(n_unique_temporal_laps),
        "training_block": "first_sequential_modeling_block",
        "holdout_usage": "final sequential holdout is not used for training",
        "best_n_estimators": best_n,
        "train_params": train_params,
        "final_params": final_params,
    }
    model_metadata_path.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    print(f"Saved final driver-filtered XGBoost EW model to: {model_path}")
    print(f"Saved final driver-filtered XGBoost EW model metadata to: {model_metadata_path}")

    results = {"window": [], "rmse": [], "mae": [], "r2": [], "std": []}
    per_fold_params = {
        int(fold_summary["fold"]): fold_summary
        for fold_summary in final_params.get("per_fold_params", [])
    }

    print("\n--- Expanding-window validation ---")
    for i, (start, split, end) in enumerate(windows, start=1):
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        X_train, y_train = X_model_raw.loc[train_mask], y_model.loc[train_mask]
        X_val, y_val = X_model_raw.loc[val_mask], y_model.loc[val_mask]
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(f"Fold {i}: empty train or validation set.")

        dtrain, dval, _, _ = build_xgb_matrix(X_train, X_val, y_train, y_val, cat_cols)
        if i in per_fold_params:
            fold_summary = per_fold_params[i]
            eval_params = window_train_params(fold_summary, config)
            eval_n = int(fold_summary["n_estimators"])
            booster = xgb.train(params=eval_params, dtrain=dtrain, num_boost_round=eval_n, verbose_eval=False)
            preds = booster.predict(dval)
        else:
            eval_n = best_n
            booster = xgb.train(
                params=train_params,
                dtrain=dtrain,
                num_boost_round=best_n,
                evals=[(dval, "validation")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            preds = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))

        rmse_value = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae_value = float(mean_absolute_error(y_val, preds))
        r2_value = float(r2_score(y_val, preds))
        std_value = float(np.std(np.asarray(y_val) - np.asarray(preds), ddof=1)) if len(y_val) > 1 else 0.0

        results["window"].append(i)
        results["rmse"].append(rmse_value)
        results["mae"].append(mae_value)
        results["r2"].append(r2_value)
        results["std"].append(std_value)

        print(
            f"Fold {i:02d} | train {decode_step_key(train_laps[0])} → {decode_step_key(train_laps[-1])} ({len(X_train)} records) | "
            f"val {decode_step_key(val_laps[0])} → {decode_step_key(val_laps[-1])} | "
            f"n_estimators={eval_n} | RMSE={rmse_value:.4f} | MAE={mae_value:.4f} | R2={r2_value:.4f}"
        )

    rmse_m, rmse_l, rmse_u = calc_stats(results["rmse"])
    mae_m, mae_l, mae_u = calc_stats(results["mae"])
    r2_m, r2_l, r2_u = calc_stats(results["r2"])
    std_m, _, _ = calc_stats(results["std"])

    preds_holdout = final_model.predict(dholdout)

    holdout_ci = calc_holdout_ci(y_holdout.to_numpy(), preds_holdout, seed=int(config["random_seed"]))
    rmse_holdout = float(np.sqrt(mean_squared_error(y_holdout, preds_holdout)))
    mae_holdout = float(mean_absolute_error(y_holdout, preds_holdout))
    r2_holdout = float(r2_score(y_holdout, preds_holdout))
    std_holdout = float(np.std(np.asarray(y_holdout) - np.asarray(preds_holdout), ddof=1)) if len(y_holdout) > 1 else 0.0
    holdout_block_results = build_holdout_block_diagnostics(
        lap_series.loc[holdout_idx],
        y_holdout,
        preds_holdout,
        val_size,
    )

    cos = summarize_cos(
        results,
        mae_m,
        rmse_m,
        mae_holdout,
        rmse_holdout,
        std_m,
        std_holdout,
        float(config["alpha_cos"]),
        float(config["beta_cos"]),
        r2_m=r2_m,
        r2_holdout=r2_holdout,
    )

    split_info = {
        "target_driver": driver,
        "filtered_record_count": int(filtered_record_count),
        "total_temporal_steps": total_laps,
        "step_min": decode_step_key(lap_min),
        "step_max": decode_step_key(lap_max),
        "model_end_step": decode_step_key(model_end_lap),
        "holdout_start_step": decode_step_key(holdout_start_lap),
        "model_records": len(X_model_raw),
        "holdout_records": len(X_holdout_raw),
        "n_unique_temporal_laps": int(n_unique_temporal_laps),
        "expanding_folds": len(windows),
        "initial_train_size": train_size,
        "validation_chunk_size": val_size,
    }
    summary_metrics = {
        "ew_rmse_mean": rmse_m,
        "ew_rmse_ci": (rmse_l, rmse_u),
        "ew_mae_mean": mae_m,
        "ew_mae_ci": (mae_l, mae_u),
        "ew_r2_mean": r2_m,
        "ew_r2_ci": (r2_l, r2_u),
        "ew_residual_std_mean": std_m,
        "holdout_rmse": rmse_holdout,
        "holdout_rmse_ci": holdout_ci["rmse"],
        "holdout_mae": mae_holdout,
        "holdout_mae_ci": holdout_ci["mae"],
        "holdout_r2": r2_holdout,
        "holdout_r2_ci": holdout_ci["r2"],
        "holdout_residual_std": std_holdout,
        "cos_mae": cos["cos_mae"],
        "cos_mae_ci": cos["cos_mae_ci"],
        "cos_rmse": cos["cos_rmse"],
        "cos_rmse_ci": cos["cos_rmse_ci"],
        "holdout_block_results": holdout_block_results,
    }
    log_mlflow_run(
        repo_root,
        config,
        "xgboost",
        num_cols,
        cat_cols,
        split_info,
        results,
        summary_metrics,
        extra_params={
            "target_driver": driver,
            "validation_protocol": "expanding_window_driver_filtered",
            "optuna_trials": config["optuna_trials"],
            "use_saved_xgb_params": config["use_saved_xgb_params"],
            "n_estimators": best_n,
            "n_estimators_source": final_params.get("n_estimators_source"),
            "aggregation_fold_count": final_params.get("aggregation_fold_count"),
            "aggregation_folds": ",".join(str(item) for item in final_params.get("aggregation_folds", [])),
            "search_space_version": final_params.get("search_space_version"),
            "tuning_strategy": final_params.get("tuning_strategy"),
            "selected_fold": selected_fold.get("fold"),
            "selected_fold_rmse": selected_fold.get("rmse"),
            "optuna_sampler": final_params.get("optuna_sampler"),
        },
        artifacts=[params_path, trials_path, model_path, model_metadata_path],
        validation_mode="ew_driver",
    )

    print("\n--- Expanding-window summary (indicative CI) ---")
    print("NOTE: expanding windows have growing training sets; confidence intervals are descriptive.")
    print(f"RMSE: {rmse_m:.4f} | 95% CI: [{rmse_l:.4f}, {rmse_u:.4f}]")
    print(f"MAE:  {mae_m:.4f} | 95% CI: [{mae_l:.4f}, {mae_u:.4f}]")
    print(f"R2:   {r2_m:.4f} | 95% CI: [{r2_l:.4f}, {r2_u:.4f}]")

    print("\n--- Sequential holdout ---")
    print(f"RMSE: {rmse_holdout:.4f} | 95% CI: [{holdout_ci['rmse'][0]:.4f}, {holdout_ci['rmse'][1]:.4f}]")
    print(f"MAE:  {mae_holdout:.4f} | 95% CI: [{holdout_ci['mae'][0]:.4f}, {holdout_ci['mae'][1]:.4f}]")
    print(f"R2:   {r2_holdout:.4f} | 95% CI: [{holdout_ci['r2'][0]:.4f}, {holdout_ci['r2'][1]:.4f}]")
    print(f"COS_MAE:  {cos['cos_mae']:.4f} | 95% CI: [{cos['cos_mae_ci'][0]:.4f}, {cos['cos_mae_ci'][1]:.4f}]")
    print(f"COS_RMSE: {cos['cos_rmse']:.4f} | 95% CI: [{cos['cos_rmse_ci'][0]:.4f}, {cos['cos_rmse_ci'][1]:.4f}]")
    print(f"COS_R2:   {cos['cos_r2']:.4f} | 95% CI: [{cos['cos_r2_ci'][0]:.4f}, {cos['cos_r2_ci'][1]:.4f}]")

    # Driver-filtered baseline does NOT export hybrid baseline predictions on purpose:
    # the LSTM Hybrid must keep consuming the full-circuit LR-EW/XGB-EW predictions only.


if __name__ == "__main__":
    main()
