"""Linear Regression with expanding-window validation and sequential holdout."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modeling_utils import (
    build_lr_ew_model_paths,
    build_sequential_split,
    build_expanding_windows,
    calc_holdout_ci,
    calc_stats,
    decode_step_key,
    fit_predict_linear_regression,
    log_mlflow_run,
    load_cleaned_data,
    prepare_raw_features,
    select_modeling_columns,
    summarize_cos,
)


def main():
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()

    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    num_cols, cat_cols = select_modeling_columns(df_base, config)
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    print("--- LINEAR REGRESSION: EXPANDING WINDOW + SEQUENTIAL HOLDOUT ---")
    print(f"Grand Prix: {target_gp_name}")
    print(
        "Config: "
        f"holdout={config['holdout_ratio']} | window={config.get('lr_ew_window_ratio', config['window_ratio'])} | "
        f"window_train={config['window_train_ratio']} | step={config['window_step_ratio']} | "
        f"alpha_cos={config['alpha_cos']} | beta_cos={config['beta_cos']}"
    )
    print(f"Numerical features: {num_cols}")
    print(f"Categorical features: {cat_cols}")

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

    windows, window_size, train_size, val_size, step_size = build_expanding_windows(
        len(unique_laps),
        float(config.get("lr_ew_window_ratio", config["window_ratio"])),
        float(config["window_train_ratio"]),
        float(config["window_step_ratio"]),
    )

    print("\n--- Sequential split ---")
    print(f"Total temporal steps: {total_laps} ({decode_step_key(lap_min)} → {decode_step_key(lap_max)})")
    print(f"Modeling block: {decode_step_key(lap_min)}–{decode_step_key(model_end_lap)} | records={len(X_model_raw)}")
    print(f"Holdout block: {decode_step_key(holdout_start_lap)}–{decode_step_key(lap_max)} | records={len(X_holdout_raw)}")
    print(
        f"Expanding windows: {len(windows)} | initial_train={train_size} | "
        f"val_chunk={val_size} | step={step_size}"
    )
    print("NOTE: expanding windows have growing training sets; confidence intervals are descriptive.")

    results = {"window": [], "rmse": [], "mae": [], "r2": [], "std": []}

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

        preds, *_ = fit_predict_linear_regression(X_train, y_train, X_val, cat_cols)

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
            f"Fold {i:02d} | train {decode_step_key(train_laps[0])}–{decode_step_key(train_laps[-1])} ({len(X_train)} recs) | "
            f"val {decode_step_key(val_laps[0])}–{decode_step_key(val_laps[-1])} | "
            f"RMSE={rmse_value:.4f} | MAE={mae_value:.4f} | R2={r2_value:.4f}"
        )

    rmse_m, rmse_l, rmse_u = calc_stats(results["rmse"])
    mae_m, mae_l, mae_u = calc_stats(results["mae"])
    r2_m, r2_l, r2_u = calc_stats(results["r2"])
    std_m, _, _ = calc_stats(results["std"])

    preds_holdout, final_model, _, _, feature_names = fit_predict_linear_regression(
        X_model_raw, y_model, X_holdout_raw, cat_cols
    )

    lr_model_path, lr_model_metadata_path = build_lr_ew_model_paths(repo_root, config)
    lr_model_path.parent.mkdir(parents=True, exist_ok=True)
    lr_payload = {
        "model": final_model,
        "feature_names": list(feature_names),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "target_col": target_col,
        "lap_col": lap_col,
        "training_block": "first_sequential_modeling_block",
        "validation_protocol": "expanding_window",
        "preprocessing": "median_imputer_standard_scaler_one_hot_drop_first",
    }
    with lr_model_path.open("wb") as file:
        pickle.dump(lr_payload, file)
    lr_model_metadata = {
        "target_gp_name": target_gp_name,
        "model": "linear_regression",
        "validation_protocol": "expanding_window",
        "model_path": str(lr_model_path),
        "target_col": target_col,
        "lap_col": lap_col,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "encoded_feature_names": list(feature_names),
        "training_block": "first_sequential_modeling_block",
        "holdout_usage": "final sequential holdout is not used for training",
        "preprocessing": "median_imputer_standard_scaler_one_hot_drop_first",
    }
    lr_model_metadata_path.write_text(json.dumps(lr_model_metadata, indent=2), encoding="utf-8")
    print(f"Saved final Linear Regression EW model to: {lr_model_path}")
    print(f"Saved final Linear Regression EW model metadata to: {lr_model_metadata_path}")

    holdout_ci = calc_holdout_ci(y_holdout.to_numpy(), preds_holdout, seed=int(config["random_seed"]))
    rmse_holdout = float(np.sqrt(mean_squared_error(y_holdout, preds_holdout)))
    mae_holdout = float(mean_absolute_error(y_holdout, preds_holdout))
    r2_holdout = float(r2_score(y_holdout, preds_holdout))
    std_holdout = float(np.std(np.asarray(y_holdout) - np.asarray(preds_holdout), ddof=1)) if len(y_holdout) > 1 else 0.0

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
        "total_temporal_steps": total_laps,
        "step_min": decode_step_key(lap_min),
        "step_max": decode_step_key(lap_max),
        "model_end_step": decode_step_key(model_end_lap),
        "holdout_start_step": decode_step_key(holdout_start_lap),
        "model_records": len(X_model_raw),
        "holdout_records": len(X_holdout_raw),
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
    }
    log_mlflow_run(
        repo_root,
        config,
        "linear_regression",
        num_cols,
        cat_cols,
        split_info,
        results,
        summary_metrics,
        extra_params={"preprocessing": "median_imputer_standard_scaler_one_hot_drop_first"},
        artifacts=[lr_model_path, lr_model_metadata_path],
        validation_mode="ew",
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
    print(f"          MAE final/EW={cos['mae_final']:.4f}/{cos['mae_sw']:.4f} | STD final/EW={cos['std_final']:.4f}/{cos['std_sw']:.4f}")
    print(f"COS_RMSE: {cos['cos_rmse']:.4f} | 95% CI: [{cos['cos_rmse_ci'][0]:.4f}, {cos['cos_rmse_ci'][1]:.4f}]")
    print(f"          RMSE final/EW={cos['rmse_final']:.4f}/{cos['rmse_sw']:.4f} | STD final/EW={cos['std_final']:.4f}/{cos['std_sw']:.4f}")
    print(f"COS_R2:   {cos['cos_r2']:.4f} | 95% CI: [{cos['cos_r2_ci'][0]:.4f}, {cos['cos_r2_ci'][1]:.4f}]")
    print(f"          R2 final/EW={cos['r2_final']:.4f}/{cos['r2_sw']:.4f} | STD final/EW={cos['std_final']:.4f}/{cos['std_sw']:.4f}")

    print("\n--- Final model coefficients ---")
    coefs = pd.Series(final_model.coef_, index=feature_names)
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(20).to_frame("coefficient"))

    # Persist per-row baseline predictions (OOF + holdout) so the hybrid model can reuse
    # this exact baseline instead of regenerating it. Auxiliary and non-destructive: a
    # failure here must not affect the reported LR-EW artifacts/metrics above.
    try:
        from baseline_utils import export_baseline_predictions

        export_baseline_predictions(
            repo_root, config, "lr_ew", df_base, num_cols, cat_cols, target_col, lap_col
        )
    except Exception as exc:  # pragma: no cover - export is best-effort
        print(f"WARNING: could not export LR-EW baseline predictions: {exc}")


if __name__ == "__main__":
    main()
