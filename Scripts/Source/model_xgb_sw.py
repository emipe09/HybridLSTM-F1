"""XGBoost with Optuna, sliding-window validation, and sequential holdout."""

from __future__ import annotations

import json
import pandas as pd

from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import optuna
from modeling_utils import (
    align_one_hot,
    build_sequential_split,
    build_sliding_windows,
    build_xgb_model_paths,
    build_xgb_params_path,
    calc_holdout_ci,
    calc_stats,
    log_mlflow_run,
    load_cleaned_data,
    prepare_raw_features,
    select_modeling_columns,
    summarize_cos,
)


BASE_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "nthread": -1,
}


def build_xgb_matrix(X_train, X_eval, y_train, y_eval, cat_cols):
    X_train_enc, X_eval_enc = align_one_hot(X_train, X_eval, cat_cols, drop_first=False)
    medians = X_train_enc.median(numeric_only=True)
    X_train_enc = X_train_enc.fillna(medians)
    X_eval_enc = X_eval_enc.fillna(medians)

    dtrain = xgb.DMatrix(X_train_enc, label=y_train)
    deval = xgb.DMatrix(X_eval_enc, label=y_eval)
    return dtrain, deval, X_train_enc, X_eval_enc


def tune_or_load_params(params_path, windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols, config):
    if bool(config["use_saved_xgb_params"]) and params_path.exists():
        print(f"Using saved XGBoost parameters: {params_path}")
        with params_path.open("r", encoding="utf-8") as file:
            loaded_params = json.load(file)

        best_n = int(loaded_params.get("n_estimators", 0))
        if best_n < 1:
            raise ValueError("Invalid saved XGBoost parameter file: missing n_estimators.")

        train_params = {key: value for key, value in loaded_params.items() if key != "n_estimators"}
        train_params = {**BASE_XGB_PARAMS, **train_params, "seed": int(config["random_seed"])}
        return train_params, best_n, {**train_params, "n_estimators": best_n}

    def objective(trial):
        params = {
            **BASE_XGB_PARAMS,
            "seed": int(config["random_seed"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        scores = []
        for start, split, end in windows:
            train_laps = unique_laps[start:split]
            val_laps = unique_laps[split:end]
            train_mask = lap_model_sorted.isin(train_laps)
            val_mask = lap_model_sorted.isin(val_laps)
            X_train, y_train = X_model_raw.loc[train_mask], y_model.loc[train_mask]
            X_val, y_val = X_model_raw.loc[val_mask], y_model.loc[val_mask]
            if len(X_train) == 0 or len(X_val) == 0:
                raise optuna.exceptions.TrialPruned()

            dtrain, dval, _, _ = build_xgb_matrix(X_train, X_val, y_train, y_val, cat_cols)
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=2000,
                evals=[(dval, "validation")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            preds = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
            scores.append(np.sqrt(mean_squared_error(y_val, preds)))

        return float(np.mean(scores))

    optuna_trials = int(config["optuna_trials"])
    print(f"Running Optuna tuning with {optuna_trials} trials...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=int(config["random_seed"])))
    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
    train_params = {**BASE_XGB_PARAMS, **study.best_params, "seed": int(config["random_seed"])}

    print("Calibrating n_estimators through sliding-window early stopping...")
    best_iterations = []
    for start, split, end in windows:
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        dtrain, dval, _, _ = build_xgb_matrix(
            X_model_raw.loc[train_mask],
            X_model_raw.loc[val_mask],
            y_model.loc[train_mask],
            y_model.loc[val_mask],
            cat_cols,
        )
        booster = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=5000,
            evals=[(dval, "validation")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        best_iterations.append(booster.best_iteration + 1)

    best_n = max(1, int(np.median(best_iterations)))
    final_params = {**train_params, "n_estimators": best_n}
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w", encoding="utf-8") as file:
        json.dump(final_params, file, indent=4)
    print(f"Saved XGBoost parameters to: {params_path}")
    return train_params, best_n, final_params


def main():
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()

    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    num_cols, cat_cols = select_modeling_columns(df_base, config)
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    print("--- XGBOOST: OPTUNA + SLIDING WINDOW + SEQUENTIAL HOLDOUT ---")
    print(f"Grand Prix: {target_gp_name}")
    print(
        "Config: "
        f"holdout={config['holdout_ratio']} | window={config['window_ratio']} | "
        f"window_train={config['window_train_ratio']} | step={config['window_step_ratio']} | "
        f"alpha_cos={config['alpha_cos']} | beta_cos={config['beta_cos']} | "
        f"optuna_trials={config['optuna_trials']}"
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

    windows, window_size, train_size, val_size, step_size = build_sliding_windows(
        len(unique_laps),
        float(config["window_ratio"]),
        float(config["window_train_ratio"]),
        float(config["window_step_ratio"]),
    )

    print("\n--- Sequential split ---")
    print(f"Total laps: {total_laps} (LapNumber {lap_min}-{lap_max})")
    print(f"Modeling block: laps {lap_min}-{model_end_lap} | records={len(X_model_raw)}")
    print(f"Holdout block: laps {holdout_start_lap}-{lap_max} | records={len(X_holdout_raw)}")
    print(f"Sliding windows: {len(windows)} | window={window_size} | train/val={train_size}/{val_size} | step={step_size}")

    params_path = build_xgb_params_path(repo_root, config)
    train_params, best_n, final_params = tune_or_load_params(
        params_path, windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols, config
    )

    print("\n--- Training final XGBoost model ---")
    dmodel_full, dholdout, X_model_enc, X_holdout_enc = build_xgb_matrix(
        X_model_raw, X_holdout_raw, y_model, y_holdout, cat_cols
    )
    final_model = xgb.train(params=train_params, dtrain=dmodel_full, num_boost_round=best_n, verbose_eval=False)
    print(f"Selected n_estimators: {best_n}")
    print(final_params)

    model_path, model_metadata_path = build_xgb_model_paths(repo_root, config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    model_metadata = {
        "target_gp_name": target_gp_name,
        "model": "xgboost",
        "model_path": str(model_path),
        "params_path": str(params_path),
        "target_col": target_col,
        "lap_col": lap_col,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "encoded_feature_names": list(X_model_enc.columns),
        "median_imputation_values": {key: float(value) for key, value in X_model_enc.median(numeric_only=True).items()},
        "training_block": "first_sequential_modeling_block",
        "holdout_usage": "final sequential holdout is not used for training",
        "best_n_estimators": best_n,
        "train_params": train_params,
        "final_params": final_params,
    }
    model_metadata_path.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    print(f"Saved final XGBoost model to: {model_path}")
    print(f"Saved final XGBoost model metadata to: {model_metadata_path}")

    results = {"window": [], "rmse": [], "mae": [], "r2": [], "std": []}

    print("\n--- Sliding-window validation ---")
    for i, (start, split, end) in enumerate(windows, start=1):
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        X_train, y_train = X_model_raw.loc[train_mask], y_model.loc[train_mask]
        X_val, y_val = X_model_raw.loc[val_mask], y_model.loc[val_mask]
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(f"Window {i}: empty train or validation fold.")

        dtrain, dval, _, _ = build_xgb_matrix(X_train, X_val, y_train, y_val, cat_cols)
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
            f"Window {i:02d} | train laps {int(train_laps[0])}-{int(train_laps[-1])} | "
            f"val laps {int(val_laps[0])}-{int(val_laps[-1])} | "
            f"RMSE={rmse_value:.4f} | MAE={mae_value:.4f} | R2={r2_value:.4f}"
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
    )

    split_info = {
        "total_laps": total_laps,
        "lap_min": lap_min,
        "lap_max": lap_max,
        "model_end_lap": model_end_lap,
        "holdout_start_lap": holdout_start_lap,
        "model_records": len(X_model_raw),
        "holdout_records": len(X_holdout_raw),
        "sliding_windows": len(windows),
        "window_size": window_size,
        "window_train_size": train_size,
        "window_validation_size": val_size,
        "window_step_size": step_size,
    }
    summary_metrics = {
        "sw_rmse_mean": rmse_m,
        "sw_rmse_ci": (rmse_l, rmse_u),
        "sw_mae_mean": mae_m,
        "sw_mae_ci": (mae_l, mae_u),
        "sw_r2_mean": r2_m,
        "sw_r2_ci": (r2_l, r2_u),
        "sw_residual_std_mean": std_m,
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
        "xgboost",
        num_cols,
        cat_cols,
        split_info,
        results,
        summary_metrics,
        extra_params={
            "optuna_trials": config["optuna_trials"],
            "use_saved_xgb_params": config["use_saved_xgb_params"],
            "n_estimators": best_n,
            **final_params,
        },
        artifacts=[params_path, model_path, model_metadata_path],
    )

    print("\n--- Sliding-window summary (indicative CI) ---")
    print("NOTE: sliding windows overlap; these confidence intervals are descriptive.")
    print(f"RMSE: {rmse_m:.4f} | 95% CI: [{rmse_l:.4f}, {rmse_u:.4f}]")
    print(f"MAE:  {mae_m:.4f} | 95% CI: [{mae_l:.4f}, {mae_u:.4f}]")
    print(f"R2:   {r2_m:.4f} | 95% CI: [{r2_l:.4f}, {r2_u:.4f}]")

    print("\n--- Sequential holdout ---")
    print(f"RMSE: {rmse_holdout:.4f} | 95% CI: [{holdout_ci['rmse'][0]:.4f}, {holdout_ci['rmse'][1]:.4f}]")
    print(f"MAE:  {mae_holdout:.4f} | 95% CI: [{holdout_ci['mae'][0]:.4f}, {holdout_ci['mae'][1]:.4f}]")
    print(f"R2:   {r2_holdout:.4f} | 95% CI: [{holdout_ci['r2'][0]:.4f}, {holdout_ci['r2'][1]:.4f}]")
    print(f"COS_MAE:  {cos['cos_mae']:.4f} | 95% CI: [{cos['cos_mae_ci'][0]:.4f}, {cos['cos_mae_ci'][1]:.4f}]")
    print(f"          MAE SW/final={cos['mae_sw']:.4f}/{cos['mae_final']:.4f} | STD SW/final={cos['std_sw']:.4f}/{cos['std_final']:.4f}")
    print(f"COS_RMSE: {cos['cos_rmse']:.4f} | 95% CI: [{cos['cos_rmse_ci'][0]:.4f}, {cos['cos_rmse_ci'][1]:.4f}]")
    print(f"          RMSE SW/final={cos['rmse_sw']:.4f}/{cos['rmse_final']:.4f} | STD SW/final={cos['std_sw']:.4f}/{cos['std_final']:.4f}")


if __name__ == "__main__":
    main()
