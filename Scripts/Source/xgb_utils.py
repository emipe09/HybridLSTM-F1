"""Shared XGBoost utilities for sliding-window and expanding-window modeling scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modeling_utils import align_one_hot, json_ready


BASE_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "nthread": -1,
}

XGB_SEARCH_SPACE_VERSION = "gp_final_random_v1"

DEFAULT_XGB_SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.10, "log": True},
    "max_depth": {"type": "int", "low": 2, "high": 6},
    "min_child_weight": {"type": "int", "low": 5, "high": 30},
    "subsample": {"type": "float", "low": 0.6, "high": 0.9},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 0.9},
    "gamma": {"type": "float", "low": 0.2, "high": 8.0},
    "reg_alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 0.1, "high": 30.0, "log": True},
}

XGB_SEARCH_SPACE_CONFIG_KEYS = {
    "learning_rate": ("xgb_learning_rate_min", "xgb_learning_rate_max"),
    "max_depth": ("xgb_max_depth_min", "xgb_max_depth_max"),
    "min_child_weight": ("xgb_min_child_weight_min", "xgb_min_child_weight_max"),
    "subsample": ("xgb_subsample_min", "xgb_subsample_max"),
    "colsample_bytree": ("xgb_colsample_bytree_min", "xgb_colsample_bytree_max"),
    "gamma": ("xgb_gamma_min", "xgb_gamma_max"),
    "reg_alpha": ("xgb_reg_alpha_min", "xgb_reg_alpha_max"),
    "reg_lambda": ("xgb_reg_lambda_min", "xgb_reg_lambda_max"),
}


def build_xgb_search_space(config):
    search_space = {name: spec.copy() for name, spec in DEFAULT_XGB_SEARCH_SPACE.items()}
    for param_name, (low_key, high_key) in XGB_SEARCH_SPACE_CONFIG_KEYS.items():
        if low_key in config:
            search_space[param_name]["low"] = config[low_key]
        if high_key in config:
            search_space[param_name]["high"] = config[high_key]

    for param_name, spec in search_space.items():
        if spec["low"] > spec["high"]:
            raise ValueError(f"Invalid XGBoost search space for {param_name}: low > high.")
    return search_space


def suggest_xgb_params(trial, search_space):
    params = {}
    for param_name, spec in search_space.items():
        if spec["type"] == "int":
            params[param_name] = trial.suggest_int(param_name, int(spec["low"]), int(spec["high"]))
        else:
            params[param_name] = trial.suggest_float(
                param_name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
    return params


def build_optuna_sampler(config):
    sampler_name = str(config.get("xgb_optuna_sampler", "tpe")).lower()
    seed = int(config["random_seed"])
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    raise ValueError(f"Unsupported xgb_optuna_sampler={sampler_name!r}. Use 'random' or 'tpe'.")


def build_xgb_matrix(X_train, X_eval, y_train, y_eval, cat_cols):
    X_train_enc, X_eval_enc = align_one_hot(X_train, X_eval, cat_cols, drop_first=False)
    medians = X_train_enc.median(numeric_only=True)
    X_train_enc = X_train_enc.fillna(medians)
    X_eval_enc = X_eval_enc.fillna(medians)
    dtrain = xgb.DMatrix(X_train_enc, label=y_train)
    deval = xgb.DMatrix(X_eval_enc, label=y_eval)
    return dtrain, deval, X_train_enc, X_eval_enc


def save_per_window_trials(trial_rows, trials_path):
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).to_csv(trials_path, index=False)


def window_train_params(window_summary, config):
    return {**BASE_XGB_PARAMS, **window_summary["params"], "seed": int(config["random_seed"])}


def median_n_estimators(window_summaries):
    return max(1, int(round(float(np.median([item["n_estimators"] for item in window_summaries])))))


def aggregate_window_params(window_summaries, search_space):
    aggregated_params = {}
    source_values = {}
    for param_name, spec in search_space.items():
        values = [item["params"][param_name] for item in window_summaries]
        source_values[param_name] = [json_ready(value) for value in values]
        median_value = float(np.median(values))
        if spec["type"] == "int":
            aggregated_params[param_name] = int(round(median_value))
        else:
            aggregated_params[param_name] = median_value
    return aggregated_params, source_values


def build_holdout_block_diagnostics(holdout_laps, y_true, y_pred, block_lap_size):
    diagnostic_frame = pd.DataFrame(
        {
            "lap": pd.to_numeric(holdout_laps, errors="coerce").to_numpy(),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        }
    ).dropna(subset=["lap", "y_true", "y_pred"])

    if diagnostic_frame.empty:
        return []

    block_lap_size = max(1, int(block_lap_size))
    unique_laps = np.sort(diagnostic_frame["lap"].unique())
    block_results = []
    for block_id, start in enumerate(range(0, len(unique_laps), block_lap_size), start=1):
        block_laps = unique_laps[start : start + block_lap_size]
        block = diagnostic_frame[diagnostic_frame["lap"].isin(block_laps)]
        y_block = block["y_true"].to_numpy()
        pred_block = block["y_pred"].to_numpy()
        residuals = y_block - pred_block
        r2_value = float(r2_score(y_block, pred_block)) if len(block) > 1 else None
        if r2_value is not None and not np.isfinite(r2_value):
            r2_value = None
        block_results.append(
            {
                "block": block_id,
                "lap_start": int(block_laps[0]),
                "lap_end": int(block_laps[-1]),
                "records": int(len(block)),
                "rmse": float(np.sqrt(mean_squared_error(y_block, pred_block))),
                "mae": float(mean_absolute_error(y_block, pred_block)),
                "r2": r2_value,
                "residual_std": float(np.std(residuals, ddof=1)) if len(block) > 1 else 0.0,
            }
        )
    return block_results
