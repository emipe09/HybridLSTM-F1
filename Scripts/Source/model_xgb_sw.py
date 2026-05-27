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
    json_ready,
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

XGB_SEARCH_SPACE_VERSION = "gp_final_random_v1"
XGB_TUNING_STRATEGY = "per_window_all_windows_median_params_v1"

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


def build_xgb_trials_path(params_path):
    return params_path.with_name(params_path.stem.replace("_params_sw", "_optuna_trials_sw") + ".csv")


def save_optuna_trials(study, trials_path):
    rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "objective_value": trial.value,
            "window_rmse_mean": trial.user_attrs.get("window_rmse_mean"),
            "window_rmse_std": trial.user_attrs.get("window_rmse_std"),
        }
        row.update(trial.params)
        rows.append(row)

    trials_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(trials_path, index=False)


def saved_params_match_current_config(loaded_params, search_space, sampler_name):
    return (
        loaded_params.get("search_space_version") == XGB_SEARCH_SPACE_VERSION
        and loaded_params.get("tuning_strategy") == XGB_TUNING_STRATEGY
        and loaded_params.get("search_space") == search_space
        and loaded_params.get("optuna_sampler") == sampler_name
    )


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


def tune_or_load_params(params_path, windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols, config):
    search_space = build_xgb_search_space(config)
    sampler_name = str(config.get("xgb_optuna_sampler", "tpe")).lower()
    trials_path = build_xgb_trials_path(params_path)

    if bool(config["use_saved_xgb_params"]) and params_path.exists():
        print(f"Using saved XGBoost parameters: {params_path}")
        with params_path.open("r", encoding="utf-8") as file:
            loaded_params = json.load(file)

        if not saved_params_match_current_config(loaded_params, search_space, sampler_name):
            print(
                "Saved XGBoost parameters do not match the current search configuration; "
                "running Optuna again."
            )
        else:
            selected_window = loaded_params.get("selected_window", {})
            best_n = int(loaded_params.get("n_estimators", selected_window.get("n_estimators", 0)))
            if best_n < 1:
                raise ValueError("Invalid saved XGBoost parameter file: missing n_estimators.")

            excluded_metadata = {
                "n_estimators",
                "search_space_version",
                "tuning_strategy",
                "search_space",
                "optuna_sampler",
                "selected_window",
                "per_window_params",
                "aggregation_window_count",
                "aggregation_windows",
                "aggregated_param_source_values",
                "n_estimators_source",
                "n_estimators_source_values",
            }
            train_params = {key: value for key, value in loaded_params.items() if key not in excluded_metadata}
            train_params = {**BASE_XGB_PARAMS, **train_params, "seed": int(config["random_seed"])}
            return train_params, best_n, loaded_params

    optuna_trials = int(config["optuna_trials"])
    print(f"Running per-window Optuna tuning with {optuna_trials} trials per window using {sampler_name} sampler...")
    print(f"XGBoost search space version: {XGB_SEARCH_SPACE_VERSION}")
    print(f"XGBoost tuning strategy: {XGB_TUNING_STRATEGY}")
    print(f"XGBoost search space: {search_space}")
    print(
        "Each window receives an independent Optuna study. The final holdout parameters are "
        "the median of the best Optuna parameters selected in all sliding windows; final "
        "n_estimators is the median early-stopping iteration across those windows."
    )

    window_summaries = []
    trial_rows = []
    for window_id, (start, split, end) in enumerate(windows, start=1):
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        X_train, y_train = X_model_raw.loc[train_mask], y_model.loc[train_mask]
        X_val, y_val = X_model_raw.loc[val_mask], y_model.loc[val_mask]
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(f"Window {window_id}: empty train or validation fold.")

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
            best_iteration = booster.best_iteration + 1
            preds = booster.predict(dval, iteration_range=(0, best_iteration))
            residuals = np.asarray(y_val) - np.asarray(preds)
            rmse_value = float(np.sqrt(mean_squared_error(y_val, preds)))
            mae_value = float(mean_absolute_error(y_val, preds))
            r2_value = float(r2_score(y_val, preds))
            std_value = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
            trial.set_user_attr("window", window_id)
            trial.set_user_attr("train_lap_start", int(train_laps[0]))
            trial.set_user_attr("train_lap_end", int(train_laps[-1]))
            trial.set_user_attr("val_lap_start", int(val_laps[0]))
            trial.set_user_attr("val_lap_end", int(val_laps[-1]))
            trial.set_user_attr("n_estimators", best_iteration)
            trial.set_user_attr("rmse", rmse_value)
            trial.set_user_attr("mae", mae_value)
            trial.set_user_attr("r2", r2_value)
            trial.set_user_attr("std", std_value)
            trial.set_user_attr("selection_score", rmse_value)
            return rmse_value

        print(
            f"Running Optuna for window {window_id:02d}: "
            f"train laps {int(train_laps[0])}-{int(train_laps[-1])} | "
            f"val laps {int(val_laps[0])}-{int(val_laps[-1])}"
        )
        study = optuna.create_study(direction="minimize", sampler=build_optuna_sampler(config))
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)

        for trial in study.trials:
            row = {
                "window": window_id,
                "number": trial.number,
                "state": trial.state.name,
                "objective_value": trial.value,
                "train_lap_start": trial.user_attrs.get("train_lap_start"),
                "train_lap_end": trial.user_attrs.get("train_lap_end"),
                "val_lap_start": trial.user_attrs.get("val_lap_start"),
                "val_lap_end": trial.user_attrs.get("val_lap_end"),
                "n_estimators": trial.user_attrs.get("n_estimators"),
                "rmse": trial.user_attrs.get("rmse"),
                "mae": trial.user_attrs.get("mae"),
                "r2": trial.user_attrs.get("r2"),
                "std": trial.user_attrs.get("std"),
                "selection_score": trial.user_attrs.get("selection_score"),
            }
            row.update(trial.params)
            trial_rows.append(row)

        best_trial = study.best_trial
        window_summary = {
            "window": window_id,
            "train_lap_start": int(train_laps[0]),
            "train_lap_end": int(train_laps[-1]),
            "val_lap_start": int(val_laps[0]),
            "val_lap_end": int(val_laps[-1]),
            "objective_value": float(best_trial.value),
            "rmse": float(best_trial.user_attrs["rmse"]),
            "mae": float(best_trial.user_attrs["mae"]),
            "r2": float(best_trial.user_attrs["r2"]),
            "std": float(best_trial.user_attrs["std"]),
            "selection_score": float(best_trial.user_attrs["selection_score"]),
            "n_estimators": int(best_trial.user_attrs["n_estimators"]),
            "params": {key: json_ready(value) for key, value in best_trial.params.items()},
        }
        window_summaries.append(window_summary)
        print(
            f"Window {window_id:02d} best | RMSE={window_summary['rmse']:.4f} | "
            f"MAE={window_summary['mae']:.4f} | R2={window_summary['r2']:.4f} | "
            f"selection_RMSE={window_summary['selection_score']:.4f} | "
            f"n_estimators={window_summary['n_estimators']}"
        )

    save_per_window_trials(trial_rows, trials_path)
    print(f"Saved per-window Optuna trial table to: {trials_path}")

    selected_window = min(window_summaries, key=lambda item: item["rmse"])
    aggregated_params, aggregated_param_source_values = aggregate_window_params(window_summaries, search_space)
    aggregated_window_summary = {
        "window": "aggregated_all_windows",
        "params": aggregated_params,
    }
    train_params = window_train_params(aggregated_window_summary, config)
    best_n = median_n_estimators(window_summaries)
    final_params = {
        **train_params,
        "n_estimators": best_n,
        "n_estimators_source": "median_all_windows",
        "n_estimators_source_values": [int(item["n_estimators"]) for item in window_summaries],
        "aggregation_window_count": len(window_summaries),
        "aggregation_windows": [int(item["window"]) for item in window_summaries],
        "aggregated_param_source_values": aggregated_param_source_values,
        "search_space_version": XGB_SEARCH_SPACE_VERSION,
        "tuning_strategy": XGB_TUNING_STRATEGY,
        "search_space": search_space,
        "optuna_sampler": sampler_name,
        "selected_window": selected_window,
        "per_window_params": window_summaries,
    }
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
    trials_path = build_xgb_trials_path(params_path)
    train_params, best_n, final_params = tune_or_load_params(
        params_path, windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols, config
    )

    print("\n--- Training final XGBoost model ---")
    selected_window = final_params.get("selected_window", {})
    if selected_window:
        print(
            "Best individual validation window by RMSE: "
            f"window {selected_window['window']:02d} "
            f"(val laps {selected_window['val_lap_start']}-{selected_window['val_lap_end']}, "
            f"RMSE={selected_window['rmse']:.4f}, "
            f"MAE={selected_window['mae']:.4f}, "
            f"selection_RMSE={selected_window['selection_score']:.4f})."
        )
    if final_params.get("aggregation_windows"):
        print(f"Final hyperparameters aggregated from windows: {final_params['aggregation_windows']}")
    dmodel_full, dholdout, X_model_enc, X_holdout_enc = build_xgb_matrix(
        X_model_raw, X_holdout_raw, y_model, y_holdout, cat_cols
    )
    final_model = xgb.train(params=train_params, dtrain=dmodel_full, num_boost_round=best_n, verbose_eval=False)
    print(f"Selected n_estimators: {best_n}")
    print(f"n_estimators source: {final_params.get('n_estimators_source')}")
    if final_params.get("n_estimators_source_values"):
        print(f"n_estimators source values: {final_params.get('n_estimators_source_values')}")
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
    per_window_params = {
        int(window_summary["window"]): window_summary
        for window_summary in final_params.get("per_window_params", [])
    }

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
        if i in per_window_params:
            window_summary = per_window_params[i]
            eval_params = window_train_params(window_summary, config)
            eval_n = int(window_summary["n_estimators"])
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
            f"Window {i:02d} | train laps {int(train_laps[0])}-{int(train_laps[-1])} | "
            f"val laps {int(val_laps[0])}-{int(val_laps[-1])} | "
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
            "optuna_trials": config["optuna_trials"],
            "use_saved_xgb_params": config["use_saved_xgb_params"],
            "n_estimators": best_n,
            "n_estimators_source": final_params.get("n_estimators_source"),
            "aggregation_window_count": final_params.get("aggregation_window_count"),
            "aggregation_windows": ",".join(str(item) for item in final_params.get("aggregation_windows", [])),
            "search_space_version": final_params.get("search_space_version"),
            "tuning_strategy": final_params.get("tuning_strategy"),
            "selected_window": selected_window.get("window"),
            "selected_window_rmse": selected_window.get("rmse"),
            "selected_window_mae": selected_window.get("mae"),
            "selected_window_r2": selected_window.get("r2"),
            "selected_window_residual_std": selected_window.get("std"),
            "selected_window_selection_score": selected_window.get("selection_score"),
            "optuna_sampler": final_params.get("optuna_sampler"),
            "learning_rate": final_params.get("learning_rate"),
            "max_depth": final_params.get("max_depth"),
            "min_child_weight": final_params.get("min_child_weight"),
            "subsample": final_params.get("subsample"),
            "colsample_bytree": final_params.get("colsample_bytree"),
            "gamma": final_params.get("gamma"),
            "reg_alpha": final_params.get("reg_alpha"),
            "reg_lambda": final_params.get("reg_lambda"),
        },
        artifacts=[params_path, trials_path, model_path, model_metadata_path],
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
    print(f"          MAE final/SW={cos['mae_final']:.4f}/{cos['mae_sw']:.4f} | STD final/SW={cos['std_final']:.4f}/{cos['std_sw']:.4f}")
    print(f"COS_RMSE: {cos['cos_rmse']:.4f} | 95% CI: [{cos['cos_rmse_ci'][0]:.4f}, {cos['cos_rmse_ci'][1]:.4f}]")
    print(f"          RMSE final/SW={cos['rmse_final']:.4f}/{cos['rmse_sw']:.4f} | STD final/SW={cos['std_final']:.4f}/{cos['std_sw']:.4f}")

    print("\n--- Sequential holdout block diagnostic ---")
    print("NOTE: diagnostic only; these blocks are not used for tuning or model selection.")
    for block in holdout_block_results:
        r2_display = f"{block['r2']:.4f}" if block["r2"] is not None else "nan"
        print(
            f"Block {block['block']:02d} | laps {block['lap_start']}-{block['lap_end']} | "
            f"records={block['records']} | RMSE={block['rmse']:.4f} | "
            f"MAE={block['mae']:.4f} | R2={r2_display} | "
            f"STD={block['residual_std']:.4f}"
        )


if __name__ == "__main__":
    main()
