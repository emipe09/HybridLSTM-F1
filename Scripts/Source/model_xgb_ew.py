"""XGBoost with Optuna, expanding-window validation, and sequential holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from modeling_utils import (
    build_expanding_windows,
    build_sequential_split,
    build_xgb_ew_model_paths,
    build_xgb_ew_params_path,
    calc_holdout_ci,
    calc_stats,
    decode_step_key,
    json_ready,
    log_mlflow_run,
    load_cleaned_data,
    prepare_raw_features,
    select_modeling_columns,
    summarize_cos,
)
from xgb_utils import (
    BASE_XGB_PARAMS,
    XGB_SEARCH_SPACE_VERSION,
    aggregate_window_params,
    build_holdout_block_diagnostics,
    build_optuna_sampler,
    build_xgb_matrix,
    build_xgb_search_space,
    median_n_estimators,
    save_per_window_trials,
    suggest_xgb_params,
    window_train_params,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


XGB_EW_TUNING_STRATEGY = "per_fold_all_folds_median_params_v1"


def build_xgb_ew_trials_path(params_path):
    return params_path.with_name(params_path.stem.replace("_params_ew", "_optuna_trials_ew") + ".csv")


def saved_ew_params_match_current_config(loaded_params, search_space, sampler_name):
    return (
        loaded_params.get("search_space_version") == XGB_SEARCH_SPACE_VERSION
        and loaded_params.get("tuning_strategy") == XGB_EW_TUNING_STRATEGY
        and loaded_params.get("search_space") == search_space
        and loaded_params.get("optuna_sampler") == sampler_name
    )


def tune_or_load_params_ew(
    params_path, windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols, config
):
    search_space = build_xgb_search_space(config)
    sampler_name = str(config.get("xgb_optuna_sampler", "tpe")).lower()
    trials_path = build_xgb_ew_trials_path(params_path)

    if bool(config["use_saved_xgb_params"]) and params_path.exists():
        print(f"Using saved XGBoost EW parameters: {params_path}")
        with params_path.open("r", encoding="utf-8") as file:
            loaded_params = json.load(file)

        if not saved_ew_params_match_current_config(loaded_params, search_space, sampler_name):
            print(
                "Saved XGBoost EW parameters do not match the current search configuration; "
                "running Optuna again."
            )
        else:
            selected_fold = loaded_params.get("selected_fold", {})
            best_n = int(loaded_params.get("n_estimators", selected_fold.get("n_estimators", 0)))
            if best_n < 1:
                raise ValueError("Invalid saved XGBoost EW parameter file: missing n_estimators.")

            excluded_metadata = {
                "n_estimators",
                "search_space_version",
                "tuning_strategy",
                "search_space",
                "optuna_sampler",
                "selected_fold",
                "per_fold_params",
                "aggregation_fold_count",
                "aggregation_folds",
                "aggregated_param_source_values",
                "n_estimators_source",
                "n_estimators_source_values",
            }
            train_params = {key: value for key, value in loaded_params.items() if key not in excluded_metadata}
            train_params = {**BASE_XGB_PARAMS, **train_params, "seed": int(config["random_seed"])}
            return train_params, best_n, loaded_params

    optuna_trials = int(config["optuna_trials"])
    print(f"Running per-fold Optuna tuning with {optuna_trials} trials per fold using {sampler_name} sampler...")
    print(f"XGBoost search space version: {XGB_SEARCH_SPACE_VERSION}")
    print(f"XGBoost EW tuning strategy: {XGB_EW_TUNING_STRATEGY}")
    print(f"XGBoost search space: {search_space}")
    print(
        "Each expanding fold receives an independent Optuna study. The final holdout parameters are "
        "the median of the best Optuna parameters selected across all folds; final n_estimators is "
        "the median early-stopping iteration across those folds."
    )

    fold_summaries = []
    trial_rows = []

    for fold_id, (start, split, end) in enumerate(windows, start=1):
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        X_train, y_train = X_model_raw.loc[train_mask], y_model.loc[train_mask]
        X_val, y_val = X_model_raw.loc[val_mask], y_model.loc[val_mask]
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(f"Fold {fold_id}: empty train or validation set.")

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
            trial.set_user_attr("fold", fold_id)
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
            f"Running Optuna for fold {fold_id:02d}: "
            f"train {decode_step_key(train_laps[0])} → {decode_step_key(train_laps[-1])} ({len(X_train)} records) | "
            f"val {decode_step_key(val_laps[0])} → {decode_step_key(val_laps[-1])}"
        )
        study = optuna.create_study(direction="minimize", sampler=build_optuna_sampler(config))
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)

        for trial in study.trials:
            row = {
                "fold": fold_id,
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
        fold_summary = {
            "fold": fold_id,
            "train_lap_start": int(train_laps[0]),
            "train_lap_end": int(train_laps[-1]),
            "val_lap_start": int(val_laps[0]),
            "val_lap_end": int(val_laps[-1]),
            "train_records": int(len(X_train)),
            "objective_value": float(best_trial.value),
            "rmse": float(best_trial.user_attrs["rmse"]),
            "mae": float(best_trial.user_attrs["mae"]),
            "r2": float(best_trial.user_attrs["r2"]),
            "std": float(best_trial.user_attrs["std"]),
            "selection_score": float(best_trial.user_attrs["selection_score"]),
            "n_estimators": int(best_trial.user_attrs["n_estimators"]),
            "params": {key: json_ready(value) for key, value in best_trial.params.items()},
        }
        fold_summaries.append(fold_summary)
        print(
            f"Fold {fold_id:02d} best | RMSE={fold_summary['rmse']:.4f} | "
            f"MAE={fold_summary['mae']:.4f} | R2={fold_summary['r2']:.4f} | "
            f"n_estimators={fold_summary['n_estimators']}"
        )

    save_per_window_trials(trial_rows, trials_path)
    print(f"Saved per-fold Optuna trial table to: {trials_path}")

    selected_fold = min(fold_summaries, key=lambda item: item["rmse"])
    aggregated_params, aggregated_param_source_values = aggregate_window_params(fold_summaries, search_space)
    aggregated_fold_summary = {
        "fold": "aggregated_all_folds",
        "params": aggregated_params,
    }
    train_params = window_train_params(aggregated_fold_summary, config)
    best_n = median_n_estimators(fold_summaries)
    final_params = {
        **train_params,
        "n_estimators": best_n,
        "n_estimators_source": "median_all_folds",
        "n_estimators_source_values": [int(item["n_estimators"]) for item in fold_summaries],
        "aggregation_fold_count": len(fold_summaries),
        "aggregation_folds": [int(item["fold"]) for item in fold_summaries],
        "aggregated_param_source_values": aggregated_param_source_values,
        "search_space_version": XGB_SEARCH_SPACE_VERSION,
        "tuning_strategy": XGB_EW_TUNING_STRATEGY,
        "search_space": search_space,
        "optuna_sampler": sampler_name,
        "selected_fold": selected_fold,
        "per_fold_params": fold_summaries,
    }
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w", encoding="utf-8") as file:
        json.dump(final_params, file, indent=4)
    print(f"Saved XGBoost EW parameters to: {params_path}")
    return train_params, best_n, final_params


def main():
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()

    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    num_cols, cat_cols = select_modeling_columns(df_base, config)
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    print("--- XGBOOST: OPTUNA + EXPANDING WINDOW + SEQUENTIAL HOLDOUT ---")
    print(f"Grand Prix: {target_gp_name}")
    print(
        "Config: "
        f"holdout={config['holdout_ratio']} | window={config.get('xgb_ew_window_ratio', config['window_ratio'])} | "
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

    windows, window_size, train_size, val_size, step_size = build_expanding_windows(
        len(unique_laps),
        float(config.get("xgb_ew_window_ratio", config["window_ratio"])),
        float(config["window_train_ratio"]),
        float(config["window_step_ratio"]),
    )

    print("\n--- Sequential split ---")
    print(f"Total temporal steps: {total_laps} ({decode_step_key(lap_min)} → {decode_step_key(lap_max)})")
    print(f"Modeling block: {decode_step_key(lap_min)} – {decode_step_key(model_end_lap)} | records={len(X_model_raw)}")
    print(f"Holdout block: {decode_step_key(holdout_start_lap)} – {decode_step_key(lap_max)} | records={len(X_holdout_raw)}")
    print(
        f"Expanding folds: {len(windows)} | initial_train={train_size} | "
        f"val_chunk={val_size} | step={step_size}"
    )

    params_path = build_xgb_ew_params_path(repo_root, config)
    trials_path = build_xgb_ew_trials_path(params_path)
    train_params, best_n, final_params = tune_or_load_params_ew(
        params_path, windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols, config
    )

    print("\n--- Training final XGBoost EW model ---")
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

    model_path, model_metadata_path = build_xgb_ew_model_paths(repo_root, config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    model_metadata = {
        "target_gp_name": target_gp_name,
        "model": "xgboost",
        "validation_protocol": "expanding_window",
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
    print(f"Saved final XGBoost EW model to: {model_path}")
    print(f"Saved final XGBoost EW model metadata to: {model_metadata_path}")

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
            "aggregation_fold_count": final_params.get("aggregation_fold_count"),
            "aggregation_folds": ",".join(str(item) for item in final_params.get("aggregation_folds", [])),
            "search_space_version": final_params.get("search_space_version"),
            "tuning_strategy": final_params.get("tuning_strategy"),
            "selected_fold": selected_fold.get("fold"),
            "selected_fold_rmse": selected_fold.get("rmse"),
            "optuna_sampler": final_params.get("optuna_sampler"),
        },
        artifacts=[params_path, trials_path, model_path, model_metadata_path],
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

    # Persist per-row baseline predictions (OOF + holdout) so the hybrid model can reuse
    # this exact baseline instead of regenerating it. Auxiliary and non-destructive: a
    # failure here must not affect the reported XGBoost-EW artifacts/metrics above.
    try:
        from baseline_utils import export_baseline_predictions

        export_baseline_predictions(
            repo_root, config, "xgb_ew", df_base, num_cols, cat_cols, target_col, lap_col
        )
    except Exception as exc:  # pragma: no cover - export is best-effort
        print(f"WARNING: could not export XGBoost-EW baseline predictions: {exc}")


if __name__ == "__main__":
    main()
