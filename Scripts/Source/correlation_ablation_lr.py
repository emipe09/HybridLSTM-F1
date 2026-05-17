"""Linear Regression ablation for strongly correlated encoded features.

The script detects encoded-feature pairs with absolute correlation above the
configured threshold inside the first sequential modeling block only. For each
pair, it reruns the Linear Regression sliding-window and final sequential
holdout protocol twice: once removing the first feature and once removing the
second feature.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from modeling_utils import (
    align_one_hot,
    build_sequential_split,
    build_sliding_windows,
    calc_holdout_ci,
    calc_stats,
    json_ready,
    load_cleaned_data,
    load_simple_yaml,
    prepare_raw_features,
    resolve_repo_path,
    safe_gp_name,
    select_modeling_columns,
    summarize_cos,
)


DEFAULT_CONFIG_ORDER = ["bahrain.yaml", "saudi.yaml", "usa.yaml", "italy.yaml", "hungary.yaml"]
DEFAULT_CORRELATION_THRESHOLD = 0.80


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test Linear Regression performance after removing one feature from "
            "each strongly correlated encoded-feature pair."
        )
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CORRELATION_THRESHOLD,
        help="Absolute correlation threshold used to select feature pairs. Default: 0.80.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional limit on the number of strongest pairs to test.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: <results_dir>/correlation_ablation_lr.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the ablation for all configured Grand Prix YAML files.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIG_ORDER,
        help="Config filenames or paths to run with --all.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining configs when one Grand Prix fails in --all mode.",
    )
    return parser.parse_args()


def fit_predict_linear_regression(X_train, y_train, X_eval, cat_cols, excluded_encoded_features=None):
    excluded_encoded_features = set(excluded_encoded_features or [])
    X_train_enc, X_eval_enc = align_one_hot(X_train, X_eval, cat_cols, drop_first=True)

    columns_to_drop = [col for col in X_train_enc.columns if col in excluded_encoded_features]
    if columns_to_drop:
        X_train_enc = X_train_enc.drop(columns=columns_to_drop)
        X_eval_enc = X_eval_enc.drop(columns=columns_to_drop, errors="ignore")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    model = LinearRegression()

    X_train_imp = imputer.fit_transform(X_train_enc)
    X_eval_imp = imputer.transform(X_eval_enc)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_eval_scaled = scaler.transform(X_eval_imp)

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_eval_scaled)
    return preds, model, imputer, scaler, X_train_enc.columns


def detect_strong_correlations(X_model_raw, cat_cols, threshold):
    X_model = X_model_raw.copy()
    for col in cat_cols:
        X_model[col] = X_model[col].fillna("Missing").astype(str)
    X_encoded = pd.get_dummies(X_model, columns=cat_cols, drop_first=True)

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)
    correlation_matrix = X_imputed.corr()

    strong_pairs = []
    columns = list(correlation_matrix.columns)
    for left_index, feature_1 in enumerate(columns):
        for feature_2 in columns[left_index + 1 :]:
            correlation = correlation_matrix.loc[feature_1, feature_2]
            if pd.isna(correlation):
                continue

            correlation = float(correlation)
            abs_correlation = abs(correlation)
            if abs_correlation < threshold:
                continue

            strong_pairs.append(
                {
                    "pair_id": len(strong_pairs) + 1,
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": correlation,
                    "abs_correlation": abs_correlation,
                    "direction": "positive" if correlation > 0 else "negative",
                }
            )

    strong_pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    for pair_id, item in enumerate(strong_pairs, start=1):
        item["pair_id"] = pair_id
    return strong_pairs


def prepare_temporal_blocks(df_base, config):
    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    num_cols, cat_cols = select_modeling_columns(df_base, config)
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

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

    return {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "X_model_raw": X_model_raw,
        "y_model": y_model,
        "X_holdout_raw": X_holdout_raw,
        "y_holdout": y_holdout,
        "lap_model_sorted": lap_model_sorted,
        "unique_laps": unique_laps,
        "windows": windows,
        "split_info": split_info,
    }


def evaluate_lr_protocol(blocks, config, excluded_encoded_features=None):
    cat_cols = blocks["cat_cols"]
    X_model_raw = blocks["X_model_raw"]
    y_model = blocks["y_model"]
    X_holdout_raw = blocks["X_holdout_raw"]
    y_holdout = blocks["y_holdout"]
    lap_model_sorted = blocks["lap_model_sorted"]
    unique_laps = blocks["unique_laps"]
    windows = blocks["windows"]

    results = {"window": [], "rmse": [], "mae": [], "r2": [], "std": []}

    for i, (start, split, end) in enumerate(windows, start=1):
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)

        X_train, y_train = X_model_raw.loc[train_mask], y_model.loc[train_mask]
        X_val, y_val = X_model_raw.loc[val_mask], y_model.loc[val_mask]
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(f"Window {i}: empty train or validation fold.")

        preds, *_ = fit_predict_linear_regression(
            X_train,
            y_train,
            X_val,
            cat_cols,
            excluded_encoded_features=excluded_encoded_features,
        )

        rmse_value = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae_value = float(mean_absolute_error(y_val, preds))
        r2_value = float(r2_score(y_val, preds))
        std_value = float(np.std(np.asarray(y_val) - np.asarray(preds), ddof=1)) if len(y_val) > 1 else 0.0

        results["window"].append(i)
        results["rmse"].append(rmse_value)
        results["mae"].append(mae_value)
        results["r2"].append(r2_value)
        results["std"].append(std_value)

    rmse_m, rmse_l, rmse_u = calc_stats(results["rmse"])
    mae_m, mae_l, mae_u = calc_stats(results["mae"])
    r2_m, r2_l, r2_u = calc_stats(results["r2"])
    std_m, _, _ = calc_stats(results["std"])

    preds_holdout, _, _, _, feature_names = fit_predict_linear_regression(
        X_model_raw,
        y_model,
        X_holdout_raw,
        cat_cols,
        excluded_encoded_features=excluded_encoded_features,
    )

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

    return {
        "excluded_encoded_features": list(excluded_encoded_features or []),
        "active_feature_count": len(feature_names),
        "window_results": results,
        "summary_metrics": {
            "sw_rmse_mean": rmse_m,
            "sw_rmse_ci_lower": rmse_l,
            "sw_rmse_ci_upper": rmse_u,
            "sw_mae_mean": mae_m,
            "sw_mae_ci_lower": mae_l,
            "sw_mae_ci_upper": mae_u,
            "sw_r2_mean": r2_m,
            "sw_r2_ci_lower": r2_l,
            "sw_r2_ci_upper": r2_u,
            "sw_residual_std_mean": std_m,
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
        },
    }


def build_output_dir(repo_root: Path, config: dict, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return resolve_repo_path(repo_root, output_dir_arg)
    return resolve_repo_path(repo_root, str(config["results_dir"])) / "correlation_ablation_lr"


def write_outputs(output_dir, safe_name, pairs, ablation_rows, summary):
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = output_dir / f"{safe_name}_correlated_feature_pairs.csv"
    results_path = output_dir / f"{safe_name}_correlation_ablation_lr_results.csv"
    summary_path = output_dir / f"{safe_name}_correlation_ablation_lr_summary.json"

    pd.DataFrame(pairs).to_csv(pairs_path, index=False)
    pd.DataFrame(ablation_rows).to_csv(results_path, index=False)
    summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")

    return pairs_path, results_path, summary_path


def build_ablation_rows(pairs, baseline_eval, blocks, config):
    baseline_metrics = baseline_eval["summary_metrics"]
    rows = []

    for pair in pairs:
        feature_1 = pair["feature_1"]
        feature_2 = pair["feature_2"]
        for removed_feature, kept_feature in ((feature_1, feature_2), (feature_2, feature_1)):
            ablation_eval = evaluate_lr_protocol(blocks, config, excluded_encoded_features=[removed_feature])
            metrics = ablation_eval["summary_metrics"]
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": pair["correlation"],
                    "abs_correlation": pair["abs_correlation"],
                    "direction": pair["direction"],
                    "removed_feature": removed_feature,
                    "kept_feature": kept_feature,
                    "active_feature_count": ablation_eval["active_feature_count"],
                    **metrics,
                    "delta_holdout_rmse_vs_baseline": metrics["holdout_rmse"] - baseline_metrics["holdout_rmse"],
                    "delta_holdout_mae_vs_baseline": metrics["holdout_mae"] - baseline_metrics["holdout_mae"],
                    "delta_holdout_r2_vs_baseline": metrics["holdout_r2"] - baseline_metrics["holdout_r2"],
                    "delta_sw_rmse_mean_vs_baseline": metrics["sw_rmse_mean"] - baseline_metrics["sw_rmse_mean"],
                    "delta_sw_mae_mean_vs_baseline": metrics["sw_mae_mean"] - baseline_metrics["sw_mae_mean"],
                    "delta_sw_r2_mean_vs_baseline": metrics["sw_r2_mean"] - baseline_metrics["sw_r2_mean"],
                }
            )

    return rows


def run_single(threshold, max_pairs, output_dir_arg):
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()

    blocks = prepare_temporal_blocks(df_base, config)
    pairs = detect_strong_correlations(blocks["X_model_raw"], blocks["cat_cols"], threshold)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    print("--- LINEAR REGRESSION: CORRELATED-FEATURE ABLATION ---")
    print(f"Grand Prix: {target_gp_name}")
    print(f"Correlation threshold: |r| >= {threshold:.2f}")
    print(f"Numerical features: {blocks['num_cols']}")
    print(f"Categorical features: {blocks['cat_cols']}")
    print(
        f"Modeling block: laps {blocks['split_info']['lap_min']}-{blocks['split_info']['model_end_lap']} | "
        f"records={blocks['split_info']['model_records']}"
    )
    print(
        f"Sequential holdout kept untouched: "
        f"laps {blocks['split_info']['holdout_start_lap']}-{blocks['split_info']['lap_max']} | "
        f"records={blocks['split_info']['holdout_records']}"
    )
    print(f"Strong correlated pairs selected: {len(pairs)}")

    baseline_eval = evaluate_lr_protocol(blocks, config)
    ablation_rows = build_ablation_rows(pairs, baseline_eval, blocks, config)

    safe_name = safe_gp_name(target_gp_name)
    output_dir = build_output_dir(repo_root, config, output_dir_arg)
    summary = {
        "target_gp_name": target_gp_name,
        "threshold": threshold,
        "max_pairs": max_pairs,
        "target_col": config["target_col"],
        "lap_col": config["lap_col"],
        "numerical_features": blocks["num_cols"],
        "categorical_features": blocks["cat_cols"],
        "split_info": blocks["split_info"],
        "preprocessing": "one_hot_drop_first_median_imputer_standard_scaler",
        "selection_scope": "correlations computed inside the first sequential modeling block only",
        "evaluation_protocol": "sliding-window validation inside modeling block plus final sequential holdout",
        "baseline": baseline_eval,
        "strong_pairs": pairs,
        "ablation_count": len(ablation_rows),
    }
    pairs_path, results_path, summary_path = write_outputs(output_dir, safe_name, pairs, ablation_rows, summary)

    baseline_metrics = baseline_eval["summary_metrics"]
    print(
        "Baseline holdout RMSE/MAE/R2: "
        f"{baseline_metrics['holdout_rmse']:.4f} / "
        f"{baseline_metrics['holdout_mae']:.4f} / "
        f"{baseline_metrics['holdout_r2']:.4f}"
    )

    if ablation_rows:
        best_row = min(ablation_rows, key=lambda row: row["holdout_rmse"])
        print(
            "Best ablation by holdout RMSE: "
            f"pair {best_row['pair_id']} remove {best_row['removed_feature']} | "
            f"holdout RMSE={best_row['holdout_rmse']:.4f} | "
            f"delta={best_row['delta_holdout_rmse_vs_baseline']:.4f}"
        )
    else:
        print("No pairs met the threshold; only the baseline was evaluated.")

    print("\nOutputs:")
    print(f"- {pairs_path}")
    print(f"- {results_path}")
    print(f"- {summary_path}")

    return {
        "target_gp_name": target_gp_name,
        "pairs": len(pairs),
        "ablations": len(ablation_rows),
        "summary_path": str(summary_path),
    }


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_config_path(repo_root: Path, config_value: str) -> Path:
    config_path = Path(config_value)
    if not config_path.is_absolute():
        if len(config_path.parts) == 1:
            config_path = repo_root / "configs" / config_path
        else:
            config_path = repo_root / config_path
    return config_path


def run_all(args):
    repo_root = repo_root_from_script()
    original_config_path = os.environ.get("CONFIG_PATH")
    original_target_gp_name = os.environ.get("TARGET_GP_NAME")
    summaries = []
    failures = []

    try:
        for config_value in args.configs:
            config_path = resolve_config_path(repo_root, config_value)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            config = load_simple_yaml(config_path)
            target_gp_name = str(config.get("target_gp_name", config_path.stem))
            os.environ["CONFIG_PATH"] = str(config_path)
            os.environ["TARGET_GP_NAME"] = target_gp_name

            print("\n" + "=" * 80)
            print(f"Correlation ablation for: {target_gp_name}")
            print(f"Config: {config_path}")
            print("=" * 80)

            try:
                summaries.append(run_single(args.threshold, args.max_pairs, args.output_dir))
            except Exception as exc:
                failures.append((target_gp_name, config_path, exc))
                print(f"FAILED: {target_gp_name} | {exc}")
                if not args.continue_on_error:
                    raise
    finally:
        if original_config_path is None:
            os.environ.pop("CONFIG_PATH", None)
        else:
            os.environ["CONFIG_PATH"] = original_config_path

        if original_target_gp_name is None:
            os.environ.pop("TARGET_GP_NAME", None)
        else:
            os.environ["TARGET_GP_NAME"] = original_target_gp_name

    print("\n--- Batch summary ---")
    for summary in summaries:
        print(
            f"{summary['target_gp_name']}: pairs={summary['pairs']} | "
            f"ablations={summary['ablations']}"
        )

    if failures:
        print("\nCompleted with failures:")
        for target_gp_name, config_path, exc in failures:
            print(f"- {target_gp_name} ({config_path.name}): {exc}")
        return 1

    print("\nAll correlation-ablation runs completed successfully.")
    return 0


def main():
    args = parse_args()
    if args.all:
        return run_all(args)

    run_single(args.threshold, args.max_pairs, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
