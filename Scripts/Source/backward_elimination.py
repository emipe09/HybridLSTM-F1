"""Backward elimination for the linear-model feature design matrix.

The final sequential holdout is never used during feature elimination. The
procedure fits preprocessing and OLS only on the first modeling block selected
by the configured temporal split.
"""

from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from numpy.linalg import LinAlgError
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:
    missing_package = exc.name
    raise SystemExit(
        f"Missing dependency: {missing_package}\n"
        "Run this script with the project virtual environment, for example:\n"
        r"  .\.venv\Scripts\python.exe Scripts\Source\backward_elimination.py --all"
        "\nOr install dependencies with:\n"
        r"  .\.venv\Scripts\python.exe -m pip install -r Utils\requirements.txt"
    ) from exc

from modeling_utils import (
    align_one_hot,
    build_sequential_split,
    json_ready,
    load_cleaned_data,
    load_simple_yaml,
    prepare_raw_features,
    resolve_repo_path,
    safe_gp_name,
    select_modeling_columns,
)

DEFAULT_CONFIG_ORDER = ["bahrain.yaml"]
NORMAL_95 = 1.959963984540054
STRONG_CORRELATION_THRESHOLD = 0.80
STRONG_CORRELATION_COLUMNS = [
    "feature_1",
    "feature_2",
    "correlation",
    "abs_correlation",
    "direction",
    "feature_1_removed",
    "feature_2_removed",
    "removed_features_in_pair",
    "pair_removed_status",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run p-value based backward elimination on the configured Grand Prix "
            "linear-model design matrix, preserving the final sequential holdout."
        )
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Maximum accepted non-intercept p-value. Default: 0.05.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: <results_dir>/backward_elimination.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run backward elimination for all configured Grand Prix YAML files.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIG_ORDER,
        help="Config filenames or paths to run with --all. Default: all supported Grand Prix YAML files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining configs when one Grand Prix fails in --all mode.",
    )
    return parser.parse_args()


class OLSResult:
    def __init__(self, feature_names, params, residuals, fitted_values, pvalues, standard_errors, r2):
        self.feature_names = list(feature_names)
        self.params = pd.Series(params, index=self.feature_names)
        self.resid = np.asarray(residuals, dtype=np.float64)
        self.fittedvalues = np.asarray(fitted_values, dtype=np.float64)
        self.pvalues = pd.Series(pvalues, index=self.feature_names)
        self.bse = pd.Series(standard_errors, index=self.feature_names)
        self.rsquared = float(r2)

    def summary_text(self):
        table = pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.bse,
                "p_value": self.pvalues,
            }
        )
        return (
            "OLS summary generated with NumPy least squares.\n"
            "P-values use a large-sample normal approximation.\n\n"
            f"R-squared: {self.rsquared:.6f}\n\n"
            + table.to_string(float_format=lambda value: f"{value:.6g}")
            + "\n"
        )


def calculate_residual_metrics(residuals):
    residuals = np.asarray(residuals, dtype=float)
    squared_errors = residuals**2
    abs_errors = np.abs(residuals)

    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_errors))

    if len(residuals) > 1:
        mse_margin = NORMAL_95 * float(np.std(squared_errors, ddof=1)) / math.sqrt(len(squared_errors))
        mae_margin = NORMAL_95 * float(np.std(abs_errors, ddof=1)) / math.sqrt(len(abs_errors))
        rmse_ci = (
            float(np.sqrt(max(0.0, mse - mse_margin))),
            float(np.sqrt(max(0.0, mse + mse_margin))),
        )
        mae_ci = (float(mae - mae_margin), float(mae + mae_margin))
    else:
        rmse_ci = (rmse, rmse)
        mae_ci = (mae, mae)

    return {
        "rmse": rmse,
        "rmse_ci": rmse_ci,
        "mae": mae,
        "mae_ci": mae_ci,
    }


def calculate_model_metrics(model):
    metrics = calculate_residual_metrics(model.resid)
    metrics["r2"] = model.rsquared
    return metrics


def fit_ols(y, X):
    """Fit OLS without importing statsmodels/scipy."""
    X = X.astype(np.float64, copy=False)
    x_values = X.to_numpy(dtype=np.float64, copy=False)
    y_values = np.asarray(y, dtype=np.float64)

    if not np.isfinite(x_values).all():
        raise ValueError("The OLS design matrix contains NaN or infinite values.")
    if not np.isfinite(y_values).all():
        raise ValueError("The OLS target vector contains NaN or infinite values.")

    try:
        params, _, rank, _ = np.linalg.lstsq(x_values, y_values, rcond=None)
    except LinAlgError:
        q_matrix, r_matrix = np.linalg.qr(x_values, mode="reduced")
        params = np.linalg.solve(r_matrix, q_matrix.T @ y_values)
        rank = int(np.linalg.matrix_rank(x_values))

    fitted_values = x_values @ params
    residuals = y_values - fitted_values
    n_obs, n_features = x_values.shape
    dof_resid = max(n_obs - rank, 1)
    sigma2 = float((residuals @ residuals) / dof_resid)

    xtx_pinv = np.linalg.pinv(x_values.T @ x_values)
    variances = np.clip(np.diag(xtx_pinv) * sigma2, a_min=0.0, a_max=None)
    standard_errors = np.sqrt(variances)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.divide(params, standard_errors, out=np.zeros_like(params), where=standard_errors > 0)
    pvalues = np.array([math.erfc(abs(float(value)) / math.sqrt(2.0)) for value in t_values])

    ss_res = float(residuals @ residuals)
    centered = y_values - float(np.mean(y_values))
    ss_tot = float(centered @ centered)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return OLSResult(X.columns, params, residuals, fitted_values, pvalues, standard_errors, r2)


def analyze_strong_correlations(X_model_design, removed_features, threshold=STRONG_CORRELATION_THRESHOLD):
    """Find strongly correlated design-matrix feature pairs inside the modeling block."""
    feature_matrix = X_model_design.drop(columns=["const"], errors="ignore")
    removed_feature_set = set(removed_features)
    correlation_matrix = feature_matrix.corr()
    strong_pairs = []
    columns = list(correlation_matrix.columns)

    for left_index, feature_1 in enumerate(columns):
        for feature_2 in columns[left_index + 1 :]:
            correlation = correlation_matrix.loc[feature_1, feature_2]
            if pd.isna(correlation):
                continue

            correlation = float(correlation)
            if correlation <= threshold and correlation >= -threshold:
                continue

            removed_in_pair = [feature for feature in (feature_1, feature_2) if feature in removed_feature_set]
            if len(removed_in_pair) == 2:
                pair_status = "both_removed"
            elif len(removed_in_pair) == 1:
                pair_status = "one_removed"
            else:
                pair_status = "none_removed"

            strong_pairs.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": correlation,
                    "abs_correlation": abs(correlation),
                    "direction": "positive" if correlation > threshold else "negative",
                    "feature_1_removed": feature_1 in removed_feature_set,
                    "feature_2_removed": feature_2 in removed_feature_set,
                    "removed_features_in_pair": removed_in_pair,
                    "pair_removed_status": pair_status,
                }
            )

    strong_pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return strong_pairs


def format_metric_triplet(metrics):
    return f"R2={metrics['r2']:.6f} | RMSE={metrics['rmse']:.6f} | MAE={metrics['mae']:.6f}"


def build_validation_summary_text(result):
    comparison = result["metric_comparison"]
    delta = comparison["delta_final_minus_baseline"]
    lines = [
        "Backward-elimination validation report",
        "",
        "Model metric comparison",
        f"Baseline full model:   {format_metric_triplet(comparison['baseline_full_model'])}",
        f"Final reduced model:   {format_metric_triplet(comparison['final_reduced_model'])}",
        f"Delta final-baseline:  R2={delta['r2']:.6f} | RMSE={delta['rmse']:.6f} | MAE={delta['mae']:.6f}",
        "",
        f"Strong correlations (r > {STRONG_CORRELATION_THRESHOLD:.2f} or r < -{STRONG_CORRELATION_THRESHOLD:.2f})",
    ]

    if result["strong_correlations"]:
        for item in result["strong_correlations"]:
            removed = ", ".join(item["removed_features_in_pair"]) or "none"
            lines.append(
                f"- {item['feature_1']} vs {item['feature_2']}: "
                f"r={item['correlation']:.6f} ({item['direction']}), "
                f"removed={removed}, status={item['pair_removed_status']}"
            )
    else:
        lines.append("- No strongly correlated encoded-feature pairs were found.")

    lines.extend(["", "Final reduced OLS summary", result["final_model"].summary_text()])
    return "\n".join(lines)


def build_modeling_design_matrix(df_base, config):
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

    model_laps = lap_series.loc[model_idx]
    model_order_idx = model_laps.sort_values(kind="mergesort").index
    X_model_raw = X_model_raw.loc[model_order_idx].reset_index(drop=True)
    y_model = y_model.loc[model_order_idx].reset_index(drop=True)

    empty_eval = X_model_raw.iloc[0:0].copy()
    X_model_encoded, _ = align_one_hot(X_model_raw, empty_eval, cat_cols, drop_first=True)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_model_imputed = pd.DataFrame(
        imputer.fit_transform(X_model_encoded),
        columns=X_model_encoded.columns,
    )
    X_model_scaled = pd.DataFrame(
        scaler.fit_transform(X_model_imputed),
        columns=X_model_imputed.columns,
        dtype=np.float64,
    )
    X_model_design = X_model_scaled.copy()
    X_model_design.insert(0, "const", 1.0)
    X_model_design = X_model_design.astype(np.float64)

    split_info = {
        "total_laps": total_laps,
        "lap_min": lap_min,
        "lap_max": lap_max,
        "model_end_lap": model_end_lap,
        "holdout_start_lap": holdout_start_lap,
        "model_records": len(X_model_raw),
        "holdout_records": len(holdout_idx),
    }
    return X_model_design, y_model.reset_index(drop=True), num_cols, cat_cols, split_info


def run_backward_elimination(X_model_design, y_model, alpha):
    baseline_model = fit_ols(y_model, X_model_design)
    baseline_metrics = calculate_model_metrics(baseline_model)

    selected_features = list(X_model_design.columns)
    history = []
    step = 0

    while True:
        current_model = fit_ols(y_model, X_model_design[selected_features])
        current_metrics = calculate_model_metrics(current_model)
        p_values = current_model.pvalues.drop(labels=["const"], errors="ignore")

        if p_values.empty:
            stop_reason = "only_intercept_remaining"
            break

        max_p_value = float(p_values.max())
        worst_feature = str(p_values.idxmax())

        if max_p_value <= alpha:
            stop_reason = "all_p_values_within_alpha"
            break

        rmse_ci = current_metrics["rmse_ci"]
        baseline_rmse_ci = baseline_metrics["rmse_ci"]
        rmse_degraded = rmse_ci[0] > baseline_rmse_ci[1]

        history.append(
            {
                "step": step,
                "removed_feature": worst_feature,
                "p_value": max_p_value,
                "rmse": current_metrics["rmse"],
                "rmse_ci": current_metrics["rmse_ci"],
                "mae": current_metrics["mae"],
                "mae_ci": current_metrics["mae_ci"],
                "r2": current_metrics["r2"],
                "rmse_degraded_vs_baseline_ci": bool(rmse_degraded),
                "features_before_removal": len(selected_features),
            }
        )
        selected_features.remove(worst_feature)
        step += 1

    final_model = fit_ols(y_model, X_model_design[selected_features])
    final_metrics = calculate_model_metrics(final_model)
    removed_features = [item["removed_feature"] for item in history]
    strong_correlations = analyze_strong_correlations(X_model_design, removed_features)

    return {
        "baseline_model": baseline_model,
        "final_model": final_model,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "metric_comparison": {
            "baseline_full_model": baseline_metrics,
            "final_reduced_model": final_metrics,
            "delta_final_minus_baseline": {
                "rmse": final_metrics["rmse"] - baseline_metrics["rmse"],
                "mae": final_metrics["mae"] - baseline_metrics["mae"],
                "r2": final_metrics["r2"] - baseline_metrics["r2"],
            },
        },
        "history": history,
        "selected_features": selected_features,
        "removed_features": removed_features,
        "strong_correlations": strong_correlations,
        "stop_reason": stop_reason,
    }


def build_output_dir(repo_root: Path, config: dict, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return resolve_repo_path(repo_root, output_dir_arg)
    return resolve_repo_path(repo_root, str(config["results_dir"])) / "backward_elimination"


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


def write_outputs(output_dir: Path, safe_name: str, result: dict, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / f"{safe_name}_backward_elimination_history.csv"
    correlation_path = output_dir / f"{safe_name}_backward_elimination_strong_correlations.csv"
    summary_path = output_dir / f"{safe_name}_backward_elimination_summary.json"
    summary_text_path = output_dir / f"{safe_name}_backward_elimination_ols_summary.txt"

    pd.DataFrame(result["history"]).to_csv(history_path, index=False)
    pd.DataFrame(result["strong_correlations"], columns=STRONG_CORRELATION_COLUMNS).to_csv(
        correlation_path, index=False
    )

    summary = {
        **metadata,
        "baseline_metrics": result["baseline_metrics"],
        "final_metrics": result["final_metrics"],
        "metric_comparison": result["metric_comparison"],
        "strong_correlation_threshold": STRONG_CORRELATION_THRESHOLD,
        "strong_correlations": result["strong_correlations"],
        "selected_features": result["selected_features"],
        "removed_features": result["removed_features"],
        "stop_reason": result["stop_reason"],
    }
    summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    summary_text_path.write_text(build_validation_summary_text(result), encoding="utf-8")

    return history_path, correlation_path, summary_path, summary_text_path


def run_single(alpha: float, output_dir_arg: str | None):
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()

    X_model_design, y_model, num_cols, cat_cols, split_info = build_modeling_design_matrix(df_base, config)
    result = run_backward_elimination(X_model_design, y_model, alpha)

    safe_name = safe_gp_name(target_gp_name)
    output_dir = build_output_dir(repo_root, config, output_dir_arg)
    history_path, correlation_path, summary_path, summary_text_path = write_outputs(
        output_dir,
        safe_name,
        result,
        {
            "target_gp_name": target_gp_name,
            "alpha": alpha,
            "target_col": config["target_col"],
            "lap_col": config["lap_col"],
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "split_info": split_info,
            "preprocessing": "one_hot_drop_first_median_imputer_standard_scaler",
            "selection_scope": "first sequential modeling block only; final holdout untouched",
        },
    )

    print("--- BACKWARD ELIMINATION: LINEAR MODEL DESIGN MATRIX ---")
    print(f"Grand Prix: {target_gp_name}")
    print(f"Alpha: {alpha}")
    print(
        f"Modeling block: laps {split_info['lap_min']}-{split_info['model_end_lap']} | "
        f"records={split_info['model_records']}"
    )
    print(
        f"Sequential holdout kept untouched: laps {split_info['holdout_start_lap']}-{split_info['lap_max']} | "
        f"records={split_info['holdout_records']}"
    )
    print(f"Initial encoded features including intercept: {len(X_model_design.columns)}")
    print(f"Removed features: {len(result['removed_features'])}")
    print(f"Selected features including intercept: {len(result['selected_features'])}")
    print(f"Strong correlations found (|r| > {STRONG_CORRELATION_THRESHOLD:.2f}): {len(result['strong_correlations'])}")
    print(f"Stop reason: {result['stop_reason']}")
    print(
        "Baseline full-model R2/RMSE/MAE: "
        f"{result['baseline_metrics']['r2']:.4f} / "
        f"{result['baseline_metrics']['rmse']:.4f} / "
        f"{result['baseline_metrics']['mae']:.4f}"
    )
    print(
        "Final reduced-model R2/RMSE/MAE: "
        f"{result['final_metrics']['r2']:.4f} / "
        f"{result['final_metrics']['rmse']:.4f} / "
        f"{result['final_metrics']['mae']:.4f}"
    )
    print(
        "Delta final-baseline R2/RMSE/MAE: "
        f"{result['metric_comparison']['delta_final_minus_baseline']['r2']:.4f} / "
        f"{result['metric_comparison']['delta_final_minus_baseline']['rmse']:.4f} / "
        f"{result['metric_comparison']['delta_final_minus_baseline']['mae']:.4f}"
    )
    print("\nSelected features:")
    for feature in result["selected_features"]:
        print(f"- {feature}")
    print("\nOutputs:")
    print(f"- {history_path}")
    print(f"- {correlation_path}")
    print(f"- {summary_path}")
    print(f"- {summary_text_path}")

    return {
        "target_gp_name": target_gp_name,
        "removed_features": len(result["removed_features"]),
        "selected_features": len(result["selected_features"]),
        "stop_reason": result["stop_reason"],
        "summary_path": str(summary_path),
    }


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
            print(f"Backward elimination for: {target_gp_name}")
            print(f"Config: {config_path}")
            print("=" * 80)

            try:
                summaries.append(run_single(args.alpha, args.output_dir))
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
            f"{summary['target_gp_name']}: removed={summary['removed_features']} | "
            f"selected={summary['selected_features']} | stop={summary['stop_reason']}"
        )

    if failures:
        print("\nCompleted with failures:")
        for target_gp_name, config_path, exc in failures:
            print(f"- {target_gp_name} ({config_path.name}): {exc}")
        return 1

    print("\nAll backward-elimination runs completed successfully.")
    return 0


def main():
    args = parse_args()
    if args.all:
        return run_all(args)

    run_single(args.alpha, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
