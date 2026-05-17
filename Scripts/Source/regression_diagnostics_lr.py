"""Linear Regression residual diagnostics on the retrained modeling block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model_lr_sw import fit_predict_linear_regression
from modeling_utils import (
    align_one_hot,
    build_sequential_split,
    load_cleaned_data,
    prepare_raw_features,
    resolve_repo_path,
    safe_gp_name,
    select_modeling_columns,
)

try:
    import scipy.stats as stats
except ModuleNotFoundError:  # pragma: no cover - scipy is listed in project requirements.
    stats = None


GP_CONFIG_ENV = {
    "Bahrain Grand Prix": "configs/bahrain.yaml",
    "Saudi Arabian Grand Prix": "configs/saudi.yaml",
    "United States Grand Prix": "configs/usa.yaml",
    "Italian Grand Prix": "configs/italy.yaml",
    "Hungarian Grand Prix": "configs/hungary.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Linear Regression residual diagnostics on the first "
            "sequential modeling block after retraining on that block."
        )
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run diagnostics for every configured Grand Prix.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving them.",
    )
    return parser.parse_args()


def build_ordered_split(
    df_base: pd.DataFrame,
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    valid_indices: pd.Index,
    config: dict,
):
    lap_col = str(config["lap_col"])
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

    model_laps = lap_series.loc[model_idx]
    model_order_idx = model_laps.sort_values(kind="mergesort").index
    X_model_raw = X_raw.loc[model_order_idx].reset_index(drop=True)
    y_model = y_raw.loc[model_order_idx].reset_index(drop=True)
    lap_model = model_laps.loc[model_order_idx].reset_index(drop=True)

    holdout_laps = lap_series.loc[holdout_idx]
    holdout_order_idx = holdout_laps.sort_values(kind="mergesort").index

    split_info = {
        "total_laps": total_laps,
        "lap_min": lap_min,
        "lap_max": lap_max,
        "model_end_lap": model_end_lap,
        "holdout_start_lap": holdout_start_lap,
        "model_records": len(X_model_raw),
        "holdout_records": len(holdout_order_idx),
    }
    return X_model_raw, y_model, lap_model, split_info


def make_diagnostics_frame(
    lap_numbers: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    diagnostics = pd.DataFrame(
        {
            "LapNumber": lap_numbers.to_numpy(),
            "actual": y_true.to_numpy(),
            "predicted": np.asarray(y_pred, dtype=float),
        }
    )
    diagnostics["residual"] = diagnostics["actual"] - diagnostics["predicted"]
    diagnostics["abs_error"] = diagnostics["residual"].abs()
    diagnostics["squared_error"] = diagnostics["residual"] ** 2
    return diagnostics


def summarize_modeling_diagnostics(diagnostics: pd.DataFrame) -> dict:
    y_true = diagnostics["actual"].to_numpy()
    y_pred = diagnostics["predicted"].to_numpy()
    residuals = diagnostics["residual"].to_numpy()

    summary = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
        "residual_abs_max": float(np.max(np.abs(residuals))),
        "residual_skewness": float(pd.Series(residuals).skew()),
        "residual_kurtosis": float(pd.Series(residuals).kurtosis()),
    }

    if stats is not None and len(residuals) >= 3:
        shapiro_n = min(len(residuals), 5000)
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:shapiro_n])
        summary["shapiro_w"] = float(shapiro_stat)
        summary["shapiro_p_value"] = float(shapiro_p)

    return summary


def fit_statsmodels_ols(
    X_model_raw: pd.DataFrame,
    y_model: pd.Series,
    cat_cols: list[str],
    imputer,
    scaler,
    feature_names,
) -> tuple[pd.DataFrame, dict]:
    import statsmodels.api as sm

    X_model_enc, _ = align_one_hot(X_model_raw, X_model_raw, cat_cols, drop_first=True)
    X_model_enc = X_model_enc.reindex(columns=list(feature_names), fill_value=0)
    X_model_imp = imputer.transform(X_model_enc)
    X_model_scaled = scaler.transform(X_model_imp)

    X_design = pd.DataFrame(X_model_scaled, columns=list(feature_names), index=y_model.index)
    X_design = sm.add_constant(X_design, has_constant="add")
    ols_result = sm.OLS(y_model.to_numpy(dtype=float), X_design).fit()
    conf_int = ols_result.conf_int()

    coefficient_table = pd.DataFrame(
        {
            "encoded_feature": ols_result.params.index,
            "coefficient": ols_result.params.to_numpy(dtype=float),
            "std_error": ols_result.bse.to_numpy(dtype=float),
            "t_statistic": ols_result.tvalues.to_numpy(dtype=float),
            "p_value": ols_result.pvalues.to_numpy(dtype=float),
            "ci_lower": conf_int[0].to_numpy(dtype=float),
            "ci_upper": conf_int[1].to_numpy(dtype=float),
        }
    )
    metadata = {
        "n_observations": int(ols_result.nobs),
        "n_parameters": int(len(ols_result.params)),
        "matrix_rank": int(ols_result.df_model + 1),
        "degrees_of_freedom_residual": int(ols_result.df_resid),
        "p_value_distribution": "student_t",
        "coefficient_scale": "standardized_numeric_and_one_hot_design_with_intercept",
    }
    return ols_result, coefficient_table, metadata


def plot_model_coefficients(
    coefficient_table: pd.DataFrame,
    model_name: str,
    output_path: Path,
    show: bool,
) -> None:
    from matplotlib import pyplot as plt

    plot_data = coefficient_table[coefficient_table["encoded_feature"] != "const"].copy()
    plot_data = plot_data.sort_values("coefficient")

    fig_height = max(8, 0.28 * len(plot_data))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    colors = np.where(plot_data["coefficient"] >= 0, "steelblue", "indianred")
    ax.barh(plot_data["encoded_feature"], plot_data["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Model Coefficients: {model_name}")
    ax.set_xlabel("Coefficient on Standardized Design Matrix")
    ax.set_ylabel("Encoded Feature")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_regression_diagnostics(
    diagnostics: pd.DataFrame,
    model_name: str,
    output_path: Path,
    show: bool,
) -> None:
    import seaborn as sns
    from matplotlib import pyplot as plt

    residuals = diagnostics["residual"]
    y_true = diagnostics["actual"]
    y_pred = diagnostics["predicted"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Regression Diagnostics: {model_name}", fontsize=16)

    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0, 0], alpha=0.5, edgecolor=None)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0, 0].set_xlabel("Predicted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("1. Residuals vs Predicted (Homoscedasticity)")

    if stats is not None:
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, "SciPy is required for QQ-plot", ha="center", va="center")
    axes[0, 1].set_title("2. Normal QQ-Plot of Residuals")

    axes[1, 0].plot(residuals.to_numpy(), marker="o", linestyle="none", alpha=0.5, markersize=3)
    axes[1, 0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1, 0].set_xlabel("Modeling-Block Instance Number")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].set_title("3. Residuals vs Instance (Independence)")

    sns.scatterplot(x=y_true, y=y_pred, ax=axes[1, 1], alpha=0.5, edgecolor=None)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.2)
    axes[1, 1].set_xlabel("Actual Value (Target)")
    axes[1, 1].set_ylabel("Predicted Value")
    axes[1, 1].set_title("4. Actual vs Predicted")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_residual_distribution(
    diagnostics: pd.DataFrame,
    model_name: str,
    output_path: Path,
    show: bool,
) -> None:
    import seaborn as sns
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(diagnostics["residual"], bins=40, kde=True, color="steelblue", ax=ax)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero error")
    ax.set_title(f"Residual Distribution: {model_name}")
    ax.set_xlabel("Residual (y_true - y_pred) [seconds]")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run_single_diagnostic(show: bool) -> None:
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()

    target_col = str(config["target_col"])
    num_cols, cat_cols = select_modeling_columns(df_base, config)
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)
    X_model_raw, y_model, lap_model, split_info = build_ordered_split(
        df_base,
        X_raw,
        y_raw,
        valid_indices,
        config,
    )

    preds_modeling, model, imputer, scaler, feature_names = fit_predict_linear_regression(
        X_model_raw,
        y_model,
        X_model_raw,
        cat_cols,
    )
    diagnostics = make_diagnostics_frame(lap_model, y_model, preds_modeling)
    summary = summarize_modeling_diagnostics(diagnostics)
    ols_result, coefficient_table, coefficient_metadata = fit_statsmodels_ols(
        X_model_raw,
        y_model,
        cat_cols,
        imputer,
        scaler,
        feature_names,
    )

    safe_name = safe_gp_name(target_gp_name)
    output_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / "regression_diagnostics" / safe_name
    diagnostics_csv = output_dir / f"{safe_name}_lr_modeling_block_diagnostics.csv"
    coefficients_csv = output_dir / f"{safe_name}_lr_modeling_block_coefficients.csv"
    ols_summary_txt = output_dir / f"{safe_name}_lr_modeling_block_ols_summary.txt"
    summary_json = output_dir / f"{safe_name}_lr_modeling_block_summary.json"
    panel_png = output_dir / f"{safe_name}_lr_regression_diagnostics.png"
    histogram_png = output_dir / f"{safe_name}_lr_residual_distribution.png"
    coefficients_png = output_dir / f"{safe_name}_lr_model_coefficients.png"

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diagnostics_csv, index=False)
    coefficient_table.to_csv(coefficients_csv, index=False)
    ols_summary_txt.write_text(str(ols_result.summary()), encoding="utf-8")
    summary_payload = {
        "target_gp_name": target_gp_name,
        "model": "linear_regression",
        "target_col": target_col,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "feature_count_after_encoding": len(feature_names),
        "split": split_info,
        "diagnostic_block": "first_sequential_modeling_block_retrained_in_sample",
        "holdout_usage": "final sequential holdout is not used for residual diagnostics",
        "modeling_block_diagnostics": summary,
        "coefficient_p_values": coefficient_metadata,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    model_name = f"Linear Regression Retrained Modeling Block - {target_gp_name}"
    plot_regression_diagnostics(diagnostics, model_name, panel_png, show)
    plot_residual_distribution(diagnostics, model_name, histogram_png, show)
    plot_model_coefficients(coefficient_table, model_name, coefficients_png, show)

    print("\n--- LINEAR REGRESSION MODELING-BLOCK DIAGNOSTICS ---")
    print(f"Grand Prix: {target_gp_name}")
    print(
        f"Modeling block: laps {split_info['lap_min']}-{split_info['model_end_lap']} | "
        f"records={split_info['model_records']}"
    )
    print(
        f"Holdout block: laps {split_info['holdout_start_lap']}-{split_info['lap_max']} | "
        f"records={split_info['holdout_records']} | not used for diagnostics"
    )
    print("NOTE: diagnostics are in-sample for the model retrained on the 80% modeling block.")
    print(f"RMSE: {summary['rmse']:.4f}")
    print(f"MAE:  {summary['mae']:.4f}")
    print(f"R2:   {summary['r2']:.4f}")
    print(
        f"Residuals: mean={summary['residual_mean']:.4f} | "
        f"std={summary['residual_std']:.4f} | "
        f"abs max={summary['residual_abs_max']:.4f}"
    )
    if "shapiro_p_value" in summary:
        print(f"Shapiro-Wilk normality p-value: {summary['shapiro_p_value']:.4g}")
    print("\n--- OLS coefficient summary ---")
    print(str(ols_result.summary()))
    print("\nSaved outputs:")
    print(f"- {diagnostics_csv}")
    print(f"- {coefficients_csv}")
    print(f"- {ols_summary_txt}")
    print(f"- {summary_json}")
    print(f"- {panel_png}")
    print(f"- {histogram_png}")
    print(f"- {coefficients_png}")


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    if not args.all:
        run_single_diagnostic(show=args.show)
        return

    import os

    original_config_path = os.environ.get("CONFIG_PATH")
    original_target_gp_name = os.environ.get("TARGET_GP_NAME")
    try:
        for gp_name, config_path in GP_CONFIG_ENV.items():
            print(f"\n=== Running diagnostics for {gp_name} ===")
            os.environ["CONFIG_PATH"] = config_path
            os.environ.pop("TARGET_GP_NAME", None)
            run_single_diagnostic(show=args.show)
    finally:
        if original_config_path is None:
            os.environ.pop("CONFIG_PATH", None)
        else:
            os.environ["CONFIG_PATH"] = original_config_path
        if original_target_gp_name is None:
            os.environ.pop("TARGET_GP_NAME", None)
        else:
            os.environ["TARGET_GP_NAME"] = original_target_gp_name


if __name__ == "__main__":
    main()
