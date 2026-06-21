"""Unified model interpretability reports for Linear Regression and XGBoost."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from modeling_utils import (
    align_one_hot,
    build_lr_ew_model_paths,
    build_sequential_split,
    build_xgb_ew_model_paths,
    load_cleaned_data,
    load_simple_yaml,
    prepare_raw_features,
    resolve_repo_path,
    safe_gp_name,
    select_modeling_columns,
)


GP_CONFIG_ENV = {
    "Bahrain Grand Prix": "configs/bahrain.yaml",
    "Saudi Arabian Grand Prix": "configs/saudi.yaml",
    "United States Grand Prix": "configs/usa.yaml",
    "Italian Grand Prix": "configs/italy.yaml",
    "Hungarian Grand Prix": "configs/hungary.yaml",
}

IMPORTANCE_TYPES = ["weight", "gain", "cover", "total_gain", "total_cover"]
TOP_IMPORTANCE_FEATURES = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the saved final Linear Regression and XGBoost models for each "
            "Grand Prix and export LR coefficients, XGBoost importance, and SHAP values."
        )
    )
    parser.add_argument("--all", action="store_true", help="Run interpretability for every configured Grand Prix.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of features to show in compact SHAP summary plots. CSV outputs always include all features.",
    )
    parser.add_argument(
        "--force-index",
        type=int,
        default=None,
        help=(
            "Zero-based row index from the sequential modeling block to explain with a SHAP force plot. "
            "When omitted, the row with prediction closest to the median modeling-block prediction is used."
        ),
    )
    return parser.parse_args()


def format_force_plot_feature_values(feature_values: pd.Series) -> pd.Series:
    """Round numeric feature values for force-plot labels without changing SHAP values."""
    formatted = feature_values.copy()
    for feature_name, value in formatted.items():
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            formatted.loc[feature_name] = round(float(value), 2)
    return formatted


def build_modeling_block(df_base: pd.DataFrame, config: dict):
    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    num_cols, cat_cols = select_modeling_columns(df_base, config)
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    (
        lap_series,
        _lap_min,
        _lap_max,
        model_idx,
        _holdout_idx,
        _holdout_start_lap,
        _model_end_lap,
        _total_laps,
    ) = build_sequential_split(df_base, valid_indices, float(config["holdout_ratio"]), lap_col)

    X_model_raw = X_raw.loc[model_idx].copy()
    y_model = y_raw.loc[model_idx].copy()

    model_laps = lap_series.loc[model_idx]
    model_order_idx = model_laps.sort_values(kind="mergesort").index
    X_model_raw = X_model_raw.loc[model_order_idx].reset_index(drop=True)
    y_model = y_model.loc[model_order_idx].reset_index(drop=True)
    return X_model_raw, y_model, num_cols, cat_cols


def load_lr_payload(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Linear Regression EW model not found: {model_path}\n"
            "Run Scripts/Source/model_lr_ew.py for this Grand Prix first."
        )
    with model_path.open("rb") as file:
        return pickle.load(file)


def load_xgb_booster(model_path: Path) -> xgb.Booster:
    if not model_path.exists():
        raise FileNotFoundError(
            f"XGBoost EW model not found: {model_path}\n"
            "Run Scripts/Source/model_xgb_ew.py for this Grand Prix first."
        )
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


def save_lr_coefficients(lr_payload: dict, output_dir: Path, safe_name: str) -> tuple[Path, Path]:
    model = lr_payload["model"]
    feature_names = list(lr_payload["feature_names"])
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": np.asarray(model.coef_, dtype=float),
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("abs_coefficient", ascending=False)

    csv_path = output_dir / f"{safe_name}_lr_coefficients.csv"
    png_path = output_dir / f"{safe_name}_lr_coefficients.png"
    coefficients.to_csv(csv_path, index=False)

    plot_data = coefficients.head(TOP_IMPORTANCE_FEATURES).sort_values("coefficient")
    fig_height = max(8, 0.28 * len(plot_data))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    colors = np.where(plot_data["coefficient"] >= 0, "steelblue", "indianred")
    ax.barh(plot_data["feature"], plot_data["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Linear Regression Coefficients")
    ax.set_xlabel("Coefficient on Standardized Encoded Design")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def prepare_xgb_design(X_model_raw: pd.DataFrame, cat_cols: list[str], metadata: dict) -> pd.DataFrame:
    feature_names = list(metadata["encoded_feature_names"])
    X_model_enc, _ = align_one_hot(X_model_raw, X_model_raw, cat_cols, drop_first=False)
    X_model_enc = X_model_enc.reindex(columns=feature_names, fill_value=0)
    medians = pd.Series(metadata["median_imputation_values"])
    return X_model_enc.fillna(medians).fillna(0)


def save_xgb_importance(booster: xgb.Booster, feature_names: list[str], output_dir: Path, safe_name: str) -> tuple[Path, Path]:
    rows = pd.DataFrame({"feature": feature_names})
    for importance_type in IMPORTANCE_TYPES:
        score = booster.get_score(importance_type=importance_type)
        rows[importance_type] = rows["feature"].map(score).fillna(0.0).astype(float)

    rows = rows.sort_values(["gain", "total_gain", "weight"], ascending=False)
    csv_path = output_dir / f"{safe_name}_xgb_feature_importance.csv"
    png_path = output_dir / f"{safe_name}_xgb_gain_importance.png"
    rows.to_csv(csv_path, index=False)

    plot_data = rows.head(TOP_IMPORTANCE_FEATURES).sort_values("gain")
    fig_height = max(8, 0.28 * len(plot_data))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(plot_data["feature"], plot_data["gain"], color="seagreen")
    ax.set_title("XGBoost Feature Importance by Gain")
    ax.set_xlabel("Gain")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def save_xgb_shap(
    booster: xgb.Booster,
    X_model_enc: pd.DataFrame,
    X_model_raw: pd.DataFrame,
    y_model: pd.Series,
    output_dir: Path,
    safe_name: str,
    top_n: int,
    force_index: int | None,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_model_enc)
    shap_values = np.asarray(shap_values, dtype=float)
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    values_path = output_dir / f"{safe_name}_xgb_shap_values.csv"
    summary_path = output_dir / f"{safe_name}_xgb_shap_summary.csv"
    beeswarm_path = output_dir / f"{safe_name}_xgb_shap_summary.png"
    bar_path = output_dir / f"{safe_name}_xgb_shap_bar.png"
    force_png_path = output_dir / f"{safe_name}_xgb_shap_force_plot.png"
    force_html_path = output_dir / f"{safe_name}_xgb_shap_force_plot.html"
    force_csv_path = output_dir / f"{safe_name}_xgb_shap_force_plot_contributions.csv"

    pd.DataFrame(shap_values, columns=X_model_enc.columns).to_csv(values_path, index=False)

    summary = pd.DataFrame(
        {
            "feature": X_model_enc.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
            "std_shap": shap_values.std(axis=0, ddof=1),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    summary.to_csv(summary_path, index=False)

    shap.summary_plot(
        shap_values,
        X_model_enc,
        max_display=top_n,
        show=False,
        plot_size=(9, 7),
    )
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=180, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values,
        X_model_enc,
        plot_type="bar",
        max_display=top_n,
        show=False,
        plot_size=(9, 7),
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close()

    predictions = booster.predict(
        xgb.DMatrix(X_model_enc, feature_names=list(X_model_enc.columns))
    )
    if force_index is None:
        force_index = int(np.argmin(np.abs(predictions - np.median(predictions))))
    if force_index < 0 or force_index >= len(X_model_enc):
        raise ValueError(
            f"--force-index must be between 0 and {len(X_model_enc) - 1}; "
            f"got {force_index}."
        )

    force_values = shap_values[force_index]
    force_features = X_model_enc.iloc[force_index]
    force_features_plot = format_force_plot_feature_values(force_features)
    force_raw_features = X_model_raw.iloc[force_index]
    force_actual = float(y_model.iloc[force_index])
    force_prediction = float(predictions[force_index])
    force_residual = force_actual - force_prediction
    force_contributions = pd.DataFrame(
        {
            "feature": X_model_enc.columns,
            "feature_value": force_features.to_numpy(),
            "shap_value": force_values,
            "abs_shap_value": np.abs(force_values),
        }
    ).sort_values("abs_shap_value", ascending=False)
    force_contributions.insert(0, "row_index", force_index)
    force_contributions.insert(1, "base_value", base_value)
    force_contributions.insert(2, "prediction", force_prediction)
    force_contributions.insert(3, "actual_lap_time", force_actual)
    force_contributions.insert(4, "residual_actual_minus_predicted", force_residual)
    force_contributions.insert(5, "driver", force_raw_features.get("Driver", ""))
    force_contributions.insert(6, "team", force_raw_features.get("Team", ""))
    force_contributions.insert(7, "year", force_raw_features.get("Year", ""))
    force_contributions.insert(8, "lap_number", force_raw_features.get("LapNumber", ""))
    force_contributions.insert(9, "pirelli_compound", force_raw_features.get("pirelliCompound", ""))
    force_contributions.to_csv(force_csv_path, index=False)

    shap.force_plot(
        base_value,
        force_values,
        force_features_plot,
        matplotlib=True,
        show=False,
    )
    force_context = (
        f"Driver: {force_raw_features.get('Driver', 'n/a')} | "
        f"Team: {force_raw_features.get('Team', 'n/a')} | "
        f"Lap: {force_raw_features.get('LapNumber', 'n/a')} | "
        f"Year: {force_raw_features.get('Year', 'n/a')} | "
        f"Actual: {force_actual:.3f}s | "
        f"Predicted: {force_prediction:.3f}s | "
        f"Residual: {force_residual:.3f}s"
    )
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02, force_context, ha="center", va="bottom", fontsize=9)
    plt.savefig(force_png_path, dpi=180, bbox_inches="tight")
    plt.close()

    force_plot = shap.force_plot(
        base_value,
        force_values,
        force_features_plot,
        matplotlib=False,
    )
    shap.save_html(str(force_html_path), force_plot)
    html = force_html_path.read_text(encoding="utf-8")
    html_context = (
        "<p style='font-family: Arial, sans-serif; margin-top: 24px; text-align: center;'>"
        f"<strong>Driver:</strong> {force_raw_features.get('Driver', 'n/a')} | "
        f"<strong>Team:</strong> {force_raw_features.get('Team', 'n/a')} | "
        f"<strong>Lap:</strong> {force_raw_features.get('LapNumber', 'n/a')} | "
        f"<strong>Year:</strong> {force_raw_features.get('Year', 'n/a')} | "
        f"<strong>Actual:</strong> {force_actual:.3f}s | "
        f"<strong>Predicted:</strong> {force_prediction:.3f}s | "
        f"<strong>Residual:</strong> {force_residual:.3f}s</p>"
    )
    force_html_path.write_text(html.replace("</body>", f"{html_context}</body>", 1), encoding="utf-8")

    return (
        values_path,
        summary_path,
        beeswarm_path,
        bar_path,
        force_png_path,
        force_html_path,
        force_csv_path,
    )


def run_single_interpretability(
    target_gp_name: str,
    config: dict,
    repo_root: Path,
    laps_cleaned: pd.DataFrame,
    top_n: int,
    force_index: int | None,
):
    safe_name = safe_gp_name(target_gp_name)
    output_dir = (
        resolve_repo_path(repo_root, str(config["results_dir"]))
        / "model_interpretability"
        / safe_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    X_model_raw, y_model, _num_cols, cat_cols = build_modeling_block(laps_cleaned.copy(), config)

    lr_model_path, lr_metadata_path = build_lr_ew_model_paths(repo_root, config)
    xgb_model_path, xgb_metadata_path = build_xgb_ew_model_paths(repo_root, config)

    lr_payload = load_lr_payload(lr_model_path)
    lr_coefficients_csv, lr_coefficients_png = save_lr_coefficients(lr_payload, output_dir, safe_name)

    booster = load_xgb_booster(xgb_model_path)
    if not xgb_metadata_path.exists():
        raise FileNotFoundError(f"XGBoost model metadata not found: {xgb_metadata_path}")
    xgb_metadata = json.loads(xgb_metadata_path.read_text(encoding="utf-8"))
    X_model_enc = prepare_xgb_design(X_model_raw, cat_cols, xgb_metadata)

    xgb_importance_csv, xgb_importance_png = save_xgb_importance(
        booster,
        list(X_model_enc.columns),
        output_dir,
        safe_name,
    )
    (
        shap_values_csv,
        shap_summary_csv,
        shap_summary_png,
        shap_bar_png,
        shap_force_png,
        shap_force_html,
        shap_force_csv,
    ) = save_xgb_shap(
        booster,
        X_model_enc,
        X_model_raw,
        y_model,
        output_dir,
        safe_name,
        top_n,
        force_index,
    )

    manifest = {
        "target_gp_name": target_gp_name,
        "validation_protocol": "expanding_window",
        "source_models": {
            "linear_regression_ew": str(lr_model_path),
            "linear_regression_ew_metadata": str(lr_metadata_path),
            "xgboost_ew": str(xgb_model_path),
            "xgboost_ew_metadata": str(xgb_metadata_path),
        },
        "diagnostic_block": "first_sequential_modeling_block",
        "outputs": {
            "lr_coefficients_csv": str(lr_coefficients_csv),
            "lr_coefficients_png": str(lr_coefficients_png),
            "xgb_importance_csv": str(xgb_importance_csv),
            "xgb_importance_png": str(xgb_importance_png),
            "xgb_shap_values_csv": str(shap_values_csv),
            "xgb_shap_summary_csv": str(shap_summary_csv),
            "xgb_shap_summary_png": str(shap_summary_png),
            "xgb_shap_bar_png": str(shap_bar_png),
            "xgb_shap_force_png": str(shap_force_png),
            "xgb_shap_force_html": str(shap_force_html),
            "xgb_shap_force_contributions_csv": str(shap_force_csv),
        },
    }
    manifest_path = output_dir / f"{safe_name}_model_interpretability_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n--- MODEL INTERPRETABILITY: {target_gp_name} ---")
    print(f"Saved outputs under: {output_dir}")
    print(f"- {lr_coefficients_csv}")
    print(f"- {xgb_importance_csv}")
    print(f"- {shap_summary_csv}")
    print(f"- {shap_force_png}")
    print(f"- {shap_force_html}")
    print(f"- {manifest_path}")


def run_from_current_environment(top_n: int, force_index: int | None) -> None:
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    run_single_interpretability(target_gp_name, config, repo_root, laps_cleaned, top_n, force_index)


def run_all(top_n: int, force_index: int | None) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    original_config_path = os.environ.get("CONFIG_PATH")
    original_target_gp_name = os.environ.get("TARGET_GP_NAME")
    try:
        for gp_name, config_path in GP_CONFIG_ENV.items():
            os.environ["CONFIG_PATH"] = config_path
            os.environ.pop("TARGET_GP_NAME", None)
            target_gp_name, config, _repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
            if target_gp_name != gp_name:
                raise ValueError(f"Expected {gp_name}, got {target_gp_name}.")
            run_single_interpretability(target_gp_name, config, repo_root, laps_cleaned, top_n, force_index)
    finally:
        if original_config_path is None:
            os.environ.pop("CONFIG_PATH", None)
        else:
            os.environ["CONFIG_PATH"] = original_config_path
        if original_target_gp_name is None:
            os.environ.pop("TARGET_GP_NAME", None)
        else:
            os.environ["TARGET_GP_NAME"] = original_target_gp_name


def main() -> None:
    args = parse_args()
    if args.all:
        run_all(args.top_n, args.force_index)
    else:
        run_from_current_environment(args.top_n, args.force_index)


if __name__ == "__main__":
    main()
