"""Plot the predicted lap-time series of a single driver over the sequential holdout.

For the configured Grand Prix this script rebuilds the same sequential split used by
the expanding-window models, predicts the holdout laps with the Linear Regression or
XGBoost final model, and plots, for one driver:

  - the actual lap times (markers coloured by Pirelli compound),
  - the predicted lap-time line,
  - an approximate 95% prediction band around the line (+/- z * residual_std).

Usage (Linux / macOS):
    CONFIG_PATH=configs/hungary.yaml python Scripts/Source/plot_driver_holdout_timeseries.py --driver VER
    CONFIG_PATH=configs/hungary.yaml python Scripts/Source/plot_driver_holdout_timeseries.py --driver NOR --model xgb
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from modeling_utils import (
    align_one_hot,
    build_lr_ew_model_paths,
    build_sequential_split,
    build_xgb_ew_model_paths,
    decode_step_key,
    fit_predict_linear_regression,
    load_cleaned_data,
    prepare_raw_features,
    resolve_repo_path,
    safe_gp_name,
    select_modeling_columns,
)

# Pirelli dry-compound colours (approximate official scheme by relative hardness).
COMPOUND_COLORS = {
    "C1": "#bdbdbd",
    "C2": "#f0d000",
    "C3": "#e10600",
    "C4": "#7a3fa0",
    "C5": "#e00084",
    "INTERMEDIATE": "#43b02a",
    "WET": "#0067ad",
}
DEFAULT_COLOR = "#1f77b4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--driver", required=True, help="Driver code to plot (e.g. VER, NOR, HAM).")
    parser.add_argument(
        "--model",
        choices=["lr", "xgb"],
        default="lr",
        help="Which final EW model to use for the predictions (default: lr).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Restrict the holdout series to a single season. Defaults to the most recent year available for the driver.",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=1.96,
        help="Multiplier of the residual standard deviation for the prediction band (default: 1.96 ~ 95%%).",
    )
    return parser.parse_args()


def build_blocks(df_base: pd.DataFrame, config: dict):
    """Rebuild the sequential modeling/holdout split shared by the EW models."""
    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    num_cols, cat_cols = select_modeling_columns(df_base, config)
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    (
        step_series,
        _step_min,
        _step_max,
        model_idx,
        holdout_idx,
        _holdout_start_step,
        _model_end_step,
        _total_steps,
    ) = build_sequential_split(df_base, valid_indices, float(config["holdout_ratio"]), lap_col)

    X_model_raw = X_raw.loc[model_idx].copy()
    y_model = y_raw.loc[model_idx].copy()

    X_holdout_raw = X_raw.loc[holdout_idx].copy()
    y_holdout = y_raw.loc[holdout_idx].copy()

    holdout_meta = pd.DataFrame(
        {
            "Driver": df_base.loc[holdout_idx, "Driver"].astype(str),
            "pirelliCompound": df_base.loc[holdout_idx, "pirelliCompound"].astype(str),
            "LapNumber": pd.to_numeric(df_base.loc[holdout_idx, "LapNumber"], errors="coerce"),
            "Year": pd.to_numeric(df_base.loc[holdout_idx, "Year"], errors="coerce").astype("Int64"),
            "TyreLife": pd.to_numeric(df_base.loc[holdout_idx, "TyreLife"], errors="coerce"),
            "step": step_series.loc[holdout_idx],
        },
        index=holdout_idx,
    )
    return X_model_raw, y_model, X_holdout_raw, y_holdout, holdout_meta, cat_cols


def prepare_xgb_design(X_raw: pd.DataFrame, cat_cols: list[str], metadata: dict) -> pd.DataFrame:
    feature_names = list(metadata["encoded_feature_names"])
    X_enc, _ = align_one_hot(X_raw, X_raw, cat_cols, drop_first=False)
    X_enc = X_enc.reindex(columns=feature_names, fill_value=0)
    medians = pd.Series(metadata["median_imputation_values"])
    return X_enc.fillna(medians).fillna(0)


def predict_holdout(
    model_choice: str,
    repo_root: Path,
    config: dict,
    X_model_raw: pd.DataFrame,
    y_model: pd.Series,
    X_holdout_raw: pd.DataFrame,
    cat_cols: list[str],
) -> np.ndarray:
    if model_choice == "lr":
        lr_model_path, _ = build_lr_ew_model_paths(repo_root, config)
        if not lr_model_path.exists():
            raise FileNotFoundError(
                f"Linear Regression EW model not found: {lr_model_path}\n"
                "Run Scripts/Source/model_lr_ew.py for this Grand Prix first."
            )
        # Refit the exact same deterministic pipeline used to produce the saved model.
        preds, *_ = fit_predict_linear_regression(X_model_raw, y_model, X_holdout_raw, cat_cols)
        return np.asarray(preds, dtype=float)

    xgb_model_path, xgb_metadata_path = build_xgb_ew_model_paths(repo_root, config)
    if not xgb_model_path.exists():
        raise FileNotFoundError(
            f"XGBoost EW model not found: {xgb_model_path}\n"
            "Run Scripts/Source/model_xgb_ew.py for this Grand Prix first."
        )
    if not xgb_metadata_path.exists():
        raise FileNotFoundError(f"XGBoost model metadata not found: {xgb_metadata_path}")
    booster = xgb.Booster()
    booster.load_model(str(xgb_model_path))
    metadata = json.loads(xgb_metadata_path.read_text(encoding="utf-8"))
    X_holdout_enc = prepare_xgb_design(X_holdout_raw, cat_cols, metadata)
    dmatrix = xgb.DMatrix(X_holdout_enc, feature_names=list(X_holdout_enc.columns))
    return np.asarray(booster.predict(dmatrix), dtype=float)


def plot_driver(
    target_gp_name: str,
    driver: str,
    model_choice: str,
    z: float,
    series: pd.DataFrame,
    residual_std: float,
    output_dir: Path,
    year: int,
) -> tuple[Path, Path]:
    laps = series["LapNumber"].to_numpy(dtype=float)
    actual = series["LapTime_seconds"].to_numpy(dtype=float)
    pred = series["prediction"].to_numpy(dtype=float)
    lower = pred - z * residual_std
    upper = pred + z * residual_std

    model_label = "Linear Regression" if model_choice == "lr" else "XGBoost"
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.fill_between(
        laps,
        lower,
        upper,
        color="0.6",
        alpha=0.25,
        label=f"~95% prediction band (±{z:.2f}·σ, σ={residual_std:.3f}s)",
        zorder=1,
    )
    ax.plot(laps, pred, color="#0b3d91", linewidth=2.0, label=f"Predicted ({model_label})", zorder=2)

    for compound in series["pirelliCompound"].unique():
        mask = series["pirelliCompound"].to_numpy() == compound
        ax.scatter(
            laps[mask],
            actual[mask],
            s=45,
            color=COMPOUND_COLORS.get(str(compound).upper(), DEFAULT_COLOR),
            edgecolor="black",
            linewidth=0.4,
            label=f"Actual — {compound}",
            zorder=3,
        )

    # Constrain the y-axis to a tight window (+/- 5 s) around the actual lap times so
    # the prediction error reads as a smaller visual deviation.
    y_low = float(np.nanmin(actual)) - 5.0
    y_high = float(np.nanmax(actual)) + 5.0
    ax.set_ylim(y_low, y_high)

    ax.set_xlabel("Lap", fontsize=28)
    ax.set_ylabel("Lap time (s)", fontsize=28)
    ax.tick_params(axis="both", labelsize=24)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="best", fontsize=20, framealpha=0.9)
    fig.tight_layout()

    safe_name = safe_gp_name(target_gp_name)
    stem = f"{safe_name}_{driver}_{model_choice}_holdout_timeseries"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    csv_path = output_dir / f"{stem}.csv"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")  # vector output for the article
    plt.close(fig)

    out = series[["Year", "LapNumber", "pirelliCompound", "TyreLife", "LapTime_seconds", "prediction"]].copy()
    out["lower_band"] = lower
    out["upper_band"] = upper
    out["residual"] = out["LapTime_seconds"] - out["prediction"]
    out.to_csv(csv_path, index=False)
    return png_path, pdf_path, csv_path


def main() -> None:
    args = parse_args()
    driver = args.driver.strip().upper()

    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))

    X_model_raw, y_model, X_holdout_raw, y_holdout, holdout_meta, cat_cols = build_blocks(
        laps_cleaned.copy(), config
    )

    preds = predict_holdout(args.model, repo_root, config, X_model_raw, y_model, X_holdout_raw, cat_cols)

    # Residual std over the whole holdout block defines the band width (matches std_holdout in the EW scripts).
    residuals = y_holdout.to_numpy(dtype=float) - preds
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    series = holdout_meta.copy()
    series["LapTime_seconds"] = y_holdout.to_numpy(dtype=float)
    series["prediction"] = preds

    driver_series = series[series["Driver"] == driver]
    if driver_series.empty:
        available = sorted(series["Driver"].unique())
        raise SystemExit(
            f"Driver {driver!r} has no laps in the holdout block.\n"
            f"Available drivers: {', '.join(available)}"
        )

    year = args.year
    if year is None:
        year = int(driver_series["Year"].max())
    driver_series = driver_series[driver_series["Year"] == year]
    if driver_series.empty:
        years = sorted(int(y) for y in series[series["Driver"] == driver]["Year"].dropna().unique())
        raise SystemExit(f"Driver {driver!r} has no holdout laps in {year}. Available years: {years}")

    driver_series = driver_series.sort_values("LapNumber", kind="mergesort").reset_index(drop=True)

    output_dir = (
        resolve_repo_path(repo_root, str(config["results_dir"])) / "driver_holdout_timeseries"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path, pdf_path, csv_path = plot_driver(
        target_gp_name, driver, args.model, args.z, driver_series, residual_std, output_dir, year
    )

    holdout_start = decode_step_key(int(holdout_meta["step"].min()))
    holdout_end = decode_step_key(int(holdout_meta["step"].max()))
    print(f"\n--- DRIVER HOLDOUT TIME SERIES: {target_gp_name} | {driver} | {args.model.upper()} ---")
    print(f"Holdout block: {holdout_start} -> {holdout_end} | total holdout records={len(series)}")
    print(f"Driver laps plotted ({year}): {len(driver_series)} | residual std (band)={residual_std:.4f}s")
    print(f"- {png_path}")
    print(f"- {pdf_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()
