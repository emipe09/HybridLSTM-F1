"""Residuals vs fitted (response) plots for the Linear Regression residuals.

For every Grand Prix this script reads the expanding-window Linear Regression holdout
predictions saved in ``Scripts/Results/lstm_hybrid/baseline`` (files
``*_lr_ew_holdout_predictions.csv`` with columns ``y_true`` and ``baseline``),
computes the residuals ``y_true - baseline`` and draws a residuals-vs-fitted plot
(residuals on the y-axis, fitted/predicted response on the x-axis) so that
homoscedasticity and the absence of structure in the residuals can be assessed
visually.

Outputs (under ``Scripts/Results/lr_residual_vs_fitted``):
  - one PNG/PDF per track,
  - a combined grid figure with all tracks,
  - a CSV per track with the fitted values and residuals.

Usage (Linux / macOS):
    python Scripts/Source/plot_lr_residuals_vs_fitted.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Holdout prediction files produced by the baseline Linear Regression (EW) pipeline.
PREDICTIONS_DIR = Path("Scripts/Results/lstm_hybrid/baseline")
PREDICTIONS_GLOB = "*_lr_ew_holdout_predictions.csv"
OUTPUT_SUBDIR = "lr_residual_vs_fitted"

TRUE_COL = "y_true"
PRED_COL = "baseline"


def repo_root() -> Path:
    # Scripts/Source/this_file.py -> repo root is two levels up.
    return Path(__file__).resolve().parents[2]


def pretty_track_name(stem: str) -> str:
    """Turn 'bahrain_grand_prix_lr_ew_holdout_predictions' into 'Bahrain Grand Prix'."""
    name = stem.replace("_lr_ew_holdout_predictions", "")
    return name.replace("_", " ").title()


def load_fitted_residuals(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (fitted values, residuals) with NaN rows dropped."""
    df = pd.read_csv(csv_path)
    missing = {TRUE_COL, PRED_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")
    y_true = pd.to_numeric(df[TRUE_COL], errors="coerce")
    fitted = pd.to_numeric(df[PRED_COL], errors="coerce")
    residuals = y_true - fitted
    mask = fitted.notna() & residuals.notna()
    return (
        fitted[mask].to_numpy(dtype=float),
        residuals[mask].to_numpy(dtype=float),
    )


def _trend_line(fitted: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binned-mean trend of the residuals to highlight any remaining structure."""
    order = np.argsort(fitted)
    x_sorted = fitted[order]
    y_sorted = residuals[order]
    n_bins = max(4, min(12, x_sorted.size // 15))
    edges = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x_sorted >= lo) & (x_sorted <= hi)
        if sel.any():
            centers.append(float((lo + hi) / 2))
            means.append(float(y_sorted[sel].mean()))
    return np.asarray(centers), np.asarray(means)


def _draw_residuals(
    ax: plt.Axes,
    fitted: np.ndarray,
    residuals: np.ndarray,
    track_name: str,
    n: int,
) -> None:
    ax.axhline(0.0, color="#e10600", linewidth=1.8, zorder=1, label="Zero residual")
    ax.scatter(
        fitted,
        residuals,
        s=22,
        color="#0b3d91",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.8,
        zorder=2,
    )
    if fitted.size >= 8:
        cx, cy = _trend_line(fitted, residuals)
        if cx.size >= 2:
            ax.plot(cx, cy, color="#f7a600", linewidth=1.6, zorder=3, label="Binned mean")

    rmse = float(np.sqrt(np.mean(residuals**2)))
    ax.set_title(f"{track_name}\nn = {n}  |  RMSE = {rmse:.3f}", fontsize=14)
    ax.set_xlabel("Fitted values (predicted response)", fontsize=12)
    ax.set_ylabel("Residuals (y_true - baseline)", fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="best", fontsize=9, framealpha=0.85)


def plot_single(
    track_name: str,
    fitted: np.ndarray,
    residuals: np.ndarray,
    output_dir: Path,
    file_stem: str,
) -> tuple[Path, Path, Path]:
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_residuals(ax, fitted, residuals, track_name, fitted.size)
    fig.tight_layout()

    png_path = output_dir / f"{file_stem}_resid_vs_fitted.png"
    pdf_path = output_dir / f"{file_stem}_resid_vs_fitted.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / f"{file_stem}_resid_vs_fitted.csv"
    pd.DataFrame({"fitted": fitted, "residual": residuals}).to_csv(csv_path, index=False)
    return png_path, pdf_path, csv_path


def plot_grid(
    tracks: list[tuple[str, np.ndarray, np.ndarray]], output_dir: Path
) -> tuple[Path, Path]:
    n = len(tracks)
    ncols = 3 if n > 4 else 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.0 * nrows), squeeze=False)

    for idx, (track_name, fitted, residuals) in enumerate(tracks):
        ax = axes[idx // ncols][idx % ncols]
        _draw_residuals(ax, fitted, residuals, track_name, fitted.size)

    # Hide any unused axes in the grid.
    for empty in range(n, nrows * ncols):
        axes[empty // ncols][empty % ncols].axis("off")

    fig.suptitle("Residuals vs fitted — Linear Regression holdout residuals", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png_path = output_dir / "all_tracks_resid_vs_fitted.png"
    pdf_path = output_dir / "all_tracks_resid_vs_fitted.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    root = repo_root()
    pred_dir = root / PREDICTIONS_DIR
    csv_files = sorted(pred_dir.glob(PREDICTIONS_GLOB))
    if not csv_files:
        raise SystemExit(f"No prediction files matching {PREDICTIONS_GLOB!r} found in {pred_dir}")

    output_dir = root / "Scripts" / "Results" / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks: list[tuple[str, np.ndarray, np.ndarray]] = []
    print("\n--- LINEAR REGRESSION RESIDUALS: RESIDUALS VS FITTED ---")
    for csv_path in csv_files:
        stem = csv_path.stem
        track_name = pretty_track_name(stem)
        fitted, residuals = load_fitted_residuals(csv_path)
        if residuals.size < 3:
            print(f"  ! Skipping {track_name}: only {residuals.size} residuals")
            continue
        file_stem = stem.replace("_lr_ew_holdout_predictions", "")
        png, pdf, csv_out = plot_single(track_name, fitted, residuals, output_dir, file_stem)
        tracks.append((track_name, fitted, residuals))
        print(f"  {track_name}: n={residuals.size} -> {png.name}, {pdf.name}, {csv_out.name}")

    if tracks:
        grid_png, grid_pdf = plot_grid(tracks, output_dir)
        print(f"\nCombined grid -> {grid_png.name}, {grid_pdf.name}")
    print(f"\nAll outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()
