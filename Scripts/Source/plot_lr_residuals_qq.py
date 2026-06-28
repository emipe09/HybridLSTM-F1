"""Normal quantile-quantile (Q-Q) plots for the Linear Regression residuals.

For every Grand Prix this script reads the expanding-window Linear Regression holdout
predictions saved in ``Scripts/Results/lstm_hybrid/baseline`` (files
``*_lr_ew_holdout_predictions.csv`` with columns ``y_true`` and ``baseline``),
computes the residuals ``y_true - baseline`` and draws a normal Q-Q plot so the
normality of the regression residuals can be assessed visually.

Outputs (under ``Scripts/Results/lr_residual_qq``):
  - one PNG/PDF per track,
  - a combined grid figure with all tracks,
  - a CSV per track with the standardised residuals and theoretical quantiles.

Usage (Linux / macOS):
    python Scripts/Source/plot_lr_residuals_qq.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Holdout prediction files produced by the baseline Linear Regression (EW) pipeline.
PREDICTIONS_DIR = Path("Scripts/Results/lstm_hybrid/baseline")
PREDICTIONS_GLOB = "*_lr_ew_holdout_predictions.csv"
OUTPUT_SUBDIR = "lr_residual_qq"

TRUE_COL = "y_true"
PRED_COL = "baseline"


def repo_root() -> Path:
    # Scripts/Source/this_file.py -> repo root is two levels up.
    return Path(__file__).resolve().parents[2]


def pretty_track_name(stem: str) -> str:
    """Turn 'bahrain_grand_prix_lr_ew_holdout_predictions' into 'Bahrain Grand Prix'."""
    name = stem.replace("_lr_ew_holdout_predictions", "")
    return name.replace("_", " ").title()


def load_residuals(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    missing = {TRUE_COL, PRED_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")
    residuals = pd.to_numeric(df[TRUE_COL], errors="coerce") - pd.to_numeric(
        df[PRED_COL], errors="coerce"
    )
    return residuals.dropna().to_numpy(dtype=float)


def qq_points(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return theoretical quantiles, ordered standardised residuals and the fit line.

    Residuals are standardised (z-scores) so the reference line is the identity
    line y = x of the standard normal distribution.
    """
    (theoretical, ordered), (slope, intercept, _r) = stats.probplot(residuals, dist="norm")
    mean = float(np.mean(residuals))
    std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 1.0
    std = std if std > 0 else 1.0
    standardised = (ordered - mean) / std
    return theoretical, standardised, slope, intercept


def plot_single(
    track_name: str,
    residuals: np.ndarray,
    output_dir: Path,
    file_stem: str,
) -> tuple[Path, Path, Path]:
    theoretical, standardised, _slope, _intercept = qq_points(residuals)
    shapiro_p = float(stats.shapiro(residuals).pvalue) if 3 <= residuals.size <= 5000 else float("nan")

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_qq(ax, theoretical, standardised, track_name, residuals.size, shapiro_p)
    fig.tight_layout()

    png_path = output_dir / f"{file_stem}_qq.png"
    pdf_path = output_dir / f"{file_stem}_qq.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / f"{file_stem}_qq.csv"
    pd.DataFrame(
        {"theoretical_quantile": theoretical, "standardised_residual": standardised}
    ).to_csv(csv_path, index=False)
    return png_path, pdf_path, csv_path


def _draw_qq(
    ax: plt.Axes,
    theoretical: np.ndarray,
    standardised: np.ndarray,
    track_name: str,
    n: int,
    shapiro_p: float,
) -> None:
    lim_low = float(min(theoretical.min(), standardised.min()))
    lim_high = float(max(theoretical.max(), standardised.max()))
    ax.plot(
        [lim_low, lim_high],
        [lim_low, lim_high],
        color="#e10600",
        linewidth=1.8,
        zorder=1,
        label="Normal reference",
    )
    ax.scatter(
        theoretical,
        standardised,
        s=22,
        color="#0b3d91",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.8,
        zorder=2,
    )
    subtitle = f"n = {n}"
    if not math.isnan(shapiro_p):
        subtitle += f"  |  Shapiro–Wilk p = {shapiro_p:.3g}"
    ax.set_title(f"{track_name}\n{subtitle}", fontsize=14)
    ax.set_xlabel("Theoretical quantiles (N(0,1))", fontsize=12)
    ax.set_ylabel("Standardised residuals", fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")


def plot_grid(tracks: list[tuple[str, np.ndarray]], output_dir: Path) -> tuple[Path, Path]:
    n = len(tracks)
    ncols = 3 if n > 4 else 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows), squeeze=False)

    for idx, (track_name, residuals) in enumerate(tracks):
        ax = axes[idx // ncols][idx % ncols]
        theoretical, standardised, _s, _i = qq_points(residuals)
        shapiro_p = (
            float(stats.shapiro(residuals).pvalue) if 3 <= residuals.size <= 5000 else float("nan")
        )
        _draw_qq(ax, theoretical, standardised, track_name, residuals.size, shapiro_p)

    # Hide any unused axes in the grid.
    for empty in range(n, nrows * ncols):
        axes[empty // ncols][empty % ncols].axis("off")

    fig.suptitle("Normal Q-Q plots — Linear Regression holdout residuals", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png_path = output_dir / "all_tracks_qq.png"
    pdf_path = output_dir / "all_tracks_qq.pdf"
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

    tracks: list[tuple[str, np.ndarray]] = []
    print("\n--- LINEAR REGRESSION RESIDUALS: NORMAL Q-Q PLOTS ---")
    for csv_path in csv_files:
        stem = csv_path.stem
        track_name = pretty_track_name(stem)
        residuals = load_residuals(csv_path)
        if residuals.size < 3:
            print(f"  ! Skipping {track_name}: only {residuals.size} residuals")
            continue
        file_stem = stem.replace("_lr_ew_holdout_predictions", "")
        png, pdf, csv_out = plot_single(track_name, residuals, output_dir, file_stem)
        tracks.append((track_name, residuals))
        print(f"  {track_name}: n={residuals.size} -> {png.name}, {pdf.name}, {csv_out.name}")

    if tracks:
        grid_png, grid_pdf = plot_grid(tracks, output_dir)
        print(f"\nCombined grid -> {grid_png.name}, {grid_pdf.name}")
    print(f"\nAll outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()
