"""Correlation between lap time and the LSTM-hybrid residuals (validation and test).

For every Grand Prix this script reads the per-row hybrid predictions exported by
``model_lstm_hybrid.py`` under ``Scripts/Results/lstm_hybrid/baseline`` and, for each
split, computes the residual ``residual = y_true - hybrid_pred`` and correlates it with
the actual lap time (``y_true``). It draws a residual-vs-laptime scatter (lap time on the
x-axis, residual on the y-axis) annotated with Pearson's r and Spearman's rho, side by
side for the validation and the test (holdout) splits.

Prediction sources (per circuit, ``{safe_gp_name}_{feature_mode}_{baseline}_*``):
  - test  -> ``*_hybrid_holdout_predictions.csv``     (columns: y_true, hybrid_pred, ...)
  - val   -> ``*_hybrid_validation_predictions.csv``  (columns: LapNumber, y_true, hybrid_pred)

The validation CSV is produced by the current ``model_lstm_hybrid.py``. If it is absent
(circuit not re-run since that export was added) the validation panel is skipped and only
the test split is plotted; re-run the hybrid for that circuit to populate it.

Note on interpretation: because ``residual = y_true - hybrid_pred``, a non-zero
correlation with ``y_true`` is partly mechanical. A clear positive slope means the model
regresses toward the mean (under-predicting slow laps and over-predicting fast laps); a
flat cloud around zero is the desired, unbiased behaviour.

Outputs (under ``Scripts/Results/hybrid_residual_vs_laptime``):
  - one PNG/PDF per track (validation | test panels),
  - combined grids across tracks (one per split present),
  - a per-track CSV with split / laptime / residual,
  - a summary CSV with Pearson and Spearman correlations per track and split.

Usage:
    python Scripts/Source/plot_hybrid_residuals_vs_laptime.py
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

PREDICTIONS_DIR = Path("Scripts/Results/lstm_hybrid/baseline")
HOLDOUT_GLOB = "*_hybrid_holdout_predictions.csv"
OUTPUT_SUBDIR = "hybrid_residual_vs_laptime"

TRUE_COL = "y_true"
PRED_COL = "hybrid_pred"

# split key -> (label, point colour)
SPLITS = {
    "validation": ("Validation", "#0b3d91"),
    "test": ("Test (holdout)", "#e10600"),
}


def repo_root() -> Path:
    # Scripts/Source/this_file.py -> repo root is two levels up.
    return Path(__file__).resolve().parents[2]


def pretty_track_name(holdout_stem: str) -> str:
    """'bahrain_grand_prix_full_embedding_lr_ew_hybrid_holdout_predictions' -> 'Bahrain Grand Prix'."""
    name = holdout_stem
    for suffix in ("_hybrid_holdout_predictions",):
        name = name.replace(suffix, "")
    # Drop the trailing '<feature_mode>_<baseline>' tags, keeping the GP name.
    for tag in ("_full_embedding", "_auxiliary_embedding", "_minimal_embedding"):
        name = name.replace(tag, "")
    for tag in ("_lr_ew", "_xgb_ew"):
        name = name.replace(tag, "")
    return name.replace("_", " ").title()


def load_laptime_residual(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (lap_time, residual) = (y_true, y_true - hybrid_pred) with NaN rows dropped."""
    df = pd.read_csv(csv_path)
    missing = {TRUE_COL, PRED_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")
    y_true = pd.to_numeric(df[TRUE_COL], errors="coerce")
    pred = pd.to_numeric(df[PRED_COL], errors="coerce")
    residual = y_true - pred
    mask = y_true.notna() & residual.notna()
    return y_true[mask].to_numpy(dtype=float), residual[mask].to_numpy(dtype=float)


def correlations(lap_time: np.ndarray, residual: np.ndarray) -> dict:
    """Pearson and Spearman correlation between lap time and residual (NaN-safe)."""
    out = {"pearson_r": float("nan"), "pearson_p": float("nan"),
           "spearman_rho": float("nan"), "spearman_p": float("nan")}
    if lap_time.size >= 3 and np.ptp(lap_time) > 0 and np.ptp(residual) > 0:
        pr = stats.pearsonr(lap_time, residual)
        sr = stats.spearmanr(lap_time, residual)
        out["pearson_r"], out["pearson_p"] = float(pr[0]), float(pr[1])
        out["spearman_rho"], out["spearman_p"] = float(sr[0]), float(sr[1])
    return out


def _trend_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binned-mean trend of the residuals to highlight any remaining structure."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    n_bins = max(4, min(12, x_sorted.size // 15))
    edges = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x_sorted >= lo) & (x_sorted <= hi)
        if sel.any():
            centers.append(float((lo + hi) / 2))
            means.append(float(y_sorted[sel].mean()))
    return np.asarray(centers), np.asarray(means)


def _draw_panel(ax: plt.Axes, lap_time: np.ndarray, residual: np.ndarray,
                title: str, colour: str) -> dict:
    stats_d = correlations(lap_time, residual)
    ax.axhline(0.0, color="black", linewidth=1.2, zorder=1, label="Zero residual")
    ax.scatter(lap_time, residual, s=22, color=colour, edgecolor="black",
               linewidth=0.3, alpha=0.75, zorder=2)

    if lap_time.size >= 2 and np.ptp(lap_time) > 0:
        slope, intercept = np.polyfit(lap_time, residual, 1)
        xs = np.array([lap_time.min(), lap_time.max()])
        ax.plot(xs, slope * xs + intercept, color="#f7a600", linewidth=1.8,
                zorder=3, label="OLS fit")
    if lap_time.size >= 8:
        cx, cy = _trend_line(lap_time, residual)
        if cx.size >= 2:
            ax.plot(cx, cy, color="#2a9d8f", linewidth=1.4, linestyle="--",
                    zorder=4, label="Binned mean")

    annot = (
        f"Pearson r = {stats_d['pearson_r']:.3f} (p = {stats_d['pearson_p']:.1e})\n"
        f"Spearman ρ = {stats_d['spearman_rho']:.3f} (p = {stats_d['spearman_p']:.1e})"
    )
    ax.text(0.03, 0.97, annot, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.set_title(f"{title}\nn = {lap_time.size}", fontsize=13)
    ax.set_xlabel("Lap time (s)", fontsize=12)
    ax.set_ylabel("Residual (y_true - hybrid_pred) [s]", fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    return stats_d


def plot_track(track_name: str, panels: list[tuple[str, np.ndarray, np.ndarray]],
               output_dir: Path, file_stem: str) -> tuple[Path, list[dict]]:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 6.0), squeeze=False)
    summary = []
    for idx, (split_key, lap_time, residual) in enumerate(panels):
        label, colour = SPLITS[split_key]
        stats_d = _draw_panel(axes[0][idx], lap_time, residual,
                              f"{track_name} — {label}", colour)
        summary.append({"track": track_name, "split": split_key, "n": int(lap_time.size), **stats_d})
    fig.suptitle(f"Lap time vs hybrid residual — {track_name}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png_path = output_dir / f"{file_stem}_resid_vs_laptime.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"{file_stem}_resid_vs_laptime.pdf", bbox_inches="tight")
    plt.close(fig)

    # Tidy per-track CSV (split / laptime / residual).
    rows = []
    for split_key, lap_time, residual in panels:
        rows.append(pd.DataFrame({"split": split_key, "lap_time": lap_time, "residual": residual}))
    pd.concat(rows, ignore_index=True).to_csv(
        output_dir / f"{file_stem}_resid_vs_laptime.csv", index=False
    )
    return png_path, summary


def plot_split_grid(split_key: str, tracks: list[tuple[str, np.ndarray, np.ndarray]],
                    output_dir: Path) -> Path:
    label, colour = SPLITS[split_key]
    n = len(tracks)
    ncols = 3 if n > 4 else 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.2 * nrows), squeeze=False)
    for idx, (track_name, lap_time, residual) in enumerate(tracks):
        _draw_panel(axes[idx // ncols][idx % ncols], lap_time, residual, track_name, colour)
    for empty in range(n, nrows * ncols):
        axes[empty // ncols][empty % ncols].axis("off")
    fig.suptitle(f"Lap time vs hybrid residual — {label}", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png_path = output_dir / f"all_tracks_{split_key}_resid_vs_laptime.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"all_tracks_{split_key}_resid_vs_laptime.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    root = repo_root()
    pred_dir = root / PREDICTIONS_DIR
    holdout_files = sorted(pred_dir.glob(HOLDOUT_GLOB))
    if not holdout_files:
        raise SystemExit(f"No prediction files matching {HOLDOUT_GLOB!r} found in {pred_dir}")

    output_dir = root / "Scripts" / "Results" / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    grids: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {"validation": [], "test": []}
    summary_rows: list[dict] = []

    print("\n--- HYBRID RESIDUALS: LAP TIME vs RESIDUAL ---")
    for holdout_path in holdout_files:
        track_name = pretty_track_name(holdout_path.stem)
        file_stem = holdout_path.stem.replace("_hybrid_holdout_predictions", "")
        panels: list[tuple[str, np.ndarray, np.ndarray]] = []

        # Validation (optional) first so it renders on the left, then test.
        val_path = holdout_path.with_name(
            holdout_path.name.replace("_hybrid_holdout_predictions", "_hybrid_validation_predictions")
        )
        if val_path.exists():
            lt_v, res_v = load_laptime_residual(val_path)
            if res_v.size >= 3:
                panels.append(("validation", lt_v, res_v))
                grids["validation"].append((track_name, lt_v, res_v))

        lt_t, res_t = load_laptime_residual(holdout_path)
        if res_t.size >= 3:
            panels.append(("test", lt_t, res_t))
            grids["test"].append((track_name, lt_t, res_t))

        if not panels:
            print(f"  ! Skipping {track_name}: not enough residuals")
            continue

        png, summary = plot_track(track_name, panels, output_dir, file_stem)
        summary_rows.extend(summary)
        splits_done = ", ".join(s for s, _, _ in panels)
        if not val_path.exists():
            splits_done += "  (validation CSV absent — re-run the hybrid for this circuit)"
        print(f"  {track_name}: {splits_done} -> {png.name}")

    for split_key, tracks in grids.items():
        if tracks:
            grid_png = plot_split_grid(split_key, tracks, output_dir)
            print(f"  Combined {split_key} grid -> {grid_png.name}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / "correlation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        pd.set_option("display.float_format", lambda v: f"{v:.4f}")
        print("\n=== Lap time vs residual correlations ===")
        print(summary_df.to_string(index=False))
        print(f"\nSaved correlation summary to: {summary_path}")

    print(f"\nAll outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()
