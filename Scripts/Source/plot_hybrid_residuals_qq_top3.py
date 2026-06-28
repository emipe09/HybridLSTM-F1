"""Normal Q-Q plots of the hybrid-model residuals for Italy's top-3 best-performing drivers.

Ranks the Italian GP drivers by their per-driver holdout performance under the LSTM hybrid model
(best = lowest RMSE by default) and draws a normal quantile-quantile plot of the residuals
(y_true - hybrid_pred) for the top 3. This lets you check the normality of the residuals for the
drivers the hybrid models best.

Predictions are read from the already-saved hybrid holdout CSV (no model is re-run), reusing the
driver-recovery logic from ``extract_driver_holdout_metrics.py`` and the Q-Q helpers from
``plot_lr_residuals_qq.py``.

Outputs (under ``Scripts/Results/hybrid_residual_qq``): one PNG/PDF per driver plus a combined
1x3 grid.

Usage:
    python Scripts/Source/plot_hybrid_residuals_qq_top3.py
    python Scripts/Source/plot_hybrid_residuals_qq_top3.py --metric mae   # rank by MAE instead
    python Scripts/Source/plot_hybrid_residuals_qq_top3.py --top 5        # top 5 drivers
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from modeling_utils import load_config
from extract_driver_holdout_metrics import MODEL_SPECS, attach_driver, resolve_csv_path
from model_lstm_baseline import metric_values
from plot_lr_residuals_qq import qq_points, _draw_qq

SCRIPT_PATH = Path(__file__)
REPO_ROOT = SCRIPT_PATH.resolve().parents[2]

CONFIG_REL = "configs/italy.yaml"
CIRCUIT_LABEL = "Italy"
MODEL = "lstm"  # the hybrid model
OUTPUT_SUBDIR = "hybrid_residual_qq"

# ranking metric -> "lower is better"?
LOWER_IS_BETTER = {"rmse": True, "mae": True, "r2": False}


def load_hybrid_with_driver() -> pd.DataFrame:
    """Return the Italy hybrid holdout predictions with y_true, prediction and Driver columns."""
    os.environ["CONFIG_PATH"] = CONFIG_REL
    config, _ = load_config(REPO_ROOT)
    pred_col = MODEL_SPECS[MODEL][1]

    csv_path = resolve_csv_path(REPO_ROOT, config, MODEL)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"[{CONFIG_REL} / {MODEL}] holdout predictions not found:\n  {csv_path}")

    df = pd.read_csv(csv_path)
    if pred_col not in df.columns or "y_true" not in df.columns:
        raise ValueError(f"CSV missing '{pred_col}'/'y_true'. Columns: {list(df.columns)}")
    if "Driver" not in df.columns:
        df = attach_driver(df, SCRIPT_PATH, REPO_ROOT, config)
    df["Driver"] = df["Driver"].astype(str).str.upper()
    df = df.rename(columns={pred_col: "y_pred"})
    return df[["Driver", "y_true", "y_pred"]].copy()


def rank_drivers(df: pd.DataFrame, metric: str, min_laps: int) -> pd.DataFrame:
    rows = []
    for driver, sub in df.groupby("Driver", sort=True):
        y = sub["y_true"].to_numpy(dtype=float)
        if len(sub) < min_laps or np.allclose(y, y[0]):
            continue
        m = metric_values(y, sub["y_pred"].to_numpy(dtype=float))
        rows.append({"Driver": driver, "n": len(sub), **{k: m[k] for k in ("mae", "rmse", "r2")}})
    table = pd.DataFrame(rows)
    return table.sort_values(metric, ascending=LOWER_IS_BETTER[metric]).reset_index(drop=True)


def driver_residuals(df: pd.DataFrame, driver: str) -> np.ndarray:
    sub = df[df["Driver"] == driver]
    resid = sub["y_true"].to_numpy(dtype=float) - sub["y_pred"].to_numpy(dtype=float)
    return resid[np.isfinite(resid)]


def shapiro_p(residuals: np.ndarray) -> float:
    return float(stats.shapiro(residuals).pvalue) if 3 <= residuals.size <= 5000 else float("nan")


def plot_single(driver: str, residuals: np.ndarray, output_dir: Path) -> Path:
    theoretical, standardised, _s, _i = qq_points(residuals)
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_qq(ax, theoretical, standardised, f"{CIRCUIT_LABEL} GP — {driver} (hybrid)",
             residuals.size, shapiro_p(residuals))
    fig.tight_layout()
    pdf_path = output_dir / f"qq_{CIRCUIT_LABEL.lower()}_hybrid_{driver.lower()}.pdf"
    fig.savefig(output_dir / f"qq_{CIRCUIT_LABEL.lower()}_hybrid_{driver.lower()}.png", dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def plot_grid(drivers_resid: list[tuple[str, np.ndarray]], metric: str, output_dir: Path) -> Path:
    n = len(drivers_resid)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), squeeze=False)
    for idx, (driver, residuals) in enumerate(drivers_resid):
        theoretical, standardised, _s, _i = qq_points(residuals)
        _draw_qq(axes[0][idx], theoretical, standardised, f"{driver} (hybrid)",
                 residuals.size, shapiro_p(residuals))
    fig.suptitle(f"Normal Q-Q — {CIRCUIT_LABEL} GP top-{n} drivers by hybrid {metric.upper()}", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf_path = output_dir / f"qq_{CIRCUIT_LABEL.lower()}_hybrid_top{n}.pdf"
    fig.savefig(output_dir / f"qq_{CIRCUIT_LABEL.lower()}_hybrid_top{n}.png", dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", choices=sorted(LOWER_IS_BETTER), default="rmse",
                        help="Per-driver metric used to rank 'best performance' (default rmse).")
    parser.add_argument("--top", type=int, default=3, help="How many top drivers to plot (default 3).")
    parser.add_argument("--min-laps", type=int, default=2, help="Minimum holdout laps a driver needs (default 2).")
    parser.add_argument("--outdir", default=f"Scripts/Results/{OUTPUT_SUBDIR}", help="Directory for the outputs.")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.outdir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_hybrid_with_driver()
    table = rank_drivers(df, args.metric, args.min_laps)
    top = table.head(args.top)
    print(f"\nTop-{args.top} {CIRCUIT_LABEL} drivers by hybrid {args.metric} (best first):")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    drivers_resid = []
    for driver in top["Driver"]:
        resid = driver_residuals(df, driver)
        if resid.size < 3:
            print(f"  ! Skipping {driver}: only {resid.size} residuals")
            continue
        pdf = plot_single(driver, resid, output_dir)
        drivers_resid.append((driver, resid))
        print(f"  {driver}: n={resid.size} -> {pdf.name}")

    if drivers_resid:
        grid_pdf = plot_grid(drivers_resid, args.metric, output_dir)
        print(f"\nCombined grid -> {grid_pdf.name}")
    print(f"\nAll outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()
