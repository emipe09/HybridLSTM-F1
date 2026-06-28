"""Plot CDF curves of per-driver holdout metrics (MAE, RMSE, R2) for Italy and USA.

For each circuit (Italian GP, United States GP) this builds, per metric, one CDF figure whose
three curves are the three models (LR baseline, XGBoost baseline, LSTM hybrid). Each curve is the
empirical CDF over the *per-driver* metric values, i.e. one point per driver that has holdout laps.

Result: 6 figures = 3 metrics (MAE, RMSE, R2) x 2 circuits.

The per-driver metrics are computed on the fly from the already-saved holdout prediction CSVs (no
model is re-run), reusing the same prediction sources and driver-recovery logic as
``extract_driver_holdout_metrics.py``.

Usage:
    python Scripts/Source/plot_cdf_driver_metrics.py
    python Scripts/Source/plot_cdf_driver_metrics.py --show          # display instead of saving
    python Scripts/Source/plot_cdf_driver_metrics.py --min-laps 3    # require >=3 holdout laps/driver
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from statsmodels.distributions.empirical_distribution import ECDF

from modeling_utils import load_config
from extract_driver_holdout_metrics import (
    MODEL_SPECS,
    attach_driver,
    resolve_csv_path,
)
from model_lstm_baseline import metric_values

SCRIPT_PATH = Path(__file__)
REPO_ROOT = SCRIPT_PATH.resolve().parents[2]

# (config file, short circuit label used in filenames/titles)
CIRCUITS = [
    ("configs/italy.yaml", "Italy"),
    ("configs/usa.yaml", "USA"),
    ("configs/saudi.yaml", "Saudi"),
]

# model key -> human label for the legend (drawn in the order listed)
MODEL_LABELS = {
    "lr": "LR",
    "xgb": "XGBoost",
    "lstm": "LSTM Hybrid",
}

# metric key -> (x-axis label, "higher is better"?)
METRICS = {
    "mae": "Per-driver MAE (s)",
    "rmse": "Per-driver RMSE (s)",
    "r2": r"Per-driver $R^2$",
}


def plot_cdf_list_curves(list_curve, list_label, x_label, chart_path='',
                         SET_LOG=True, CCDF=False, LEG_LOC='best', XLIM=False, xlimits=[], YLIM=False, ylimits=[],
                         OUT_LEG=False, SAVE_FIG=False, SET_TITLE=False, title_name='',
                         FONT_SIZE_LEG=17, FONT_SIZE=17, FIG_SIZE_X=7, FIG_SIZE_Y=5, BOUND_CORR=False,
                         SET_MARKER=False, SET_GRID=False, SET_LEG=True):
    fig, ax = plt.subplots(figsize=(FIG_SIZE_X, FIG_SIZE_Y))

    size = len(list_curve)
    style = ['-', '-.', '--', ':', '-', '--', '-.', ':', '-.', '--', '-', '-.']
    color = ['blue', 'black', 'red', 'green', 'purple', 'brown', 'magenta', 'pink', 'goldenrod', 'cyan', 'grey', 'yellow']
    linewidth = [2.5, 2., 2.5, 2.8, 3.8, 3.5, 3., 2.5, 2.5, 3., 3.5, 3.8]
    minorLocatory = AutoMinorLocator(2)
    minorLocatorx = AutoMinorLocator(5)

    for i in range(size):
        list_curve[i].sort()
        ecdf = ECDF(list_curve[i])
        if SET_LOG:
            ax.set_xscale('log')
        ax.tick_params(axis='both', length=8, width=2, which='major', bottom=True, top=True, left=True,
                       right=True, direction='in', labelsize=FONT_SIZE)
        ax.tick_params(axis='both', length=6, width=1, which='minor', bottom=True, top=True, left=True,
                       right=True, direction='in', labelsize=FONT_SIZE)
        ax.yaxis.set_minor_locator(minorLocatory)
        markers_on = []
        for idx in range(34):
            idx_value = int(np.ceil(idx * 0.03 * len(list_curve[i])))
            markers_on.append(idx_value)
        markers_on.append(len(list_curve[i]) - 1)
        if SET_GRID:
            plt.grid(True)
        if not SET_LOG:
            ax.xaxis.set_minor_locator(minorLocatorx)
        if XLIM:
            ax.set_xlim(xlimits)
        if YLIM:
            ax.set_ylim(ylimits)
        if CCDF and BOUND_CORR is not True:
            if not SET_MARKER:
                plt.plot(ecdf.x, 1 - ecdf.y, label=list_label[i], lw=linewidth[i], ls=style[i], c=color[i])
            else:
                plt.plot(ecdf.x, 1 - ecdf.y, label=list_label[i], c=color[i], ms=4, lw=linewidth[i], ls=style[i])
        else:
            if not SET_MARKER:
                plt.plot(ecdf.x, ecdf.y, label=list_label[i], lw=linewidth[i], ls=style[i], c=color[i])
            else:
                plt.plot(ecdf.x, ecdf.y, label=list_label[i], c=color[i], ms=4, lw=linewidth[i], ls=style[i])
        if BOUND_CORR is True and size == 3:
            plt.fill([-0.5, -0.5, 0.5, 0.5], [0, 1, 1, 0], color='red', alpha=0.005)

    plt.rcParams.update({'font.size': FONT_SIZE_LEG, 'font.family': 'sans-serif', 'axes.linewidth': '2.'})
    if SET_TITLE:
        ax.set_title(title_name)
    ax.set_xlabel(x_label, fontsize=FONT_SIZE)
    if CCDF:
        if LEG_LOC == '' and OUT_LEG is True:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            plt.legend(loc=LEG_LOC)
        ax.set_ylabel('CCDF', fontsize=FONT_SIZE + 2)
    else:
        if LEG_LOC == '' and OUT_LEG is True:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            plt.legend(loc=LEG_LOC)
        ax.set_ylabel('CDF', fontsize=FONT_SIZE + 2)
    if not SET_LEG:
        ax.legend().set_visible(False)
    plt.tight_layout()
    if SAVE_FIG:
        plt.draw()
        fig.savefig(chart_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()
    else:
        plt.show()


def per_driver_metrics(config_rel: str, model: str, min_laps: int) -> pd.DataFrame:
    """Return a DataFrame with one row per driver (Driver, n, mae, rmse, r2, std) for one model."""
    os.environ["CONFIG_PATH"] = config_rel
    config, _ = load_config(REPO_ROOT)
    pred_col = MODEL_SPECS[model][1]

    csv_path = resolve_csv_path(REPO_ROOT, config, model)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"[{config_rel} / {model}] holdout predictions not found:\n  {csv_path}")

    df = pd.read_csv(csv_path)
    if pred_col not in df.columns or "y_true" not in df.columns:
        raise ValueError(f"[{config_rel} / {model}] CSV missing '{pred_col}'/'y_true'. Columns: {list(df.columns)}")
    if "Driver" not in df.columns:
        df = attach_driver(df, SCRIPT_PATH, REPO_ROOT, config)
    df["Driver"] = df["Driver"].astype(str).str.upper()

    rows = []
    for driver, sub in df.groupby("Driver", sort=True):
        y = sub["y_true"].to_numpy(dtype=float)
        if len(sub) < min_laps or np.allclose(y, y[0]):
            # too few laps (R2 ill-defined) or no variance in the target
            continue
        m = metric_values(y, sub[pred_col].to_numpy(dtype=float))
        rows.append({"Driver": driver, "n": len(sub), **{k: m[k] for k in ("mae", "rmse", "r2", "std")}})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="Display the figures instead of saving them.")
    parser.add_argument("--min-laps", type=int, default=2, help="Minimum holdout laps a driver needs to be included (default 2).")
    parser.add_argument("--outdir", default="Scripts/Results/cdf_driver_metrics", help="Directory for the saved PDFs.")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.outdir
    if not args.show:
        out_dir.mkdir(parents=True, exist_ok=True)

    # tables[(circuit_label, model)] = per-driver DataFrame
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    for config_rel, circuit_label in CIRCUITS:
        for model in MODEL_LABELS:
            tbl = per_driver_metrics(config_rel, model, args.min_laps)
            tables[(circuit_label, model)] = tbl
            print(f"[{circuit_label} / {model}] drivers={len(tbl)}")

    for _, circuit_label in CIRCUITS:
        for metric, x_label in METRICS.items():
            list_curve, list_label = [], []
            for model, model_label in MODEL_LABELS.items():
                tbl = tables[(circuit_label, model)]
                vals = tbl[metric].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    print(f"WARN: no {metric} values for {circuit_label}/{model}; skipping that curve.")
                    continue
                list_curve.append(list(vals))
                list_label.append(f"{model_label} (n={vals.size})")

            if not list_curve:
                print(f"WARN: no curves for {circuit_label} {metric}; skipping figure.")
                continue

            title = f"{circuit_label} GP — per-driver {metric.upper()}"
            chart_path = str(out_dir / f"cdf_{circuit_label.lower()}_{metric}.pdf")
            plot_cdf_list_curves(
                list_curve, list_label, x_label,
                chart_path=chart_path,
                SET_LOG=False, SET_GRID=True, LEG_LOC="best",
                SET_TITLE=True, title_name=title,
                SAVE_FIG=not args.show,
            )
            if not args.show:
                print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
