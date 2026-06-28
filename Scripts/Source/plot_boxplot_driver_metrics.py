"""Boxplots of per-driver holdout metrics (MAE, RMSE, R2) per model, one figure per circuit.

For each circuit and each metric this builds one figure with three boxes — LR baseline, XGBoost
baseline, LSTM hybrid — where each box is the distribution of the *per-driver* metric values on that
circuit (one value per driver).

Result: 15 figures = 5 circuits x 3 metrics (MAE, RMSE, R2).

The per-driver metrics are computed on the fly from the already-saved holdout prediction CSVs (no
model is re-run), reusing ``per_driver_metrics`` from ``plot_cdf_driver_metrics.py``.

Usage:
    python Scripts/Source/plot_boxplot_driver_metrics.py
    python Scripts/Source/plot_boxplot_driver_metrics.py --show     # display instead of saving
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from plot_cdf_driver_metrics import per_driver_metrics

SCRIPT_PATH = Path(__file__)
REPO_ROOT = SCRIPT_PATH.resolve().parents[2]

# all five circuits (config file, short label)
CIRCUITS = [
    ("configs/bahrain.yaml", "Bahrain"),
    ("configs/saudi.yaml", "Saudi"),
    ("configs/italy.yaml", "Italy"),
    ("configs/usa.yaml", "USA"),
    ("configs/hungary.yaml", "Hungary"),
]

# model key -> human label (box order, left to right)
MODEL_LABELS = {
    "lr": "LR",
    "xgb": "XGBoost",
    "lstm": "LSTM Hybrid",
}

# metric key -> y-axis label
METRICS = {
    "mae": "Per-driver MAE (s)",
    "rmse": "Per-driver RMSE (s)",
    "r2": r"Per-driver $R^2$",
}

# box fill colours, matching the CDF curve colours (LR=blue, XGB=black, LSTM=red)
BOX_COLORS = {"lr": "blue", "xgb": "black", "lstm": "red"}


def plot_metric_boxplot(metric, y_label, data_by_model, chart_path, show, title_name="",
                        FONT_SIZE=17, FIG_SIZE_X=7, FIG_SIZE_Y=5):
    fig, ax = plt.subplots(figsize=(FIG_SIZE_X, FIG_SIZE_Y))

    models = list(MODEL_LABELS)
    data = [data_by_model[m] for m in models]
    labels = [f"{MODEL_LABELS[m]}\n(n={len(data_by_model[m])})" for m in models]

    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True, widths=0.55,
                    medianprops=dict(color="goldenrod", linewidth=2.5),
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=7),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor="none", alpha=0.6))
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(BOX_COLORS[m])
        patch.set_alpha(0.25)
        patch.set_edgecolor(BOX_COLORS[m])
        patch.set_linewidth(2.0)
    for whisker, cap in zip(bp["whiskers"], bp["caps"]):
        whisker.set_linewidth(1.8)
        cap.set_linewidth(1.8)

    ax.tick_params(axis="both", length=8, width=2, which="major", direction="in", labelsize=FONT_SIZE)
    if title_name:
        ax.set_title(title_name, fontsize=FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE + 2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.rcParams.update({"font.size": FONT_SIZE, "font.family": "sans-serif", "axes.linewidth": "2."})
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.draw()
        fig.savefig(chart_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.clf()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="Display the figures instead of saving them.")
    parser.add_argument("--min-laps", type=int, default=2, help="Minimum holdout laps a driver needs to be included (default 2).")
    parser.add_argument("--outdir", default="Scripts/Results/boxplot_driver_metrics", help="Directory for the saved PDFs.")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.outdir
    if not args.show:
        out_dir.mkdir(parents=True, exist_ok=True)

    for config_rel, circuit_label in CIRCUITS:
        # data_by_model[metric][model] = list of per-driver metric values on this circuit
        data = {metric: {m: [] for m in MODEL_LABELS} for metric in METRICS}
        for model in MODEL_LABELS:
            tbl = per_driver_metrics(config_rel, model, args.min_laps)
            for metric in METRICS:
                vals = tbl[metric].to_numpy(dtype=float)
                data[metric][model] = vals[np.isfinite(vals)].tolist()
            print(f"[{circuit_label} / {model}] drivers={len(tbl)}")

        for metric, y_label in METRICS.items():
            chart_path = str(out_dir / f"boxplot_{circuit_label.lower()}_{metric}.pdf")
            title = f"{circuit_label} GP"
            plot_metric_boxplot(metric, y_label, data[metric], chart_path, args.show, title_name=title)
            if not args.show:
                print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
