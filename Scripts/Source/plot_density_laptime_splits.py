"""Kernel-density (KDE) curves of lap times in the train / validation / test splits of the hybrid.

Same data and split as ``plot_hist_laptime_splits.py`` (train/val/test reconstructed exactly from
the hybrid's sequential split + saved metadata), but drawn as smooth Gaussian-KDE density curves
instead of histograms. One figure per circuit, three filled curves (train, validation, test).

Result: 5 figures = 5 circuits.

Usage:
    python Scripts/Source/plot_density_laptime_splits.py
    python Scripts/Source/plot_density_laptime_splits.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from plot_hist_laptime_splits import CIRCUITS, SPLIT_STYLE, split_laptimes

SCRIPT_PATH = Path(__file__)
REPO_ROOT = SCRIPT_PATH.resolve().parents[2]


def plot_circuit(circuit_label, splits, x_label, chart_path, show, n_points=400,
                 FONT_SIZE=17, FIG_SIZE_X=8, FIG_SIZE_Y=5):
    fig, ax = plt.subplots(figsize=(FIG_SIZE_X, FIG_SIZE_Y))

    all_vals = np.concatenate([splits[s] for s in SPLIT_STYLE])
    pad = 0.05 * (all_vals.max() - all_vals.min())
    grid = np.linspace(all_vals.min() - pad, all_vals.max() + pad, n_points)

    for split, (label, color) in SPLIT_STYLE.items():
        vals = splits[split]
        kde = gaussian_kde(vals)
        dens = kde(grid)
        ax.plot(grid, dens, color=color, linewidth=2.5, label=f"{label} (n={vals.size})")
        ax.fill_between(grid, dens, color=color, alpha=0.25)

    ax.tick_params(axis="both", length=8, width=2, which="major", direction="in", labelsize=FONT_SIZE)
    ax.set_title(f"{circuit_label} GP", fontsize=FONT_SIZE)
    ax.set_xlabel(x_label, fontsize=FONT_SIZE)
    ax.set_ylabel("Density", fontsize=FONT_SIZE + 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=FONT_SIZE - 3)
    plt.rcParams.update({"font.family": "sans-serif", "axes.linewidth": "2."})
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
    parser.add_argument("--outdir", default="Scripts/Results/density_laptime_splits", help="Directory for the saved PDFs.")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.outdir
    if not args.show:
        out_dir.mkdir(parents=True, exist_ok=True)

    for config_rel, circuit_label in CIRCUITS:
        train, val, test, _target_gp_name = split_laptimes(config_rel)
        splits = {"train": train, "val": val, "test": test}
        print(f"[{circuit_label}] train={train.size} val={val.size} test={test.size}")
        chart_path = str(out_dir / f"density_laptime_{circuit_label.lower()}.pdf")
        plot_circuit(circuit_label, splits, "Lap time (s)", chart_path, args.show)
        if not args.show:
            print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
