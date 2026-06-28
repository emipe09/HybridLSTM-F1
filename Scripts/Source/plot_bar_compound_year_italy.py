"""Grouped bar chart of tyre-compound usage per season for the Italian GP.

X-axis = season (2022..2025); within each season one bar per Pirelli compound (C2..C5),
height = number of laps run on that compound. Article-ready, matching the weather boxplot style
(same font sizes, English labels, no title).

Counts come straight from the cleaned per-lap dataset using the ``pirelliCompound`` column
(the actual compound, also used as a model feature) rather than the generic SOFT/MEDIUM/HARD.

Usage:
    python Scripts/Source/plot_bar_compound_year_italy.py
    python Scripts/Source/plot_bar_compound_year_italy.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

SCRIPT_PATH = Path(__file__)
REPO_ROOT = SCRIPT_PATH.resolve().parents[2]

CLEANED_CSV = REPO_ROOT / "Scripts/ModelData/Italian Grand Prix/italian_grand_prix_cleaned_data.csv"

YEARS = [2022, 2023, 2024, 2025]

# compound -> bar colour (softer = warmer, matching Pirelli's scale intuition)
COMPOUND_COLORS = {
    "C1": "#000000",
    "C2": "#000000",
    "C3": "#b0b0b0",
    "C4": "#f1c40f",
    "C5": "#d62728",
}

FONT_SIZE = 22


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="Display the figure instead of saving it.")
    parser.add_argument("--outdir", default="Scripts/Results/bar_compound_year", help="Directory for the saved PDF.")
    args = parser.parse_args()

    df = pd.read_csv(CLEANED_CSV)
    counts = pd.crosstab(df["Year"], df["pirelliCompound"]).reindex(YEARS, fill_value=0)
    compounds = [c for c in COMPOUND_COLORS if c in counts.columns]

    plt.rcParams.update({"font.size": FONT_SIZE, "font.family": "sans-serif", "axes.linewidth": 2.0})
    fig, ax = plt.subplots(figsize=(11, 7))

    x = np.arange(len(YEARS))
    n = len(compounds)
    width = 0.8 / n
    for i, comp in enumerate(compounds):
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, counts[comp].to_numpy(), width=width, label=comp,
               color=COMPOUND_COLORS[comp], edgecolor="black", linewidth=1.5, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in YEARS])
    ax.set_xlabel("Year", fontsize=FONT_SIZE + 2)
    ax.set_ylabel("Number of laps", fontsize=FONT_SIZE + 2)
    ax.tick_params(axis="both", length=8, width=2, which="major", direction="in", labelsize=FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Compound", fontsize=FONT_SIZE - 4, title_fontsize=FONT_SIZE - 3)

    fig.tight_layout()

    if args.show:
        plt.show()
    else:
        out_dir = REPO_ROOT / args.outdir
        out_dir.mkdir(parents=True, exist_ok=True)
        chart_path = out_dir / "bar_compound_year_italy.pdf"
        fig.savefig(chart_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
