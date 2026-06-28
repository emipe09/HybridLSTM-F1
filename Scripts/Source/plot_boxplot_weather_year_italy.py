"""Boxplots of the five (original, pre-RBF) weather features plus lap time for Italy, by year.

One figure, 2 rows x 3 columns = 6 panels. Each panel shows one variable, with four boxes
side by side, one per season (2022, 2023, 2024, 2025). Article-ready: large English labels,
no per-panel titles.

The values are taken straight from the cleaned per-lap dataset, using the *original* weather
columns (AirTemp/Humidity/Pressure/TrackTemp/WindSpeed/TempDelta) rather than their RBF-median
transforms used for modelling.

Usage:
    python Scripts/Source/plot_boxplot_weather_year_italy.py
    python Scripts/Source/plot_boxplot_weather_year_italy.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

SCRIPT_PATH = Path(__file__)
REPO_ROOT = SCRIPT_PATH.resolve().parents[2]

CLEANED_CSV = REPO_ROOT / "Scripts/ModelData/Italian Grand Prix/italian_grand_prix_cleaned_data.csv"

YEARS = [2022, 2023, 2024, 2025]

# (column, panel y-axis label) — five original weather features + lap time, in panel order
PANELS = [
    ("TrackTemp", "Track Temperature (°C)"),
    ("Humidity", "Humidity (%)"),
    ("Pressure", "Pressure (hPa)"),
    ("WindSpeed", "Wind Speed (m/s)"),
    ("TempDelta", "Temp. Delta (°C)"),
    ("LapTime_seconds", "Lap Time (s)"),
]

FONT_SIZE = 26
BOX_COLOR = "#1f6fb4"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="Display the figure instead of saving it.")
    parser.add_argument("--outdir", default="Scripts/Results/boxplot_weather_year", help="Directory for the saved PDF.")
    args = parser.parse_args()

    df = pd.read_csv(CLEANED_CSV)

    plt.rcParams.update({"font.size": FONT_SIZE, "font.family": "sans-serif", "axes.linewidth": 2.0})
    fig, axes = plt.subplots(1, 6, figsize=(36, 6), layout="constrained", sharey=False, squeeze=False)

    for ax, (col, y_label) in zip(axes.flat, PANELS):
        data = [df.loc[df["Year"] == y, col].dropna().to_numpy(dtype=float) for y in YEARS]
        bp = ax.boxplot(data, tick_labels=[str(y) for y in YEARS], showmeans=True,
                        patch_artist=True, widths=0.6,
                        medianprops=dict(color="goldenrod", linewidth=2.5),
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black", markersize=8),
                        flierprops=dict(marker="o", markersize=4, markerfacecolor="none", alpha=0.6))
        for patch in bp["boxes"]:
            patch.set_facecolor(BOX_COLOR)
            patch.set_alpha(0.3)
            patch.set_edgecolor(BOX_COLOR)
            patch.set_linewidth(3.0)
        for whisker, cap in zip(bp["whiskers"], bp["caps"]):
            whisker.set_linewidth(1.8)
            cap.set_linewidth(2.5)

        ax.set_ylabel(y_label, fontsize=FONT_SIZE + 2)
        ax.set_xlabel("Year", fontsize=FONT_SIZE + 2)
        ax.tick_params(axis="both", length=8, width=2, which="major", direction="in", labelsize=FONT_SIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.get_layout_engine().set(wspace=0.02)

    if args.show:
        plt.show()
    else:
        out_dir = REPO_ROOT / args.outdir
        out_dir.mkdir(parents=True, exist_ok=True)
        chart_path = out_dir / "boxplot_weather_year_italy.pdf"
        fig.savefig(chart_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
