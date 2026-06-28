"""Histograms of lap times in the train / validation / test splits of the LSTM hybrid.

For each circuit this builds one figure with three overlaid histograms — train, validation and test
(holdout) — of the lap-time target, using exactly the same sequential split the hybrid model uses:

  * test (holdout): the most recent unique (Year, LapNumber) steps (``holdout_ratio``), via
    ``build_sequential_split``;
  * within the modeling block, the first ``train_laps`` unique steps are train and the rest are
    validation. ``train_laps`` is read from the saved hybrid metadata so the boundary matches the
    trained model exactly.

Histograms are normalised to densities because the three splits have very different sizes.

Result: 5 figures = 5 circuits.

Usage:
    python Scripts/Source/plot_hist_laptime_splits.py
    python Scripts/Source/plot_hist_laptime_splits.py --show
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from modeling_utils import build_sequential_split, load_cleaned_data, resolve_repo_path, safe_gp_name

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

# split -> (legend label, colour)
SPLIT_STYLE = {
    "train": ("Train", "blue"),
    "val": ("Validation", "goldenrod"),
    "test": ("Test (holdout)", "red"),
}


def hybrid_metadata_path(config: dict) -> Path:
    safe_name = safe_gp_name(str(config["target_gp_name"]))
    feature_mode = str(config.get("hybrid_lstm_feature_mode", "full_embedding")).lower()
    model_kind = str(config.get("hybrid_baseline_model", "lr_ew")).lower()
    filename = f"{safe_name}_{feature_mode}_{model_kind}_lstm_hybrid_model_metadata.json"
    return resolve_repo_path(REPO_ROOT, str(config["results_dir"])) / "lstm_hybrid" / "models" / filename


def split_laptimes(config_rel: str):
    """Return (train, val, test) arrays of lap times for one circuit, matching the hybrid split."""
    os.environ["CONFIG_PATH"] = config_rel
    target_gp_name, config, _repo_root, df_base = load_cleaned_data(SCRIPT_PATH)

    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    y = pd.to_numeric(df_base[target_col], errors="coerce")
    valid_indices = y.dropna().index

    step_series, *_, model_idx, holdout_idx, _hs, _me, _tot = build_sequential_split(
        df_base, valid_indices, float(config["holdout_ratio"]), lap_col
    )

    # number of train steps from the trained model's metadata (exact boundary)
    meta_path = hybrid_metadata_path(config)
    if not meta_path.exists():
        raise FileNotFoundError(f"Hybrid metadata not found (needed for the train/val boundary):\n  {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_train_steps = int(meta["train_laps"])

    model_steps = step_series.loc[model_idx]
    unique_steps = np.sort(pd.to_numeric(model_steps, errors="coerce").dropna().unique())
    train_steps = set(unique_steps[:n_train_steps].tolist())

    is_train = model_steps.isin(train_steps)
    train_idx = model_steps[is_train].index
    val_idx = model_steps[~is_train].index

    return (
        y.loc[train_idx].to_numpy(dtype=float),
        y.loc[val_idx].to_numpy(dtype=float),
        y.loc[holdout_idx].to_numpy(dtype=float),
        target_gp_name,
    )


def plot_circuit(circuit_label, splits, target_col_label, chart_path, show,
                 bins=30, FONT_SIZE=17, FIG_SIZE_X=8, FIG_SIZE_Y=5):
    fig, ax = plt.subplots(figsize=(FIG_SIZE_X, FIG_SIZE_Y))

    all_vals = np.concatenate([splits[s] for s in SPLIT_STYLE])
    edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    for split, (label, color) in SPLIT_STYLE.items():
        vals = splits[split]
        ax.hist(vals, bins=edges, density=True, color=color, alpha=0.35,
                label=f"{label} (n={vals.size})")
        ax.hist(vals, bins=edges, density=True, histtype="step", color=color, linewidth=2.0)

    ax.tick_params(axis="both", length=8, width=2, which="major", direction="in", labelsize=FONT_SIZE)
    ax.set_title(f"{circuit_label} GP", fontsize=FONT_SIZE)
    ax.set_xlabel(target_col_label, fontsize=FONT_SIZE)
    ax.set_ylabel("Density", fontsize=FONT_SIZE + 1)
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
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins (default 30).")
    parser.add_argument("--outdir", default="Scripts/Results/hist_laptime_splits", help="Directory for the saved PDFs.")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.outdir
    if not args.show:
        out_dir.mkdir(parents=True, exist_ok=True)

    for config_rel, circuit_label in CIRCUITS:
        train, val, test, target_gp_name = split_laptimes(config_rel)
        splits = {"train": train, "val": val, "test": test}
        print(f"[{circuit_label}] train={train.size} val={val.size} test={test.size}")
        chart_path = str(out_dir / f"hist_laptime_{circuit_label.lower()}.pdf")
        plot_circuit(circuit_label, splits, "Lap time (s)", chart_path, args.show, bins=args.bins)
        if not args.show:
            print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
