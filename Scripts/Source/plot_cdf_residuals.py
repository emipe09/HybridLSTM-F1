"""Plot CDF curves of the holdout residuals (y_true - y_pred) per model, for Italy and USA.

For each circuit (Italian GP, United States GP) this builds one CDF figure whose three curves are
the three models (LR baseline, XGBoost baseline, LSTM hybrid). Each curve is the empirical CDF over
the per-lap residuals (real minus predicted) on the holdout set.

Result: 2 figures = 2 circuits.

Residuals are read from the already-saved holdout prediction CSVs (no model is re-run), reusing the
same prediction sources as ``extract_driver_holdout_metrics.py``.

Usage:
    python Scripts/Source/plot_cdf_residuals.py
    python Scripts/Source/plot_cdf_residuals.py --show     # display instead of saving
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from modeling_utils import load_config
from extract_driver_holdout_metrics import MODEL_SPECS, resolve_csv_path
from plot_cdf_driver_metrics import plot_cdf_list_curves

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


def model_residuals(config_rel: str, model: str) -> np.ndarray:
    """Return the per-lap holdout residuals (y_true - y_pred) for one circuit/model."""
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

    resid = df["y_true"].to_numpy(dtype=float) - df[pred_col].to_numpy(dtype=float)
    return resid[np.isfinite(resid)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="Display the figures instead of saving them.")
    parser.add_argument("--outdir", default="Scripts/Results/cdf_residuals", help="Directory for the saved PDFs.")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.outdir
    if not args.show:
        out_dir.mkdir(parents=True, exist_ok=True)

    for config_rel, circuit_label in CIRCUITS:
        list_curve, list_label = [], []
        for model, model_label in MODEL_LABELS.items():
            resid = model_residuals(config_rel, model)
            print(f"[{circuit_label} / {model}] residuals n={resid.size}")
            if resid.size == 0:
                print(f"WARN: no residuals for {circuit_label}/{model}; skipping that curve.")
                continue
            list_curve.append(list(resid))
            list_label.append(f"{model_label} (n={resid.size})")

        if not list_curve:
            print(f"WARN: no curves for {circuit_label}; skipping figure.")
            continue

        title = f"{circuit_label} GP — holdout residuals"
        chart_path = str(out_dir / f"cdf_residuals_{circuit_label.lower()}.pdf")
        plot_cdf_list_curves(
            list_curve, list_label, "Residual: real - predicted (s)",
            chart_path=chart_path,
            SET_LOG=False, SET_GRID=True, LEG_LOC="best",
            SET_TITLE=True, title_name=title,
            SAVE_FIG=not args.show,
        )
        if not args.show:
            print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
