"""Slice the LSTM_hybrid sequential-holdout performance by driver.

Reads the per-row hybrid holdout predictions exported by model_lstm_hybrid.py
(``Scripts/Results/lstm_hybrid/baseline/{safe_gp_name}_{feature_mode}_{baseline}_hybrid_holdout_predictions.csv``,
columns: Year, Driver, LapNumber, y_true, hybrid_pred, baseline_pred) and reports holdout
RMSE/MAE/R² (with bootstrap CIs) for one driver, or a per-driver summary table for all
drivers. The hybrid model itself is trained on the full circuit; this only re-slices its
already-computed holdout predictions, so no model is re-run here.

Run the hybrid first so the CSV exists:
    CONFIG_PATH=configs/bahrain.yaml python Scripts/Source/model_lstm_hybrid.py

Then, for one driver:
    CONFIG_PATH=configs/bahrain.yaml python Scripts/Source/extract_driver_hybrid_holdout.py --driver VER

Or a table over all drivers (optionally saved with --save):
    CONFIG_PATH=configs/bahrain.yaml python Scripts/Source/extract_driver_hybrid_holdout.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modeling_utils import (
    calc_holdout_ci,
    load_config,
    resolve_repo_path,
    safe_gp_name,
)
from model_lstm_baseline import metric_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--driver", default=None, help="Driver code to report (e.g. VER). Omit for a per-driver table.")
    parser.add_argument("--year", type=int, default=None, help="Restrict to a single season (e.g. 2024).")
    parser.add_argument("--csv", default=None, help="Path to the hybrid holdout predictions CSV (overrides the config-derived path).")
    parser.add_argument("--save", action="store_true", help="Save the per-driver summary table next to the predictions CSV.")
    return parser.parse_args()


def resolve_predictions_path(repo_root: Path, config: dict) -> Path:
    safe_name = safe_gp_name(str(config["target_gp_name"]))
    feature_mode = str(config.get("hybrid_lstm_feature_mode", "full_embedding")).lower()
    model_kind = str(config.get("hybrid_baseline_model", "lr_ew")).lower()
    subdir = str(config.get("hybrid_baseline_predictions_subdir", "lstm_hybrid/baseline"))
    filename = f"{safe_name}_{feature_mode}_{model_kind}_hybrid_holdout_predictions.csv"
    return resolve_repo_path(repo_root, str(config["results_dir"])) / subdir / filename


def report_one(label: str, y_true: np.ndarray, y_pred: np.ndarray, seed: int) -> dict:
    m = metric_values(y_true, y_pred)
    ci = calc_holdout_ci(y_true, y_pred, seed=seed)
    print(f"\n--- {label} (n={len(y_true)}) ---")
    print(f"RMSE: {m['rmse']:.4f} | 95% CI: [{ci['rmse'][0]:.4f}, {ci['rmse'][1]:.4f}]")
    print(f"MAE:  {m['mae']:.4f} | 95% CI: [{ci['mae'][0]:.4f}, {ci['mae'][1]:.4f}]")
    print(f"R2:   {m['r2']:.4f} | 95% CI: [{ci['r2'][0]:.4f}, {ci['r2'][1]:.4f}]")
    print(f"Residual STD: {m['std']:.4f}")
    return m


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config, config_path = load_config(repo_root)
    target_gp_name = str(config["target_gp_name"])
    seed = int(config["random_seed"])

    csv_path = Path(args.csv) if args.csv else resolve_predictions_path(repo_root, config)
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path
    print(f"Using config:\n{config_path}")
    print(f"Reading hybrid holdout predictions from:\n{csv_path}")
    if not csv_path.exists():
        print(
            "\nERROR: predictions CSV not found. Run the hybrid first so it exports the file:\n"
            f"  CONFIG_PATH={config_path} python Scripts/Source/model_lstm_hybrid.py"
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required = {"Driver", "y_true", "hybrid_pred", "baseline_pred"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing expected columns: {sorted(missing)}. Re-run the hybrid to regenerate it.")
        sys.exit(1)

    if args.year is not None:
        if "Year" not in df.columns:
            print("ERROR: --year given but the CSV has no 'Year' column.")
            sys.exit(1)
        df = df[df["Year"].astype(int) == args.year]
        if df.empty:
            print(f"No holdout rows for year={args.year}.")
            sys.exit(0)

    df["Driver"] = df["Driver"].astype(str).str.upper()
    print(f"\nGrand Prix: {target_gp_name}" + (f" | Year: {args.year}" if args.year is not None else ""))
    print(f"Holdout rows available: {len(df)} | Drivers: {', '.join(sorted(df['Driver'].unique()))}")

    if args.driver:
        driver = args.driver.strip().upper()
        sub = df[df["Driver"] == driver]
        if sub.empty:
            print(
                f"\nSKIP: driver={driver!r} has no holdout rows in this CSV. "
                f"Available: {', '.join(sorted(df['Driver'].unique()))}."
            )
            sys.exit(0)
        y = sub["y_true"].to_numpy(dtype=float)
        print(f"\n=== Driver {driver} — LSTM_hybrid sequential holdout ===")
        report_one("Hybrid", y, sub["hybrid_pred"].to_numpy(dtype=float), seed)
        report_one("Tabular baseline (reference)", y, sub["baseline_pred"].to_numpy(dtype=float), seed)
        return

    # Per-driver summary table (hybrid), plus an OVERALL row over all holdout sequences.
    rows = []
    for driver, sub in df.groupby("Driver", sort=True):
        y = sub["y_true"].to_numpy(dtype=float)
        mh = metric_values(y, sub["hybrid_pred"].to_numpy(dtype=float))
        mb = metric_values(y, sub["baseline_pred"].to_numpy(dtype=float))
        rows.append(
            {
                "Driver": driver,
                "n": len(sub),
                "hybrid_rmse": mh["rmse"],
                "hybrid_mae": mh["mae"],
                "hybrid_r2": mh["r2"],
                "baseline_rmse": mb["rmse"],
                "baseline_mae": mb["mae"],
                "baseline_r2": mb["r2"],
            }
        )
    y_all = df["y_true"].to_numpy(dtype=float)
    mh_all = metric_values(y_all, df["hybrid_pred"].to_numpy(dtype=float))
    mb_all = metric_values(y_all, df["baseline_pred"].to_numpy(dtype=float))

    table = pd.DataFrame(rows).sort_values("hybrid_rmse").reset_index(drop=True)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== Per-driver LSTM_hybrid sequential holdout (sorted by hybrid RMSE) ===")
    print(table.to_string(index=False))
    print(
        f"\nOVERALL (n={len(df)}) | "
        f"hybrid RMSE={mh_all['rmse']:.4f} MAE={mh_all['mae']:.4f} R2={mh_all['r2']:.4f} | "
        f"baseline RMSE={mb_all['rmse']:.4f} MAE={mb_all['mae']:.4f} R2={mb_all['r2']:.4f}"
    )

    if args.save:
        out_path = csv_path.with_name(csv_path.stem + "_per_driver_summary.csv")
        table.to_csv(out_path, index=False)
        print(f"\nSaved per-driver summary to: {out_path}")


if __name__ == "__main__":
    main()
