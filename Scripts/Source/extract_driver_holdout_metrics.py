"""Report per-driver sequential-holdout performance for LR, XGB, or LSTM (hybrid).

Reads the already-saved per-lap holdout prediction CSVs (no model is re-run) and reports
holdout RMSE / MAE / R² (with 95% bootstrap CIs) and the residual STD for one driver, or a
per-driver summary table over all drivers. You pick the model with ``--model`` and the circuit
via the ``CONFIG_PATH`` environment variable (same convention as the other scripts).

Prediction sources (all under ``Scripts/Results/lstm_hybrid/baseline/``):
    lr   -> {safe_gp_name}_lr_ew_holdout_predictions.csv   (pred column: baseline)
    xgb  -> {safe_gp_name}_xgb_ew_holdout_predictions.csv  (pred column: baseline)
    lstm -> {safe_gp_name}_{feature_mode}_{baseline}_hybrid_holdout_predictions.csv (pred column: hybrid_pred)

The LR/XGB CSVs have no Driver column (only row_index), so the driver of each lap is recovered
by joining row_index back to the cleaned dataset (the original RangeIndex), the same source the
training scripts use. The hybrid CSV already carries Driver.

Examples:
    CONFIG_PATH=configs/bahrain.yaml python Scripts/Source/extract_driver_holdout_metrics.py --model lr --driver VER
    CONFIG_PATH=configs/bahrain.yaml python Scripts/Source/extract_driver_holdout_metrics.py --model xgb --driver VER
    CONFIG_PATH=configs/bahrain.yaml python Scripts/Source/extract_driver_holdout_metrics.py --model lstm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_utils import baseline_prediction_paths
from modeling_utils import (
    calc_holdout_ci,
    load_cleaned_data,
    load_config,
    resolve_repo_path,
    safe_gp_name,
)
from model_lstm_baseline import metric_values

# model key -> (baseline_prediction_paths model_kind or None for hybrid, prediction column)
MODEL_SPECS = {
    "lr": ("lr_ew", "baseline"),
    "xgb": ("xgb_ew", "baseline"),
    "lstm": (None, "hybrid_pred"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS), help="Which model's holdout predictions to read.")
    parser.add_argument("--driver", default=None, help="Driver code to report (e.g. VER). Omit for a per-driver table.")
    parser.add_argument("--all", action="store_true", help="Print the full detailed report (RMSE/MAE/R2 with 95% CIs + STD) for every driver individually.")
    parser.add_argument("--year", type=int, default=None, help="Restrict to a single season (e.g. 2024).")
    parser.add_argument("--csv", default=None, help="Path to the holdout predictions CSV (overrides the config-derived path).")
    parser.add_argument("--save", action="store_true", help="Save the per-driver summary table next to the predictions CSV.")
    return parser.parse_args()


def resolve_hybrid_path(repo_root: Path, config: dict) -> Path:
    safe_name = safe_gp_name(str(config["target_gp_name"]))
    feature_mode = str(config.get("hybrid_lstm_feature_mode", "full_embedding")).lower()
    model_kind = str(config.get("hybrid_baseline_model", "lr_ew")).lower()
    subdir = str(config.get("hybrid_baseline_predictions_subdir", "lstm_hybrid/baseline"))
    filename = f"{safe_name}_{feature_mode}_{model_kind}_hybrid_holdout_predictions.csv"
    return resolve_repo_path(repo_root, str(config["results_dir"])) / subdir / filename


def resolve_csv_path(repo_root: Path, config: dict, model: str) -> Path:
    model_kind, _ = MODEL_SPECS[model]
    if model_kind is None:
        return resolve_hybrid_path(repo_root, config)
    _, holdout_path = baseline_prediction_paths(repo_root, config, model_kind)
    return holdout_path


def generating_script(model: str) -> str:
    if model == "lstm":
        return "Scripts/Source/model_lstm_hybrid.py"
    return "Scripts/Source/model_lr_ew.py / model_xgb_ew.py (baseline export)"


def _driver_from_row_index(row_index, cleaned: pd.DataFrame) -> np.ndarray:
    if "Driver" not in cleaned.columns:
        print("ERROR: cleaned dataset has no 'Driver' column to recover from.")
        sys.exit(1)
    idx = np.asarray(row_index)
    missing = set(idx) - set(cleaned.index)
    if missing:
        print(f"ERROR: {len(missing)} row_index values not found in the cleaned dataset (e.g. {sorted(missing)[:5]}).")
        sys.exit(1)
    return cleaned.loc[idx, "Driver"].to_numpy()


def attach_driver(df: pd.DataFrame, script_path: Path, repo_root: Path, config: dict) -> pd.DataFrame:
    """Recover the Driver column when the CSV lacks it.

    Two cases:
      * The CSV has ``row_index`` (LR/XGB baseline CSVs): join it to the cleaned data.
      * The CSV has neither Driver nor row_index (some older hybrid exports, e.g. Saudi):
        align it positionally with the baseline holdout CSV, which does carry row_index, after
        validating that the two blocks are the same length and share y_true.
    """
    _, _, _, cleaned = load_cleaned_data(script_path)
    out = df.copy()
    if "row_index" in df.columns:
        out["Driver"] = _driver_from_row_index(df["row_index"].to_numpy(), cleaned)
        return out

    baseline_kind = str(config.get("hybrid_baseline_model", "lr_ew")).lower()
    _, baseline_holdout = baseline_prediction_paths(repo_root, config, baseline_kind)
    if not baseline_holdout.exists():
        print(
            "ERROR: CSV has neither 'Driver' nor 'row_index', and the baseline holdout CSV needed to "
            f"recover the driver is missing:\n  {baseline_holdout}"
        )
        sys.exit(1)
    base = pd.read_csv(baseline_holdout)
    if len(base) != len(df) or not np.allclose(base["y_true"].to_numpy(float), df["y_true"].to_numpy(float)):
        print(
            "ERROR: cannot recover Driver — this CSV has no Driver/row_index and does not align with the "
            f"baseline holdout block ({baseline_holdout.name}). Re-run model_lstm_hybrid.py to regenerate it."
        )
        sys.exit(1)
    out["Driver"] = _driver_from_row_index(base["row_index"].to_numpy(), cleaned)
    print(f"(recovered Driver by positional alignment with {baseline_holdout.name})")
    return out


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
    script_path = Path(__file__)
    repo_root = script_path.resolve().parents[2]
    config, config_path = load_config(repo_root)
    target_gp_name = str(config["target_gp_name"])
    seed = int(config["random_seed"])

    pred_col = MODEL_SPECS[args.model][1]
    csv_path = Path(args.csv) if args.csv else resolve_csv_path(repo_root, config, args.model)
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path

    print(f"Using config:\n{config_path}")
    print(f"Model: {args.model} (prediction column: {pred_col})")
    print(f"Reading holdout predictions from:\n{csv_path}")
    if not csv_path.exists():
        print(
            "\nERROR: predictions CSV not found. Generate it first with:\n"
            f"  CONFIG_PATH={config_path} python {generating_script(args.model)}"
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if pred_col not in df.columns or "y_true" not in df.columns:
        print(f"ERROR: CSV is missing expected columns ('{pred_col}' and/or 'y_true'). Columns: {list(df.columns)}")
        sys.exit(1)

    if "Driver" not in df.columns:
        df = attach_driver(df, script_path, repo_root, config)

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
        print(f"\n=== Driver {driver} — {args.model} sequential holdout ===")
        report_one(args.model, y, sub[pred_col].to_numpy(dtype=float), seed)
        return

    if args.all:
        # Full detailed report for every driver individually.
        for driver in sorted(df["Driver"].unique()):
            sub = df[df["Driver"] == driver]
            y = sub["y_true"].to_numpy(dtype=float)
            print(f"\n=== Driver {driver} — {args.model} sequential holdout ===")
            report_one(args.model, y, sub[pred_col].to_numpy(dtype=float), seed)
        m_all = metric_values(df["y_true"].to_numpy(dtype=float), df[pred_col].to_numpy(dtype=float))
        print(
            f"\nOVERALL (n={len(df)}) | "
            f"RMSE={m_all['rmse']:.4f} MAE={m_all['mae']:.4f} R2={m_all['r2']:.4f} STD={m_all['std']:.4f}"
        )
        return

    # Per-driver summary table, plus an OVERALL row over all holdout sequences.
    rows = []
    for driver, sub in df.groupby("Driver", sort=True):
        y = sub["y_true"].to_numpy(dtype=float)
        m = metric_values(y, sub[pred_col].to_numpy(dtype=float))
        rows.append({"Driver": driver, "n": len(sub), "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"], "std": m["std"]})
    y_all = df["y_true"].to_numpy(dtype=float)
    m_all = metric_values(y_all, df[pred_col].to_numpy(dtype=float))

    table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\n=== Per-driver {args.model} sequential holdout (sorted by RMSE) ===")
    print(table.to_string(index=False))
    print(
        f"\nOVERALL (n={len(df)}) | "
        f"RMSE={m_all['rmse']:.4f} MAE={m_all['mae']:.4f} R2={m_all['r2']:.4f} STD={m_all['std']:.4f}"
    )

    if args.save:
        out_path = csv_path.with_name(csv_path.stem + f"_{args.model}_per_driver_metrics.csv")
        table.to_csv(out_path, index=False)
        print(f"\nSaved per-driver summary to: {out_path}")


if __name__ == "__main__":
    main()
