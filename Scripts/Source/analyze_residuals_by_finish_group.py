"""Holdout-residual analysis grouped by 2025 race-finish bands, per circuit and model.

For the three reference models (LR-EW, XGBoost-EW and the LSTM Hybrid) this script reads
the per-row *holdout* predictions exported under ``Scripts/Results/lstm_hybrid/baseline``,
computes the residual ``residual = y_true - prediction`` for every holdout lap (the holdout
block is the 2025 race), attaches each lap to its ``Driver`` and to that driver's 2025
finishing position, and then summarises the residuals by finishing band:

    Group 1 -> P1-P5     Group 2 -> P6-P10
    Group 3 -> P11-P15   Group 4 -> P16-P20

For every (circuit, model, group) it reports the residual mean (signed bias) and the
residual standard deviation (dispersion), pooling all holdout laps of the drivers in the
band. Results are written as a tidy CSV plus a per-model wide table, and a grouped bar
chart (mean +/- std) per model.

Prediction sources (per circuit ``{safe_gp_name}_*``):
  - LR-EW   -> ``*_lr_ew_holdout_predictions.csv``   (row_index, Year, LapNumber, y_true, baseline)
  - XGB-EW  -> ``*_xgb_ew_holdout_predictions.csv``  (idem)
  - Hybrid  -> ``*_hybrid_holdout_predictions.csv``  (Year, Driver, LapNumber, y_true, hybrid_pred, ...)

The LR/XGB exports do not carry ``Driver``; it is recovered from the circuit's cleaned
dataset via ``row_index``. The hybrid export already carries ``Driver``.

Finishing position (2025): for each driver the ``Position`` value on their last completed
lap. Drivers who finished (non-NaN position) are ranked by that position; drivers who
retired (NaN position) are classified last, ordered by laps completed (more laps first).

Usage:
    python Scripts/Source/analyze_residuals_by_finish_group.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HOLDOUT_YEAR = 2025

BASELINE_DIR = Path("Scripts/Results/lstm_hybrid/baseline")
MODELDATA_DIR = Path("Scripts/ModelData")
OUTPUT_DIR = Path("Scripts/Results/residuals_by_finish_group")

# model key -> (holdout filename glob (relative to a circuit safe name), prediction column,
# needs driver lookup). The hybrid file embeds the feature mode and baseline tag between the
# circuit name and the suffix (e.g. ``..._full_embedding_lr_ew_hybrid_holdout_predictions.csv``),
# so it is matched with a wildcard.
MODELS = {
    "lr_ew": ("{safe}_lr_ew_holdout_predictions.csv", "baseline", True),
    "xgb_ew": ("{safe}_xgb_ew_holdout_predictions.csv", "baseline", True),
    "hybrid": ("{safe}_*_hybrid_holdout_predictions.csv", "hybrid_pred", False),
}
MODEL_LABELS = {"lr_ew": "LR-EW", "xgb_ew": "XGB-EW", "hybrid": "LSTM Hybrid"}

# finishing band -> (rank_lo, rank_hi) inclusive, 1-based.
GROUPS = [
    ("P1-P5", 1, 5),
    ("P6-P10", 6, 10),
    ("P11-P15", 11, 15),
    ("P16-P20", 16, 20),
]


def repo_root() -> Path:
    # Scripts/Source/this_file.py -> repo root is two levels up.
    return Path(__file__).resolve().parents[2]


def pretty_circuit(safe_name: str) -> str:
    """'italian_grand_prix' -> 'Italian Grand Prix'."""
    return safe_name.replace("_", " ").title()


def discover_circuits(modeldata_dir: Path) -> list[str]:
    """Safe circuit names, derived from the cleaned-data filenames under ModelData."""
    safe_names = [
        p.name[: -len("_cleaned_data.csv")]
        for p in modeldata_dir.glob("*/*_cleaned_data.csv")
    ]
    return sorted(safe_names)


def load_cleaned(safe_name: str, modeldata_dir: Path) -> pd.DataFrame | None:
    """Locate the cleaned dataset for a safe circuit name (matched by filename prefix)."""
    matches = sorted(modeldata_dir.glob(f"*/{safe_name}_cleaned_data.csv"))
    if not matches:
        return None
    return pd.read_csv(matches[0])


def finishing_positions(cleaned: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return per-driver 2025 finishing rank (1-based), classified + DNF handling.

    Rank order: drivers with a non-NaN ``Position`` on their last lap come first, sorted by
    that position; retired drivers (NaN position on their last lap) follow, sorted by laps
    completed (descending) so that drivers who covered more distance rank higher.
    """
    race = cleaned[cleaned["Year"] == year].copy()
    race["LapNumber"] = pd.to_numeric(race["LapNumber"], errors="coerce")
    last = race.loc[race.groupby("Driver")["LapNumber"].idxmax()].copy()
    last["Position"] = pd.to_numeric(last["Position"], errors="coerce")

    classified = last[last["Position"].notna()].sort_values("Position", kind="mergesort")
    retired = last[last["Position"].isna()].sort_values(
        "LapNumber", ascending=False, kind="mergesort"
    )
    ordered = pd.concat([classified, retired], ignore_index=True)
    ordered["finish_rank"] = np.arange(1, len(ordered) + 1)
    return ordered[["Driver", "Position", "LapNumber", "finish_rank"]]


def group_for_rank(rank: int) -> str | None:
    for label, lo, hi in GROUPS:
        if lo <= rank <= hi:
            return label
    return None


def load_residuals(
    safe_name: str, model_key: str, baseline_dir: Path, cleaned: pd.DataFrame
) -> pd.DataFrame | None:
    """Return a frame with columns [Driver, residual] for a circuit/model holdout, or None."""
    suffix, pred_col, needs_driver = MODELS[model_key]
    path = baseline_dir / f"{safe_name}{suffix}"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if pred_col not in df.columns or "y_true" not in df.columns:
        raise ValueError(f"{path.name} missing required columns y_true/{pred_col}.")

    y_true = pd.to_numeric(df["y_true"], errors="coerce")
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    residual = y_true - pred

    if needs_driver:
        if "row_index" not in df.columns:
            raise ValueError(f"{path.name} has no row_index to recover Driver.")
        driver = cleaned.loc[df["row_index"].to_numpy(), "Driver"].to_numpy()
    else:
        if "Driver" not in df.columns:
            raise ValueError(f"{path.name} has no Driver column.")
        driver = df["Driver"].to_numpy()

    out = pd.DataFrame({"Driver": driver, "residual": residual})
    return out[out["residual"].notna()].reset_index(drop=True)


def summarise(residuals: pd.DataFrame, finish: pd.DataFrame) -> pd.DataFrame:
    """Per finishing band: residual mean/std plus driver and lap counts."""
    merged = residuals.merge(finish[["Driver", "finish_rank"]], on="Driver", how="left")
    merged["group"] = merged["finish_rank"].map(
        lambda r: group_for_rank(int(r)) if pd.notna(r) else None
    )

    rows = []
    for label, _, _ in GROUPS:
        sel = merged[merged["group"] == label]
        rows.append(
            {
                "group": label,
                "n_drivers": int(sel["Driver"].nunique()),
                "n_laps": int(len(sel)),
                "residual_mean": float(sel["residual"].mean()) if len(sel) else float("nan"),
                "residual_std": float(sel["residual"].std(ddof=1)) if len(sel) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def plot_model(model_key: str, per_circuit: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    """Grouped bar chart of residual mean +/- std across finishing bands, one panel/circuit."""
    circuits = sorted(per_circuit)
    n = len(circuits)
    ncols = 3 if n > 4 else 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.6 * nrows), squeeze=False)
    labels = [g[0] for g in GROUPS]
    x = np.arange(len(labels))

    for idx, circuit in enumerate(circuits):
        ax = axes[idx // ncols][idx % ncols]
        summary = per_circuit[circuit].set_index("group").reindex(labels)
        means = summary["residual_mean"].to_numpy(dtype=float)
        stds = summary["residual_std"].to_numpy(dtype=float)
        ax.axhline(0.0, color="black", linewidth=1.0, zorder=1)
        ax.bar(
            x, means, yerr=stds, capsize=4, color="#0b3d91", edgecolor="black",
            linewidth=0.4, alpha=0.85, zorder=2, error_kw={"elinewidth": 1.2},
        )
        ax.set_title(pretty_circuit(circuit), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Residual (s)", fontsize=10)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    for empty in range(n, nrows * ncols):
        axes[empty // ncols][empty % ncols].axis("off")

    fig.suptitle(
        f"Holdout residuals by 2025 finishing band — {MODEL_LABELS[model_key]}\n"
        "bar = mean residual, whisker = std",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = output_dir / f"residuals_by_finish_group_{model_key}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"residuals_by_finish_group_{model_key}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> None:
    root = repo_root()
    baseline_dir = root / BASELINE_DIR
    modeldata_dir = root / MODELDATA_DIR
    output_dir = root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    circuits = discover_circuits(baseline_dir)
    if not circuits:
        raise SystemExit(f"No holdout prediction files found under {baseline_dir}")

    tidy_rows: list[dict] = []
    finish_rows: list[dict] = []
    per_model_plots: dict[str, dict[str, pd.DataFrame]] = {m: {} for m in MODELS}

    print("\n=== HOLDOUT RESIDUALS BY 2025 FINISHING BAND ===")
    for safe_name in circuits:
        cleaned = load_cleaned(safe_name, modeldata_dir)
        if cleaned is None:
            print(f"  ! {safe_name}: cleaned dataset not found, skipping circuit")
            continue
        finish = finishing_positions(cleaned, HOLDOUT_YEAR)
        finish_circuit = finish.assign(circuit=pretty_circuit(safe_name))
        finish_rows.extend(finish_circuit.to_dict("records"))

        print(f"\n# {pretty_circuit(safe_name)}")
        for model_key in MODELS:
            residuals = load_residuals(safe_name, model_key, baseline_dir, cleaned)
            if residuals is None:
                print(f"  - {MODEL_LABELS[model_key]:11s}: holdout predictions absent (run the model)")
                continue
            summary = summarise(residuals, finish)
            per_model_plots[model_key][safe_name] = summary
            for _, r in summary.iterrows():
                tidy_rows.append(
                    {
                        "circuit": pretty_circuit(safe_name),
                        "model": MODEL_LABELS[model_key],
                        "group": r["group"],
                        "n_drivers": int(r["n_drivers"]),
                        "n_laps": int(r["n_laps"]),
                        "residual_mean": r["residual_mean"],
                        "residual_std": r["residual_std"],
                    }
                )
            shown = summary.assign(model=MODEL_LABELS[model_key])
            print(f"  - {MODEL_LABELS[model_key]}:")
            for _, r in summary.iterrows():
                print(
                    f"      {r['group']:7s} | drivers={int(r['n_drivers']):2d} laps={int(r['n_laps']):4d} "
                    f"| mean={r['residual_mean']:+.4f} std={r['residual_std']:.4f}"
                )

    if not tidy_rows:
        raise SystemExit("No residuals summarised; ensure holdout prediction CSVs exist.")

    tidy = pd.DataFrame(tidy_rows)
    tidy_path = output_dir / "residuals_by_finish_group.csv"
    tidy.to_csv(tidy_path, index=False)

    finish_df = pd.DataFrame(finish_rows)
    finish_path = output_dir / "finishing_positions_2025.csv"
    finish_df.to_csv(finish_path, index=False)

    # Wide table: rows (circuit, group) x columns (model -> mean / std).
    wide = tidy.pivot_table(
        index=["circuit", "group"], columns="model",
        values=["residual_mean", "residual_std"],
    )
    wide_path = output_dir / "residuals_by_finish_group_wide.csv"
    wide.to_csv(wide_path)

    plots = []
    for model_key, per_circuit in per_model_plots.items():
        if per_circuit:
            plots.append(plot_model(model_key, per_circuit, output_dir))

    print("\n--- Saved outputs ---")
    print(f"  tidy table:   {tidy_path}")
    print(f"  wide table:   {wide_path}")
    print(f"  finish table: {finish_path}")
    for p in plots:
        print(f"  plot:         {p}")
    print(f"\nAll outputs under: {output_dir}")


if __name__ == "__main__":
    main()
