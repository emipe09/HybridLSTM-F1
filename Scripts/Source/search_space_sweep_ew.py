"""
search_space_sweep_ew.py

Runs the same 19 baseline XGBoost configurations as search_space_sweep.py,
but uses the **final selected EW window size** for each circuit
(xgb_ew_window_ratio from the YAML) and evaluates on expanding windows instead
of sliding windows.

This produces a circuit-specific search space grounded in the exact validation
protocol and window size that will be used in the final experiment.

Usage (Linux / macOS):
    TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/search_space_sweep_ew.py
    TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/search_space_sweep_ew.py --dry-run
    python Scripts/Source/search_space_sweep_ew.py --all-gps
    python Scripts/Source/search_space_sweep_ew.py --all-gps --dry-run

Usage (Windows / PowerShell):
    $env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/search_space_sweep_ew.py
    $env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/search_space_sweep_ew.py --dry-run

Options:
    --dry-run        Print derived bounds without writing YAML files.
    --all-gps        Run for all supported GPs (TARGET_GP_NAME is ignored).
    --top-n N        Number of top baselines used to derive bounds (default: 5).
    --max-windows N  Maximum number of expanding windows used for evaluation (default: 3).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from modeling_utils import (
    build_expanding_windows,
    build_sequential_split,
    load_cleaned_data,
    prepare_raw_features,
    resolve_repo_path,
    safe_gp_name,
    select_modeling_columns,
)
from xgb_utils import BASE_XGB_PARAMS, build_xgb_matrix


# ---------------------------------------------------------------------------
# Supported GPs
# ---------------------------------------------------------------------------

ALL_GP_NAMES = [
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "United States Grand Prix",
    "Italian Grand Prix",
    "Hungarian Grand Prix",
]


# ---------------------------------------------------------------------------
# Baseline configurations (identical to search_space_sweep.py)
# ---------------------------------------------------------------------------

BASELINE_CONFIGS = [
    # XGBoost defaults
    {
        "name": "xgb_default",
        "learning_rate": 0.30, "max_depth": 6,  "min_child_weight": 1,
        "subsample": 1.00, "colsample_bytree": 1.00,
        "gamma": 0.00, "reg_alpha": 1e-5, "reg_lambda": 1.00,
    },
    # Very shallow trees (depth 2)
    {
        "name": "very_shallow_low_lr",
        "learning_rate": 0.02, "max_depth": 2, "min_child_weight": 5,
        "subsample": 0.80, "colsample_bytree": 0.80,
        "gamma": 0.10, "reg_alpha": 0.001, "reg_lambda": 1.00,
    },
    {
        "name": "very_shallow_mid_lr",
        "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 5,
        "subsample": 0.80, "colsample_bytree": 0.80,
        "gamma": 0.10, "reg_alpha": 0.001, "reg_lambda": 1.00,
    },
    # Shallow trees (depth 4)
    {
        "name": "shallow_low_lr",
        "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 5,
        "subsample": 0.75, "colsample_bytree": 0.85,
        "gamma": 0.20, "reg_alpha": 0.001, "reg_lambda": 0.50,
    },
    {
        "name": "shallow_mid_lr",
        "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5,
        "subsample": 0.75, "colsample_bytree": 0.85,
        "gamma": 0.20, "reg_alpha": 0.001, "reg_lambda": 0.50,
    },
    {
        "name": "shallow_high_lr",
        "learning_rate": 0.10, "max_depth": 4, "min_child_weight": 5,
        "subsample": 0.75, "colsample_bytree": 0.85,
        "gamma": 0.20, "reg_alpha": 0.001, "reg_lambda": 0.50,
    },
    # Medium trees (depth 6)
    {
        "name": "medium_low_lr",
        "learning_rate": 0.02, "max_depth": 6, "min_child_weight": 5,
        "subsample": 0.75, "colsample_bytree": 0.85,
        "gamma": 0.20, "reg_alpha": 0.010, "reg_lambda": 1.00,
    },
    {
        "name": "medium_mid_lr",
        "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5,
        "subsample": 0.75, "colsample_bytree": 0.85,
        "gamma": 0.20, "reg_alpha": 0.010, "reg_lambda": 1.00,
    },
    {
        "name": "medium_high_lr",
        "learning_rate": 0.10, "max_depth": 6, "min_child_weight": 5,
        "subsample": 0.75, "colsample_bytree": 0.85,
        "gamma": 0.20, "reg_alpha": 0.010, "reg_lambda": 1.00,
    },
    # Deep trees (depth 8)
    {
        "name": "deep_low_lr",
        "learning_rate": 0.02, "max_depth": 8, "min_child_weight": 5,
        "subsample": 0.70, "colsample_bytree": 0.80,
        "gamma": 0.50, "reg_alpha": 0.010, "reg_lambda": 2.00,
    },
    {
        "name": "deep_mid_lr",
        "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 5,
        "subsample": 0.70, "colsample_bytree": 0.80,
        "gamma": 0.50, "reg_alpha": 0.010, "reg_lambda": 2.00,
    },
    {
        "name": "deep_high_lr",
        "learning_rate": 0.10, "max_depth": 8, "min_child_weight": 5,
        "subsample": 0.70, "colsample_bytree": 0.80,
        "gamma": 0.50, "reg_alpha": 0.010, "reg_lambda": 2.00,
    },
    # Very deep trees (depth 10)
    {
        "name": "very_deep_low_lr",
        "learning_rate": 0.02, "max_depth": 10, "min_child_weight": 3,
        "subsample": 0.65, "colsample_bytree": 0.75,
        "gamma": 1.00, "reg_alpha": 0.050, "reg_lambda": 5.00,
    },
    {
        "name": "very_deep_mid_lr",
        "learning_rate": 0.05, "max_depth": 10, "min_child_weight": 3,
        "subsample": 0.65, "colsample_bytree": 0.75,
        "gamma": 1.00, "reg_alpha": 0.050, "reg_lambda": 5.00,
    },
    # High regularization
    {
        "name": "high_reg_medium",
        "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 20,
        "subsample": 0.70, "colsample_bytree": 0.75,
        "gamma": 2.00, "reg_alpha": 1.000, "reg_lambda": 10.00,
    },
    {
        "name": "high_reg_shallow",
        "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 15,
        "subsample": 0.75, "colsample_bytree": 0.80,
        "gamma": 1.00, "reg_alpha": 0.500, "reg_lambda": 5.00,
    },
    # Low regularization (expressive)
    {
        "name": "low_reg_medium",
        "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1,
        "subsample": 0.90, "colsample_bytree": 0.90,
        "gamma": 0.00, "reg_alpha": 1e-5, "reg_lambda": 0.10,
    },
    # High min_child_weight (conservative splits)
    {
        "name": "high_mcw_medium_lr",
        "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 25,
        "subsample": 0.75, "colsample_bytree": 0.80,
        "gamma": 0.50, "reg_alpha": 0.010, "reg_lambda": 1.00,
    },
    # Low min_child_weight + deep (flexible)
    {
        "name": "low_mcw_deep",
        "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 1,
        "subsample": 0.80, "colsample_bytree": 0.85,
        "gamma": 0.10, "reg_alpha": 0.001, "reg_lambda": 0.20,
    },
]

BASELINE_PARAM_NAMES = [
    "learning_rate", "max_depth", "min_child_weight",
    "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
]


# ---------------------------------------------------------------------------
# Parameter metadata for bounds derivation
# ---------------------------------------------------------------------------

PARAM_SPECS = {
    "learning_rate":    {"type": "float", "log": True,  "abs_min": 0.001,  "abs_max": 0.500},
    "max_depth":        {"type": "int",   "log": False, "abs_min": 1,      "abs_max": 15},
    "min_child_weight": {"type": "int",   "log": False, "abs_min": 1,      "abs_max": 50},
    "subsample":        {"type": "float", "log": False, "abs_min": 0.30,   "abs_max": 1.00},
    "colsample_bytree": {"type": "float", "log": False, "abs_min": 0.30,   "abs_max": 1.00},
    "gamma":            {"type": "gamma", "log": False, "abs_min": 0.00,   "abs_max": 10.00},
    "reg_alpha":        {"type": "float", "log": True,  "abs_min": 1e-8,   "abs_max": 100.0},
    "reg_lambda":       {"type": "float", "log": True,  "abs_min": 1e-8,   "abs_max": 100.0},
}

YAML_PARAM_KEYS = {
    "learning_rate":    ("xgb_learning_rate_min",    "xgb_learning_rate_max"),
    "max_depth":        ("xgb_max_depth_min",         "xgb_max_depth_max"),
    "min_child_weight": ("xgb_min_child_weight_min",  "xgb_min_child_weight_max"),
    "subsample":        ("xgb_subsample_min",          "xgb_subsample_max"),
    "colsample_bytree": ("xgb_colsample_bytree_min",  "xgb_colsample_bytree_max"),
    "gamma":            ("xgb_gamma_min",              "xgb_gamma_max"),
    "reg_alpha":        ("xgb_reg_alpha_min",         "xgb_reg_alpha_max"),
    "reg_lambda":       ("xgb_reg_lambda_min",        "xgb_reg_lambda_max"),
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _select_eval_windows(all_windows: list, max_windows: int) -> list:
    """Pick up to max_windows representative windows (first, middle, last)."""
    n = len(all_windows)
    if n <= max_windows:
        return all_windows
    indices_set: set[int] = {0, n // 2, n - 1}
    step = max(1, n // max_windows)
    for idx in range(0, n, step):
        indices_set.add(idx)
        if len(indices_set) >= max_windows:
            break
    selected = sorted(indices_set)[:max_windows]
    return [all_windows[i] for i in selected]


def _evaluate_baseline(
    baseline_params: dict,
    eval_windows: list,
    unique_laps: np.ndarray,
    lap_model_sorted: pd.Series,
    X_raw: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    seed: int,
    n_boost_max: int = 3000,
    early_stop: int = 50,
) -> dict:
    """Train and evaluate one baseline config on the given expanding windows."""
    xgb_params = {**BASE_XGB_PARAMS, **baseline_params, "seed": seed}
    window_rmses: list[float] = []
    window_maes: list[float] = []

    for start, split, end in eval_windows:
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        X_train, y_train = X_raw.loc[train_mask], y.loc[train_mask]
        X_val, y_val = X_raw.loc[val_mask], y.loc[val_mask]

        if len(X_train) < 2 or len(X_val) < 2:
            continue

        dtrain, dval, _, _ = build_xgb_matrix(X_train, X_val, y_train, y_val, cat_cols)
        booster = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=n_boost_max,
            evals=[(dval, "validation")],
            early_stopping_rounds=early_stop,
            verbose_eval=False,
        )
        best_iter = booster.best_iteration + 1
        preds = booster.predict(dval, iteration_range=(0, best_iter))
        window_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        window_maes.append(float(mean_absolute_error(y_val, preds)))

    if not window_rmses:
        return {"mean_rmse": float("inf"), "mean_mae": float("inf"), "n_windows": 0}

    return {
        "mean_rmse": float(np.mean(window_rmses)),
        "mean_mae": float(np.mean(window_maes)),
        "n_windows": len(window_rmses),
    }


# ---------------------------------------------------------------------------
# Bounds derivation
# ---------------------------------------------------------------------------

def _derive_bounds(
    top_configs: list[dict],
    margin: float = 0.25,
    int_pad: int = 1,
    log_factor: float = 3.0,
) -> dict:
    bounds: dict = {}

    for param, spec in PARAM_SPECS.items():
        values = [cfg[param] for cfg in top_configs]
        val_min = min(values)
        val_max = max(values)

        if spec["type"] == "int":
            low = max(int(spec["abs_min"]), int(val_min) - int_pad)
            high = min(int(spec["abs_max"]), int(val_max) + int_pad)
            if high - low < 2:
                center = (low + high) // 2
                low = max(int(spec["abs_min"]), center - 1)
                high = min(int(spec["abs_max"]), center + 1)
                if high <= low:
                    high = low + 1
            bounds[param] = (int(low), int(high))

        elif spec.get("log"):
            low = max(spec["abs_min"], val_min / log_factor)
            high = min(spec["abs_max"], val_max * log_factor)
            if high <= low:
                high = max(low * log_factor, low + 1e-8)
            bounds[param] = (float(low), float(high))

        elif spec["type"] == "gamma":
            low = max(0.0, val_min - 0.5)
            high = min(spec["abs_max"], max(val_max, 0.001) + 0.5)
            if high <= low:
                high = low + 0.5
            bounds[param] = (float(low), float(high))

        else:  # linear float
            center = (val_min + val_max) / 2
            half_range = max((val_max - val_min) / 2, center * 0.10, 0.01)
            low = max(spec["abs_min"], center - half_range * (1.0 + margin))
            high = min(spec["abs_max"], center + half_range * (1.0 + margin))
            if high <= low:
                high = min(spec["abs_max"], low + 0.05)
            bounds[param] = (float(low), float(high))

    return bounds


# ---------------------------------------------------------------------------
# YAML update
# ---------------------------------------------------------------------------

def _format_yaml_value(value: float, is_log: bool) -> str:
    abs_val = abs(value)
    if abs_val == 0.0:
        return "0.0"
    if abs_val < 0.0001:
        return f"{value:.8f}"
    if abs_val < 0.01:
        return f"{value:.6f}"
    if abs_val < 0.10:
        return f"{value:.4f}"
    if abs_val < 10.0:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _update_yaml_bounds(config_path: Path, bounds: dict, dry_run: bool) -> None:
    text = config_path.read_text(encoding="utf-8")

    for param, (low, high) in bounds.items():
        min_key, max_key = YAML_PARAM_KEYS[param]
        spec = PARAM_SPECS[param]

        if spec["type"] == "int":
            low_str = str(int(low))
            high_str = str(int(high))
        else:
            is_log = bool(spec.get("log", False))
            low_str = _format_yaml_value(float(low), is_log)
            high_str = _format_yaml_value(float(high), is_log)

        text = re.sub(
            rf"^({re.escape(min_key)}:).*$",
            rf"\g<1> {low_str}",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            rf"^({re.escape(max_key)}:).*$",
            rf"\g<1> {high_str}",
            text,
            flags=re.MULTILINE,
        )

    if dry_run:
        print(f"\n[dry-run] Would update: {config_path}")
        print("  Updated search space lines:")
        for line in text.splitlines():
            stripped = line.strip()
            if any(
                stripped.startswith(key)
                for keys in YAML_PARAM_KEYS.values()
                for key in keys
            ):
                print(f"    {line}")
    else:
        config_path.write_text(text, encoding="utf-8")
        print(f"  YAML updated: {config_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_results_table(ranked: list[dict], gp_name: str, window_ratio: float) -> None:
    print(f"\n{'=' * 72}")
    print(f"  Baseline results — {gp_name}  (EW window_ratio={window_ratio:.0%})")
    print(f"{'=' * 72}")
    header = f"{'Rank':>4}  {'Name':<24}  {'RMSE':>8}  {'MAE':>8}  {'Windows':>7}"
    print(header)
    print("-" * 72)
    for rank, row in enumerate(ranked, start=1):
        print(
            f"{rank:>4}  {row['name']:<24}  {row['mean_rmse']:>8.4f}"
            f"  {row['mean_mae']:>8.4f}  {row['n_windows']:>7}"
        )
    print()


def _fmt(value, spec) -> str:
    if value is None:
        return "?"
    if spec["type"] == "int":
        return str(int(value))
    is_log = bool(spec.get("log", False))
    return _format_yaml_value(float(value), is_log)


def _print_bounds_comparison(
    current_bounds: dict,
    derived_bounds: dict,
    gp_name: str,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"  Search space bounds — {gp_name}")
    print(f"{'=' * 72}")
    print(f"  {'Parameter':<22}  {'Current [low, high]':^26}  {'Derived [low, high]':^26}")
    print("  " + "-" * 78)
    for param in BASELINE_PARAM_NAMES:
        spec = PARAM_SPECS[param]
        cur = current_bounds.get(param, (None, None))
        der = derived_bounds.get(param, (None, None))
        cur_str = f"[{_fmt(cur[0], spec)}, {_fmt(cur[1], spec)}]" if cur[0] is not None else "(not set)"
        der_str = f"[{_fmt(der[0], spec)}, {_fmt(der[1], spec)}]" if der[0] is not None else "(not set)"
        changed = " *" if cur != der else "  "
        print(f"  {param:<22}  {cur_str:^26}  {der_str:^26}{changed}")
    print()
    print("  * = value changed")
    print()


def _read_current_bounds(config: dict) -> dict:
    bounds: dict = {}
    for param, (min_key, max_key) in YAML_PARAM_KEYS.items():
        spec = PARAM_SPECS[param]
        raw_low = config.get(min_key)
        raw_high = config.get(max_key)
        if raw_low is None or raw_high is None:
            bounds[param] = (None, None)
            continue
        if spec["type"] == "int":
            bounds[param] = (int(raw_low), int(raw_high))
        else:
            bounds[param] = (float(raw_low), float(raw_high))
    return bounds


# ---------------------------------------------------------------------------
# Per-GP orchestration
# ---------------------------------------------------------------------------

def _run_for_gp(
    gp_name: str,
    top_n: int,
    max_windows: int,
    dry_run: bool,
    output_dir: Path,
) -> None:
    print(f"\n{'#' * 72}")
    print(f"  {gp_name}")
    print(f"{'#' * 72}")

    target_gp_name, config, repo_root, df_raw = load_cleaned_data(Path(__file__))

    num_cols, cat_cols = select_modeling_columns(df_raw, config)
    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    holdout_ratio = float(config["holdout_ratio"])
    window_train_ratio = float(config["window_train_ratio"])
    window_step_ratio = float(config["window_step_ratio"])
    seed = int(config["random_seed"])

    # Use the final selected EW window size for this circuit
    window_ratio = float(config.get("xgb_ew_window_ratio", config["window_ratio"]))

    X_raw, y, valid_indices = prepare_raw_features(df_raw, num_cols, cat_cols, target_col)
    df_valid = df_raw.loc[valid_indices].sort_values(["Year", lap_col]).reset_index(drop=True)
    X_raw = X_raw.loc[df_valid.index] if set(df_valid.index).issubset(set(X_raw.index)) else X_raw
    y = y.loc[df_valid.index] if set(df_valid.index).issubset(set(y.index)) else y

    (
        step_series,
        _step_min,
        _step_max,
        model_idx,
        _holdout_idx,
        _holdout_start,
        _model_end,
        _total_laps,
    ) = build_sequential_split(df_valid, df_valid.index, holdout_ratio, lap_col)

    lap_model_sorted = step_series.loc[model_idx]
    unique_laps = np.sort(lap_model_sorted.unique())
    n_model_laps = len(unique_laps)

    X_model_raw = X_raw.loc[model_idx]
    y_model = y.loc[model_idx]

    all_windows, _ws, _ts, _vs, _ss = build_expanding_windows(
        n_model_laps, window_ratio, window_train_ratio, window_step_ratio
    )
    eval_windows = _select_eval_windows(all_windows, max_windows)

    print(
        f"  EW window_ratio: {window_ratio:.0%} (xgb_ew_window_ratio) | "
        f"Modeling block: {n_model_laps} unique lap steps | "
        f"All windows: {len(all_windows)} | Eval windows: {len(eval_windows)}"
    )
    print(f"  Evaluating {len(BASELINE_CONFIGS)} baseline configs...\n")

    results: list[dict] = []
    for i, baseline in enumerate(BASELINE_CONFIGS, start=1):
        baseline_params = {k: v for k, v in baseline.items() if k != "name"}
        metrics = _evaluate_baseline(
            baseline_params,
            eval_windows,
            unique_laps,
            lap_model_sorted,
            X_model_raw,
            y_model,
            cat_cols,
            seed,
        )
        row = {"name": baseline["name"], "params": baseline_params, **metrics}
        results.append(row)
        print(
            f"  [{i:02d}/{len(BASELINE_CONFIGS)}] {baseline['name']:<24}"
            f"  RMSE={metrics['mean_rmse']:.4f}  MAE={metrics['mean_mae']:.4f}"
        )

    ranked = sorted(results, key=lambda r: r["mean_rmse"])
    _print_results_table(ranked, gp_name, window_ratio)

    top_configs = [r["params"] for r in ranked[:top_n]]
    derived_bounds = _derive_bounds(top_configs)
    current_bounds = _read_current_bounds(config)
    _print_bounds_comparison(current_bounds, derived_bounds, gp_name)

    from modeling_utils import CONFIG_ALIASES, safe_gp_name as _sgn
    configs_dir = repo_root / "configs"
    safe_name = _sgn(gp_name)
    yaml_filename = CONFIG_ALIASES.get(safe_name, f"{safe_name}.yaml")
    config_path = configs_dir / yaml_filename

    _update_yaml_bounds(config_path, derived_bounds, dry_run)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{safe_name}_baseline_results_ew.json"
    serializable = {
        "gp_name": gp_name,
        "validation_scheme": "expanding_window",
        "window_ratio": window_ratio,
        "top_n": top_n,
        "eval_windows": len(eval_windows),
        "all_windows": len(all_windows),
        "ranked_baselines": [
            {
                "rank": rank,
                "name": r["name"],
                "mean_rmse": r["mean_rmse"],
                "mean_mae": r["mean_mae"],
                "n_windows": r["n_windows"],
                "params": r["params"],
            }
            for rank, r in enumerate(ranked, start=1)
        ],
        "derived_bounds": {
            param: {"low": low, "high": high}
            for param, (low, high) in derived_bounds.items()
        },
        "current_bounds": {
            param: {"low": low, "high": high}
            for param, (low, high) in current_bounds.items()
            if low is not None
        },
    }
    output_file.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"  Baseline results saved: {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline XGBoost configs on the final EW window size and update YAML search space bounds."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print derived bounds without writing YAML files.",
    )
    parser.add_argument(
        "--all-gps",
        action="store_true",
        help="Run for all supported GPs (TARGET_GP_NAME is ignored).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of top baselines used to derive bounds (default: 5).",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=3,
        metavar="N",
        help="Maximum number of expanding windows used for evaluation (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    output_dir = resolve_repo_path(repo_root, "Scripts/Results/search_space_sweep")

    gp_names = ALL_GP_NAMES if args.all_gps else [None]

    for gp_name in gp_names:
        if gp_name is not None:
            os.environ["TARGET_GP_NAME"] = gp_name
        elif "TARGET_GP_NAME" not in os.environ:
            print(
                "ERROR: Set TARGET_GP_NAME or pass --all-gps.\n"
                "Example: TARGET_GP_NAME=\"Bahrain Grand Prix\" "
                "python Scripts/Source/search_space_sweep_ew.py"
            )
            sys.exit(1)

        effective_gp_name = os.environ["TARGET_GP_NAME"]
        _run_for_gp(
            gp_name=effective_gp_name,
            top_n=args.top_n,
            max_windows=args.max_windows,
            dry_run=args.dry_run,
            output_dir=output_dir,
        )

    if args.dry_run:
        print("\n[dry-run] No YAML files were modified.")
    else:
        print("\nDone. Re-run your XGBoost scripts with use_saved_xgb_params: false to apply the new search spaces.")


if __name__ == "__main__":
    main()
