"""Out-of-sample tabular baseline generation for the hybrid LSTM-residual model.

The hybrid model trains an LSTM to predict the residual of the best tabular model
(LR-EW or XGBoost-EW) selected per circuit. To avoid stacking leakage, every tabular
prediction that feeds an LSTM *training target* must be out-of-sample. This module
provides two primitives:

  - ``generate_oof_baseline``: out-of-fold predictions over an ordered block using the
    same expanding-window folds as the tabular models. Used for LSTM training targets.
  - ``generate_block_baseline``: fit on one block, predict another (strictly OOS). Used
    for validation targets (train -> val) and holdout targets (modeling -> holdout).

Tabular hyperparameters are reused, never re-derived from the holdout:
  - LR-EW has no hyperparameters (``fit_predict_linear_regression``).
  - XGBoost-EW reuses the circuit's saved EW parameters via ``tune_or_load_params_ew``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from modeling_utils import (
    build_expanding_windows,
    build_sequential_split,
    build_xgb_ew_params_path,
    fit_predict_linear_regression,
    prepare_raw_features,
    resolve_repo_path,
    safe_gp_name,
)
from xgb_utils import build_xgb_matrix

BASELINE_MODEL_KINDS = ("lr_ew", "xgb_ew")


def resolve_baseline_window_ratio(config: dict, model_kind: str) -> float:
    """Window ratio of the selected tabular model (mirrors model_lr_ew/model_xgb_ew)."""
    if model_kind == "lr_ew":
        return float(config.get("lr_ew_window_ratio", config["window_ratio"]))
    if model_kind == "xgb_ew":
        return float(config.get("xgb_ew_window_ratio", config["window_ratio"]))
    raise ValueError(f"Unsupported baseline model_kind={model_kind!r}. Use one of {BASELINE_MODEL_KINDS}.")


def resolve_xgb_ew_hparams(
    repo_root, config, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols
):
    """Return (train_params, best_n) for XGBoost-EW, reusing saved params when available.

    Builds the modeling-block expanding folds with ``xgb_ew_window_ratio`` so the
    Optuna fallback inside ``tune_or_load_params_ew`` is consistent with the standalone
    XGBoost-EW script. When saved params match the current config, no tuning runs.
    """
    # Imported lazily to avoid importing TensorFlow-free heavy deps at module load.
    from model_xgb_ew import tune_or_load_params_ew

    params_path = build_xgb_ew_params_path(repo_root, config)
    windows, *_ = build_expanding_windows(
        len(unique_laps),
        resolve_baseline_window_ratio(config, "xgb_ew"),
        float(config["window_train_ratio"]),
        float(config["window_step_ratio"]),
    )
    train_params, best_n, _ = tune_or_load_params_ew(
        params_path, windows, unique_laps, lap_model_sorted, X_model_raw, y_model, cat_cols, config
    )
    return train_params, int(best_n)


def _predict_tabular(
    model_kind, X_fit, y_fit, X_target, cat_cols, xgb_train_params=None, xgb_best_n=None
):
    """Fit the tabular model on (X_fit, y_fit) and predict X_target. Returns np.ndarray."""
    if model_kind == "lr_ew":
        preds, *_ = fit_predict_linear_regression(X_fit, y_fit, X_target, cat_cols)
        return np.asarray(preds, dtype=float)
    if model_kind == "xgb_ew":
        if xgb_train_params is None or xgb_best_n is None:
            raise ValueError("xgb_ew baseline requires xgb_train_params and xgb_best_n.")
        y_dummy = pd.Series(np.zeros(len(X_target)), index=X_target.index)
        dtrain, dtarget, _, _ = build_xgb_matrix(X_fit, X_target, y_fit, y_dummy, cat_cols)
        booster = xgb.train(xgb_train_params, dtrain, num_boost_round=int(xgb_best_n))
        return np.asarray(booster.predict(dtarget), dtype=float)
    raise ValueError(f"Unsupported baseline model_kind={model_kind!r}. Use one of {BASELINE_MODEL_KINDS}.")


def generate_block_baseline(
    model_kind, X_fit, y_fit, X_target, cat_cols, xgb_train_params=None, xgb_best_n=None
):
    """Train on the fit block, predict the target block (strictly out-of-sample).

    Returns a Series aligned to ``X_target.index``.
    """
    preds = _predict_tabular(
        model_kind,
        X_fit.reset_index(drop=True),
        y_fit.reset_index(drop=True),
        X_target.reset_index(drop=True),
        cat_cols,
        xgb_train_params,
        xgb_best_n,
    )
    return pd.Series(preds, index=X_target.index)


def generate_oof_baseline(
    model_kind,
    X_block,
    y_block,
    lap_block_sorted,
    cat_cols,
    config,
    window_ratio,
    xgb_train_params=None,
    xgb_best_n=None,
):
    """Out-of-fold tabular predictions over an ordered block (Year->LapNumber preserved).

    Coverage strategy (100% of the block, every target gets a baseline):
      - Expanding folds (``build_expanding_windows``) supply OOS predictions for each
        validation chunk, i.e. laps in ``[train_size, last_fold_end)``.
      - The first ``train_size`` laps have no earlier data; they receive an in-sample
        prediction from a model fit on those same laps. These oldest laps are also the
        ones the LSTM drops first when building sequences, so the in-sample fallback has
        minimal effect and is documented here.
      - Any uncovered tail laps ``[last_fold_end, n_laps)`` are predicted out-of-sample
        from a model fit on all preceding laps.

    The three inputs must share the same index; the returned Series is aligned to
    ``X_block.index`` with no NaNs. The index is preserved (not reset) so callers can pass
    a row subset (e.g. only train_laps) and map the result back positionally.
    """
    if not (X_block.index.equals(y_block.index) and X_block.index.equals(lap_block_sorted.index)):
        raise ValueError("generate_oof_baseline requires X_block, y_block and lap_block_sorted to share an index.")

    unique_laps = np.sort(pd.to_numeric(lap_block_sorted, errors="coerce").dropna().unique())
    n_laps = len(unique_laps)
    windows, window_size, train_size, val_size, step_size = build_expanding_windows(
        n_laps,
        float(window_ratio),
        float(config["window_train_ratio"]),
        float(config["window_step_ratio"]),
    )

    oof = pd.Series(np.nan, index=X_block.index, dtype=float)

    def _fill(fit_laps, target_laps, only_missing):
        fit_mask = lap_block_sorted.isin(fit_laps)
        target_mask = lap_block_sorted.isin(target_laps)
        if not fit_mask.any() or not target_mask.any():
            return
        preds = _predict_tabular(
            model_kind,
            X_block.loc[fit_mask],
            y_block.loc[fit_mask],
            X_block.loc[target_mask],
            cat_cols,
            xgb_train_params,
            xgb_best_n,
        )
        target_index = X_block.loc[target_mask].index
        assign = pd.Series(preds, index=target_index)
        if only_missing:
            assign = assign[oof.loc[target_index].isna().to_numpy()]
        oof.loc[assign.index] = assign.to_numpy()

    # Expanding folds: OOS predictions for each validation chunk.
    for start, split, end in windows:
        _fill(unique_laps[start:split], unique_laps[split:end], only_missing=False)

    # First train_size laps: in-sample fallback (no earlier data exists).
    init_laps = unique_laps[:train_size]
    _fill(init_laps, init_laps, only_missing=True)

    # Uncovered tail laps: fit on everything before them, predict OOS.
    if oof.isna().any():
        tail_laps = np.sort(
            pd.to_numeric(lap_block_sorted.loc[oof.isna()], errors="coerce").dropna().unique()
        )
        fit_laps = unique_laps[~np.isin(unique_laps, tail_laps)]
        _fill(fit_laps, tail_laps, only_missing=True)

    if oof.isna().any():
        missing = int(oof.isna().sum())
        raise ValueError(f"OOF baseline left {missing} records uncovered; check lap continuity.")
    return oof


def baseline_prediction_paths(repo_root, config, model_kind):
    """CSV paths for the persisted baseline predictions of a given tabular model."""
    safe_name = safe_gp_name(str(config["target_gp_name"]))
    subdir = str(config.get("hybrid_baseline_predictions_subdir", "lstm_hybrid/baseline"))
    base_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / subdir
    return (
        base_dir / f"{safe_name}_{model_kind}_oof_predictions.csv",
        base_dir / f"{safe_name}_{model_kind}_holdout_predictions.csv",
    )


def _ordered_blocks(df_base, config, num_cols_tab, cat_cols_tab, target_col, lap_col):
    """Reproduce the tabular EW split/ordering so predictions key to the original index."""
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols_tab, cat_cols_tab, target_col)
    lap_series, _, _, model_idx, holdout_idx, _, _, _ = build_sequential_split(
        df_base, valid_indices, float(config["holdout_ratio"]), lap_col
    )
    model_order_idx = lap_series.loc[model_idx].sort_values(kind="mergesort").index
    holdout_order_idx = lap_series.loc[holdout_idx].sort_values(kind="mergesort").index
    return X_raw, y_raw, lap_series, model_order_idx, holdout_order_idx


def export_baseline_predictions(
    repo_root, config, model_kind, df_base, num_cols_tab, cat_cols_tab, target_col, lap_col
):
    """Persist the tabular model's out-of-fold (modeling) and holdout predictions.

    Self-contained: re-derives the same temporal split/ordering as the tabular EW scripts
    and writes two CSVs keyed by the original dataframe index, so the hybrid can reuse the
    exact baseline instead of regenerating it. Does not touch any existing artifact.
    """
    if model_kind not in BASELINE_MODEL_KINDS:
        raise ValueError(f"Unsupported baseline model_kind={model_kind!r}.")

    X_raw, y_raw, _, model_order_idx, holdout_order_idx = _ordered_blocks(
        df_base, config, num_cols_tab, cat_cols_tab, target_col, lap_col
    )
    X_model = X_raw.loc[model_order_idx]
    y_model = y_raw.loc[model_order_idx]
    lap_model = pd.to_numeric(df_base.loc[model_order_idx, lap_col], errors="coerce")
    X_holdout = X_raw.loc[holdout_order_idx]
    y_holdout = y_raw.loc[holdout_order_idx]

    window_ratio = resolve_baseline_window_ratio(config, model_kind)
    xgb_params, xgb_best_n = (None, None)
    if model_kind == "xgb_ew":
        unique_laps = np.sort(lap_model.dropna().unique())
        xgb_params, xgb_best_n = resolve_xgb_ew_hparams(
            repo_root, config, unique_laps, lap_model, X_model, y_model, cat_cols_tab
        )

    oof = generate_oof_baseline(
        model_kind, X_model, y_model, lap_model, cat_cols_tab, config, window_ratio,
        xgb_params, xgb_best_n,
    )
    holdout_pred = generate_block_baseline(
        model_kind, X_model, y_model, X_holdout, cat_cols_tab, xgb_params, xgb_best_n
    )

    oof_path, holdout_path = baseline_prediction_paths(repo_root, config, model_kind)
    oof_path.parent.mkdir(parents=True, exist_ok=True)

    def _frame(index, y_series, pred_series):
        data = {"row_index": np.asarray(index)}
        if "Year" in df_base.columns:
            data["Year"] = df_base.loc[index, "Year"].to_numpy()
        data[lap_col] = df_base.loc[index, lap_col].to_numpy()
        data["y_true"] = y_series.loc[index].to_numpy()
        data["baseline"] = pred_series.loc[index].to_numpy()
        return pd.DataFrame(data)

    _frame(model_order_idx, y_model, oof).to_csv(oof_path, index=False)
    _frame(holdout_order_idx, y_holdout, holdout_pred).to_csv(holdout_path, index=False)
    print(f"Saved baseline predictions ({model_kind}): {oof_path.name}, {holdout_path.name}")
    return oof_path, holdout_path


def load_baseline_predictions(repo_root, config, model_kind):
    """Load persisted baseline predictions as (oof_series, holdout_series) indexed by
    the original dataframe row index, or (None, None) if not available."""
    oof_path, holdout_path = baseline_prediction_paths(repo_root, config, model_kind)
    if not (oof_path.exists() and holdout_path.exists()):
        return None, None
    oof_df = pd.read_csv(oof_path).set_index("row_index")
    holdout_df = pd.read_csv(holdout_path).set_index("row_index")
    return oof_df["baseline"], holdout_df["baseline"]


def _audit_main():
    """Standalone leakage audit for the configured circuit (Etapa 1 verification).

    Generates the three baseline series and writes them to CSV alongside coverage and
    RMSE diagnostics, so non-leakage can be inspected manually.
    """
    import json
    from pathlib import Path

    from sklearn.metrics import mean_squared_error

    from modeling_utils import (
        build_sequential_split,
        load_cleaned_data,
        prepare_raw_features,
        resolve_repo_path,
        safe_gp_name,
        select_modeling_columns,
    )

    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()
    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])

    model_kind = str(config.get("hybrid_baseline_model", "")).lower()
    if model_kind not in BASELINE_MODEL_KINDS:
        raise ValueError(
            f"hybrid_baseline_model must be one of {BASELINE_MODEL_KINDS}; got {model_kind!r}."
        )
    window_ratio = resolve_baseline_window_ratio(config, model_kind)

    num_cols, cat_cols = select_modeling_columns(df_base, config)
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    (lap_series, _, _, model_idx, holdout_idx, _, _, _) = build_sequential_split(
        df_base, valid_indices, float(config["holdout_ratio"]), lap_col
    )
    model_laps = lap_series.loc[model_idx]
    model_order_idx = model_laps.sort_values(kind="mergesort").index
    holdout_laps = lap_series.loc[holdout_idx]
    holdout_order_idx = holdout_laps.sort_values(kind="mergesort").index

    X_model = X_raw.loc[model_order_idx].reset_index(drop=True)
    y_model = y_raw.loc[model_order_idx].reset_index(drop=True)
    lap_model_sorted = model_laps.loc[model_order_idx].reset_index(drop=True)
    X_holdout = X_raw.loc[holdout_order_idx].reset_index(drop=True)
    y_holdout = y_raw.loc[holdout_order_idx].reset_index(drop=True)

    unique_laps = np.sort(pd.to_numeric(lap_model_sorted, errors="coerce").dropna().unique())
    xgb_params, xgb_best_n = (None, None)
    if model_kind == "xgb_ew":
        xgb_params, xgb_best_n = resolve_xgb_ew_hparams(
            repo_root, config, unique_laps, lap_model_sorted, X_model, y_model, cat_cols
        )

    print(f"--- Baseline audit: {target_gp_name} | model={model_kind} | window_ratio={window_ratio} ---")
    oof = generate_oof_baseline(
        model_kind, X_model, y_model, lap_model_sorted, cat_cols, config, window_ratio,
        xgb_params, xgb_best_n,
    )
    holdout_baseline = generate_block_baseline(
        model_kind, X_model, y_model, X_holdout, cat_cols, xgb_params, xgb_best_n
    )

    oof_rmse = float(np.sqrt(mean_squared_error(y_model, oof)))
    holdout_rmse = float(np.sqrt(mean_squared_error(y_holdout, holdout_baseline)))
    print(f"OOF coverage: {int(oof.notna().sum())}/{len(oof)} (no NaN={not oof.isna().any()})")
    print(f"OOF baseline RMSE (modeling block): {oof_rmse:.4f}")
    print(f"Holdout baseline RMSE: {holdout_rmse:.4f}")

    subdir = str(config.get("hybrid_baseline_predictions_subdir", "lstm_hybrid/baseline"))
    out_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_gp_name(target_gp_name)
    pd.DataFrame({"lap": lap_model_sorted, "y": y_model, "baseline_oof": oof}).to_csv(
        out_dir / f"{safe_name}_baseline_oof.csv", index=False
    )
    pd.DataFrame({"y": y_holdout, "baseline_holdout": holdout_baseline}).to_csv(
        out_dir / f"{safe_name}_baseline_holdout.csv", index=False
    )
    (out_dir / f"{safe_name}_baseline_audit.json").write_text(
        json.dumps(
            {
                "model_kind": model_kind,
                "window_ratio": window_ratio,
                "oof_rmse": oof_rmse,
                "holdout_rmse": holdout_rmse,
                "oof_records": int(len(oof)),
                "holdout_records": int(len(holdout_baseline)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved baseline audit artifacts to: {out_dir}")


if __name__ == "__main__":
    _audit_main()
