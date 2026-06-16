"""LSTM with single sequential train/val split and sequential holdout.

Validation protocol:
  - Sequence length  = ceil(n_race_laps * lstm_window_ratio)
  - Modeling block   = first (1 - holdout_ratio) of all laps.
  - Train split      = first window_train_ratio of the modeling block.
  - Val split        = remaining (1 - window_train_ratio) of the modeling block.
  - Holdout          = last holdout_ratio of all laps (never used during training or tuning).
  - Final model      = retrained on full modeling block for the calibrated epoch count.

Single sequential split is used instead of expanding/sliding window because:
  - With grouping by (Year, Driver), each group contributes ~50 sequences after windowing.
  - Multiple folds would fragment this already small pool and multiply training cost linearly.
  - EarlyStopping on val_loss robustly calibrates the epoch count on the single split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from modeling_utils import (
    align_one_hot,
    build_sequential_split,
    calc_holdout_ci,
    calc_stats,
    log_mlflow_run,
    load_cleaned_data,
    prepare_raw_features,
    resolve_repo_path,
    safe_gp_name,
    select_modeling_columns,
    summarize_cos,
)

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "TensorFlow is required for the LSTM model. Install project dependencies with "
        "`pip install -r Utils/requirements.txt` before running model_lstm.py."
    ) from exc


LSTM_SEARCH_SPACE_VERSION = "v8"
LSTM_TUNING_STRATEGY = "single_sequential_split_v1"

DEFAULT_LSTM_CONFIG = {
    "lstm_units": 64,
    "lstm_dense_units": 32,
    "lstm_dropout": 0.2,
    "lstm_recurrent_dropout": 0.0,
    "lstm_learning_rate": 0.001,
    "lstm_batch_size": 32,
    "lstm_epochs": 100,
    "lstm_patience": 10,
    "lstm_group_cols": ["Year", "Driver"],
    "lstm_tuning_enabled": True,
    "lstm_optuna_trials": 20,
    "lstm_tuning_epochs": 40,
    "lstm_tuning_patience": 5,
    "lstm_min_final_epochs": 10,
    "lstm_l2_reg": 0.0,
    "lstm_stacked": False,
    "lstm_models_subdir": "lstm/models",
    "lstm_model_filename_template": "{safe_gp_name}_lstm_model.keras",
    "lstm_model_metadata_filename_template": "{safe_gp_name}_lstm_model_metadata.json",
    "use_saved_lstm_params": False,
    "lstm_params_subdir": "lstm/params",
    "lstm_params_filename_template": "{safe_gp_name}_lstm_params.json",
    "lstm_trials_filename_template": "{safe_gp_name}_lstm_optuna_trials.csv",
}


def lstm_config(config: dict) -> dict:
    return {**DEFAULT_LSTM_CONFIG, **{k: v for k, v in config.items() if k.startswith("lstm_")}}


def build_lstm_model_paths(repo_root: Path, config: dict, lstm_cfg: dict) -> tuple[Path, Path]:
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    model_filename = str(lstm_cfg["lstm_model_filename_template"]).format(
        target_gp_name=target_gp_name, safe_gp_name=safe_name
    )
    metadata_filename = str(lstm_cfg["lstm_model_metadata_filename_template"]).format(
        target_gp_name=target_gp_name, safe_gp_name=safe_name
    )
    model_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / str(lstm_cfg["lstm_models_subdir"])
    return model_dir / model_filename, model_dir / metadata_filename


def build_lstm_params_paths(repo_root: Path, config: dict, lstm_cfg: dict) -> tuple[Path, Path]:
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    params_filename = str(lstm_cfg["lstm_params_filename_template"]).format(
        target_gp_name=target_gp_name, safe_gp_name=safe_name
    )
    trials_filename = str(lstm_cfg["lstm_trials_filename_template"]).format(
        target_gp_name=target_gp_name, safe_gp_name=safe_name
    )
    params_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / str(lstm_cfg["lstm_params_subdir"])
    return params_dir / params_filename, params_dir / trials_filename


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def fit_feature_transformers(X_train: pd.DataFrame, X_context: pd.DataFrame, cat_cols: list[str]):
    X_train_enc, X_context_enc = align_one_hot(X_train, X_context, cat_cols, drop_first=False)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train_enc)
    X_context_imp = imputer.transform(X_context_enc)
    scaler.fit(X_train_imp)
    X_context_scaled = scaler.transform(X_context_imp)
    return (
        pd.DataFrame(X_context_scaled, index=X_context.index, columns=X_train_enc.columns),
        imputer,
        scaler,
        list(X_train_enc.columns),
    )


def build_sequences(
    X_scaled: pd.DataFrame,
    y: pd.Series,
    laps: pd.Series,
    groups: pd.DataFrame,
    target_indices: pd.Index,
    sequence_length: int,
):
    """Build (sequence_length × features) input tensors for each target index."""
    group_names = [f"__group_{i}__" for i in range(len(groups.columns))]
    sequence_groups = groups.reset_index(drop=True).copy()
    sequence_groups.columns = group_names
    context = pd.concat([X_scaled, y.rename("__target__"), laps.rename("__lap__"), sequence_groups], axis=1)
    target_index_set = set(target_indices)
    sequence_frames, targets, target_laps = [], [], []

    grouped = context.groupby(group_names, sort=False, dropna=False) if group_names else [(None, context)]
    for _, group in grouped:
        group = group.sort_values("__lap__", kind="mergesort")
        ordered_indices = list(group.index)
        for position, row_index in enumerate(ordered_indices):
            if row_index not in target_index_set or position < sequence_length:
                continue
            previous_indices = ordered_indices[position - sequence_length : position]
            sequence_frames.append(X_scaled.loc[previous_indices].to_numpy(dtype=np.float32))
            targets.append(float(y.loc[row_index]))
            target_laps.append(float(laps.loc[row_index]))

    if not sequence_frames:
        n_features = X_scaled.shape[1]
        return (
            np.empty((0, sequence_length, n_features), dtype=np.float32),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
        )
    return np.stack(sequence_frames), np.asarray(targets, dtype=float), np.asarray(target_laps, dtype=float)


def make_lstm_model(sequence_length: int, n_features: int, lstm_cfg: dict):
    l2_reg = float(lstm_cfg.get("lstm_l2_reg", 0.0))
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    stacked = bool(lstm_cfg.get("lstm_stacked", False))
    units = int(lstm_cfg["lstm_units"])
    dropout = float(lstm_cfg["lstm_dropout"])
    recurrent_dropout = float(lstm_cfg["lstm_recurrent_dropout"])

    layers = [tf.keras.layers.Input(shape=(sequence_length, n_features))]
    layers.append(
        tf.keras.layers.LSTM(
            units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            return_sequences=stacked,
        )
    )
    if stacked:
        layers.append(
            tf.keras.layers.LSTM(
                max(16, units // 2),
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
            )
        )
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(
        tf.keras.layers.Dense(
            int(lstm_cfg["lstm_dense_units"]),
            activation="relu",
            kernel_regularizer=regularizer,
        )
    )
    layers.append(tf.keras.layers.Dense(1))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lstm_cfg["lstm_learning_rate"])),
        loss="mse",
    )
    return model


def suggest_lstm_config(trial, base_cfg: dict) -> dict:
    # Search space v8 (v7→v8):
    # - lstm_sequence_length: removido do espaço de busca; fixado pelo usuário via
    #   lstm_window_ratio no YAML (sequence_length = ceil(n_race_laps * lstm_window_ratio)).
    # - Todos os outros bounds inalterados de v7.
    tuned = dict(base_cfg)
    tuned.update(
        {
            "lstm_units": trial.suggest_categorical("lstm_units", [64, 128]),
            "lstm_dense_units": trial.suggest_categorical("lstm_dense_units", [64, 128]),
            "lstm_dropout": trial.suggest_float("lstm_dropout", 0.05, 0.50),
            "lstm_recurrent_dropout": trial.suggest_float("lstm_recurrent_dropout", 0.20, 0.45),
            "lstm_learning_rate": trial.suggest_float("lstm_learning_rate", 3e-4, 5e-3, log=True),
            "lstm_batch_size": trial.suggest_categorical("lstm_batch_size", [32, 64]),
            "lstm_l2_reg": trial.suggest_float("lstm_l2_reg", 1e-4, 3e-3),
            "lstm_stacked": False,
            "lstm_epochs": int(base_cfg["lstm_tuning_epochs"]),
            "lstm_patience": int(base_cfg["lstm_tuning_patience"]),
        }
    )
    return tuned


def training_callbacks(lstm_cfg: dict):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(lstm_cfg["lstm_patience"]), restore_best_weights=True
        ),
    ]


def fit_predict_lstm(
    X_train: pd.DataFrame,
    X_context: pd.DataFrame,
    y_context: pd.Series,
    lap_context: pd.Series,
    group_context: pd.DataFrame,
    train_target_indices: pd.Index,
    eval_target_indices: pd.Index,
    cat_cols: list[str],
    lstm_cfg: dict,
    seed: int,
    epochs: int | None = None,
    callbacks_enabled: bool = True,
):
    set_random_seed(seed)
    sequence_length = int(lstm_cfg["lstm_sequence_length"])

    X_context_scaled, imputer, feature_scaler, feature_names = fit_feature_transformers(
        X_train.reset_index(drop=True),
        X_context.reset_index(drop=True),
        cat_cols,
    )
    y_ctx = y_context.reset_index(drop=True)
    lap_ctx = lap_context.reset_index(drop=True)
    group_ctx = group_context.reset_index(drop=True)

    X_train_seq, y_train_raw, _ = build_sequences(
        X_context_scaled, y_ctx, lap_ctx, group_ctx, train_target_indices, sequence_length
    )
    X_eval_seq, y_eval_seq, eval_laps = build_sequences(
        X_context_scaled, y_ctx, lap_ctx, group_ctx, eval_target_indices, sequence_length
    )

    if len(X_train_seq) == 0 or len(X_eval_seq) == 0:
        raise ValueError("Unable to build LSTM sequences. Check sequence_length vs. lap continuity.")

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_eval_scaled = target_scaler.transform(y_eval_seq.reshape(-1, 1)).ravel()

    model = make_lstm_model(sequence_length, X_train_seq.shape[2], lstm_cfg)
    fit_kwargs: dict = {
        "epochs": int(epochs or lstm_cfg["lstm_epochs"]),
        "batch_size": int(lstm_cfg["lstm_batch_size"]),
        "shuffle": False,
        "verbose": 0,
        "validation_data": (X_eval_seq, y_eval_scaled),
    }
    if callbacks_enabled:
        fit_kwargs["callbacks"] = training_callbacks(lstm_cfg)
    history = model.fit(X_train_seq, y_train_scaled, **fit_kwargs)

    preds_scaled = model.predict(X_eval_seq, verbose=0).ravel()
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    if history.history.get("val_loss"):
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    else:
        best_epoch = int(len(history.history.get("loss", [])) or fit_kwargs["epochs"])
    return preds, y_eval_seq, eval_laps, model, imputer, feature_scaler, target_scaler, feature_names, best_epoch


def fit_final_lstm(
    X_model: pd.DataFrame,
    y_model: pd.Series,
    lap_model: pd.Series,
    group_model: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    lap_holdout: pd.Series,
    group_holdout: pd.DataFrame,
    cat_cols: list[str],
    lstm_cfg: dict,
    seed: int,
    final_epoch_count: int,
):
    set_random_seed(seed)
    sequence_length = int(lstm_cfg["lstm_sequence_length"])

    X_model_r = X_model.reset_index(drop=True)
    y_model_r = y_model.reset_index(drop=True)
    lap_model_r = lap_model.reset_index(drop=True)
    group_model_r = group_model.reset_index(drop=True)

    X_holdout_r = X_holdout.reset_index(drop=True)
    y_holdout_r = y_holdout.reset_index(drop=True)
    lap_holdout_r = lap_holdout.reset_index(drop=True)
    group_holdout_r = group_holdout.reset_index(drop=True)

    X_model_scaled, imputer, feature_scaler, feature_names = fit_feature_transformers(
        X_model_r, X_model_r, cat_cols
    )
    X_holdout_scaled, _, _, _ = fit_feature_transformers(
        X_model_r, X_holdout_r, cat_cols
    )

    model_index = pd.RangeIndex(0, len(X_model_r))
    holdout_index = pd.RangeIndex(0, len(X_holdout_r))

    X_model_seq, y_model_raw, _ = build_sequences(
        X_model_scaled, y_model_r, lap_model_r, group_model_r, model_index, sequence_length
    )
    X_holdout_seq, y_holdout_seq, holdout_laps = build_sequences(
        X_holdout_scaled, y_holdout_r, lap_holdout_r, group_holdout_r, holdout_index, sequence_length
    )

    if len(X_model_seq) == 0 or len(X_holdout_seq) == 0:
        raise ValueError("Unable to build final LSTM sequences. Check sequence_length vs. lap continuity.")

    target_scaler = StandardScaler()
    y_model_scaled = target_scaler.fit_transform(y_model_raw.reshape(-1, 1)).ravel()

    final_epoch_count = max(int(final_epoch_count), int(lstm_cfg["lstm_min_final_epochs"]))
    print(f"  Final epoch count (median of Optuna trials): {final_epoch_count}")

    model = make_lstm_model(sequence_length, X_model_seq.shape[2], lstm_cfg)
    model.fit(
        X_model_seq, y_model_scaled,
        epochs=final_epoch_count,
        batch_size=int(lstm_cfg["lstm_batch_size"]),
        shuffle=False,
        verbose=0,
    )
    preds_scaled = model.predict(X_holdout_seq, verbose=0).ravel()
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    return preds, y_holdout_seq, holdout_laps, model, imputer, feature_scaler, target_scaler, feature_names, final_epoch_count


def metric_values(y_true, preds):
    residuals = np.asarray(y_true) - np.asarray(preds)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        "mae": float(mean_absolute_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
        "std": float(np.std(residuals, ddof=1)) if len(y_true) > 1 else 0.0,
    }


def build_split_indices(X_model_raw, lap_model_sorted, train_laps, val_laps):
    """Return context-relative integer indices for train and val targets."""
    context_mask = lap_model_sorted.isin(np.concatenate([train_laps, val_laps]))
    train_mask = lap_model_sorted.isin(train_laps)
    val_mask = lap_model_sorted.isin(val_laps)
    context_positions = pd.Series(
        np.arange(int(context_mask.sum())), index=X_model_raw.loc[context_mask].index
    )
    train_idx = pd.Index(context_positions.loc[X_model_raw.loc[train_mask].index])
    val_idx = pd.Index(context_positions.loc[X_model_raw.loc[val_mask].index])
    return context_mask, train_mask, val_mask, train_idx, val_idx


def tune_lstm_hyperparams(
    X_model_raw, y_model, lap_model_sorted, group_model,
    train_laps, val_laps,
    cat_cols, base_cfg, seed,
    params_path: Path | None = None,
    trials_path: Path | None = None,
):
    if not bool(base_cfg["lstm_tuning_enabled"]):
        print("LSTM Optuna tuning disabled; using YAML hyperparameters.")
        return base_cfg, int(base_cfg["lstm_epochs"]), None

    n_trials = int(base_cfg["lstm_optuna_trials"])

    if bool(base_cfg.get("use_saved_lstm_params", False)) and params_path is not None and params_path.exists():
        print(f"Found saved LSTM parameters: {params_path}")
        with params_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if (
            loaded.get("search_space_version") == LSTM_SEARCH_SPACE_VERSION
            and loaded.get("tuning_strategy") == LSTM_TUNING_STRATEGY
            and int(loaded.get("n_trials", 0)) == n_trials
        ):
            print("Using saved LSTM parameters (search space version, strategy, and n_trials match).")
            best_cfg = dict(base_cfg)
            best_cfg.update(loaded["best_params"])
            best_epoch_count = int(loaded.get("best_epoch_count", int(base_cfg["lstm_tuning_epochs"])))
            return best_cfg, best_epoch_count, loaded
        print(
            "Saved LSTM parameters do not match current search space version, strategy, or n_trials; "
            "running Optuna again."
        )

    context_mask, train_mask, val_mask, train_idx, val_idx = build_split_indices(
        X_model_raw, lap_model_sorted, train_laps, val_laps
    )

    print("\n--- LSTM Optuna tuning ---")
    print(
        f"Trials={n_trials} | "
        f"train_laps={len(train_laps)} | val_laps={len(val_laps)} | "
        f"tuning_epochs={int(base_cfg['lstm_tuning_epochs'])} | "
        f"tuning_patience={int(base_cfg['lstm_tuning_patience'])}"
    )
    print(f"Search space version: {LSTM_SEARCH_SPACE_VERSION} | Tuning strategy: {LSTM_TUNING_STRATEGY}")
    print("Objective: validation RMSE on the single sequential val split.")

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    trial_rows = []

    def objective(trial):
        trial_cfg = suggest_lstm_config(trial, base_cfg)
        try:
            preds, y_val_seq, _, *_, best_epoch = fit_predict_lstm(
                X_model_raw.loc[train_mask],
                X_model_raw.loc[context_mask],
                y_model.loc[context_mask],
                lap_model_sorted.loc[context_mask],
                group_model.loc[context_mask],
                train_idx, val_idx,
                cat_cols, trial_cfg,
                seed=seed,
            )
        except ValueError:
            tf.keras.backend.clear_session()
            return float("inf")
        rmse = float(np.sqrt(mean_squared_error(y_val_seq, preds)))
        trial.set_user_attr("best_epoch_count", best_epoch)
        trial.set_user_attr("rmse", rmse)
        tf.keras.backend.clear_session()
        return rmse

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    for t in study.trials:
        row = {
            "trial_number": t.number,
            "state": t.state.name,
            "rmse": t.user_attrs.get("rmse"),
            "best_epoch_count": t.user_attrs.get("best_epoch_count"),
        }
        row.update(t.params)
        trial_rows.append(row)

    best_cfg = dict(base_cfg)
    best_cfg.update(study.best_params)
    completed_epoch_counts = [
        int(t.user_attrs["best_epoch_count"])
        for t in study.trials
        if t.state.name == "COMPLETE" and "best_epoch_count" in t.user_attrs
    ]
    median_epoch_count = int(np.median(completed_epoch_counts)) if completed_epoch_counts else int(base_cfg["lstm_tuning_epochs"])
    print(f"Best LSTM Optuna RMSE: {study.best_value:.4f}")
    print(f"Best LSTM params: {study.best_params}")
    print(f"Median epoch count across {len(completed_epoch_counts)} completed trials: {median_epoch_count}")

    optuna_summary = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_epoch_count": median_epoch_count,
        "best_epoch_count_source": "median_completed_trials",
        "best_epoch_count_values": completed_epoch_counts,
        "n_trials": n_trials,
        "search_space_version": LSTM_SEARCH_SPACE_VERSION,
        "tuning_strategy": LSTM_TUNING_STRATEGY,
        "validation_strategy": "single_sequential_val_split",
    }

    if params_path is not None:
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with params_path.open("w", encoding="utf-8") as f:
            json.dump(optuna_summary, f, indent=2)
        print(f"Saved LSTM parameters to: {params_path}")
    if trials_path is not None and trial_rows:
        trials_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trial_rows).to_csv(trials_path, index=False)
        print(f"Saved LSTM Optuna trial table to: {trials_path}")

    return best_cfg, median_epoch_count, optuna_summary


def main():
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()
    lstm_cfg = lstm_config(config)

    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    num_cols, cat_cols = select_modeling_columns(df_base, config)
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    group_cols = [col for col in list(lstm_cfg["lstm_group_cols"]) if col in df_base.columns]

    print("--- LSTM: SINGLE SEQUENTIAL SPLIT + SEQUENTIAL HOLDOUT ---")
    print(f"Grand Prix: {target_gp_name}")
    print(f"Numerical features: {num_cols}")
    print(f"Categorical features: {cat_cols}")
    print(f"LSTM sequence groups: {group_cols if group_cols else 'none (flat sequences)'}")

    (
        lap_series, lap_min, lap_max,
        model_idx, holdout_idx,
        holdout_start_lap, model_end_lap, total_laps,
    ) = build_sequential_split(df_base, valid_indices, float(config["holdout_ratio"]), lap_col)

    model_laps = lap_series.loc[model_idx]
    model_order_idx = model_laps.sort_values(kind="mergesort").index
    holdout_laps = lap_series.loc[holdout_idx]
    holdout_order_idx = holdout_laps.sort_values(kind="mergesort").index

    X_model_raw = X_raw.loc[model_order_idx].reset_index(drop=True)
    y_model = y_raw.loc[model_order_idx].reset_index(drop=True)
    lap_model_sorted = model_laps.loc[model_order_idx].reset_index(drop=True)
    group_model = df_base.loc[model_order_idx, group_cols].reset_index(drop=True)

    X_holdout_raw = X_raw.loc[holdout_order_idx].reset_index(drop=True)
    y_holdout = y_raw.loc[holdout_order_idx].reset_index(drop=True)
    lap_holdout_sorted = holdout_laps.loc[holdout_order_idx].reset_index(drop=True)
    group_holdout = df_base.loc[holdout_order_idx, group_cols].reset_index(drop=True)

    unique_laps = np.sort(pd.to_numeric(lap_model_sorted, errors="coerce").dropna().unique())
    n_model_laps = len(unique_laps)

    # lstm_window_ratio controls the sequence lookback length.
    # Falls back to lstm_ew_window_ratio (legacy key) then window_ratio.
    lstm_window_ratio = float(
        config.get("lstm_window_ratio",
        config.get("lstm_ew_window_ratio",
        config["window_ratio"]))
    )
    if "Year" in df_base.columns:
        n_race_laps = len(
            pd.to_numeric(df_base.loc[model_order_idx, lap_col], errors="coerce").dropna().unique()
        )
    else:
        n_race_laps = n_model_laps
    sequence_length = max(1, int(np.ceil(n_race_laps * lstm_window_ratio)))
    lstm_cfg["lstm_sequence_length"] = sequence_length
    lstm_cfg["lstm_sequence_length_source"] = "lstm_window_ratio_times_race_laps"

    n_train_laps = max(sequence_length + 1, int(np.floor(n_model_laps * float(config["window_train_ratio"]))))
    train_laps = unique_laps[:n_train_laps]
    val_laps = unique_laps[n_train_laps:]
    if len(val_laps) == 0:
        raise ValueError("Validation split is empty. Reduce window_train_ratio or add more laps.")

    print("\n--- Sequential split ---")
    print(f"Total laps: {total_laps} (LapNumber {lap_min}-{lap_max})")
    print(f"Modeling block: laps {lap_min}-{model_end_lap} | records={len(X_model_raw)} | unique_laps={n_model_laps}")
    print(f"Holdout block:  laps {holdout_start_lap}-{lap_max} | records={len(X_holdout_raw)}")
    print(
        f"Train split: laps {int(train_laps[0])}-{int(train_laps[-1])} ({len(train_laps)} laps) | "
        f"Val split: laps {int(val_laps[0])}-{int(val_laps[-1])} ({len(val_laps)} laps)"
    )
    print(f"Sequence length (LSTM steps): {sequence_length} | lstm_window_ratio={lstm_window_ratio}")

    seed = int(config["random_seed"])
    lstm_params_path, lstm_trials_path = build_lstm_params_paths(repo_root, config, lstm_cfg)

    if bool(lstm_cfg["lstm_tuning_enabled"]):
        lstm_cfg, optuna_best_epoch, optuna_summary = tune_lstm_hyperparams(
            X_model_raw, y_model, lap_model_sorted, group_model,
            train_laps, val_laps, cat_cols, lstm_cfg, seed=seed,
            params_path=lstm_params_path,
            trials_path=lstm_trials_path,
        )
        lstm_cfg["lstm_sequence_length_source"] = "lstm_window_ratio_times_race_laps"
        # Ensure n_train_laps respects the fixed sequence_length
        tuned_seq_len = int(lstm_cfg["lstm_sequence_length"])
        if len(train_laps) <= tuned_seq_len:
            n_train_laps = max(tuned_seq_len + 1, int(np.floor(n_model_laps * float(config["window_train_ratio"]))))
            train_laps = unique_laps[:n_train_laps]
            val_laps = unique_laps[n_train_laps:]
            if len(val_laps) == 0:
                raise ValueError(
                    f"Validation split empty after sequence_length={tuned_seq_len} adjustment. "
                    "Reduce window_train_ratio or add more data."
                )
    else:
        optuna_best_epoch = int(lstm_cfg["lstm_epochs"])
        optuna_summary = None
        print("LSTM Optuna tuning disabled; using YAML hyperparameters.")

    final_epoch_count = max(optuna_best_epoch, int(lstm_cfg["lstm_min_final_epochs"]))
    print(
        f"\nSelected LSTM config: "
        f"sequence_length={lstm_cfg['lstm_sequence_length']} | units={lstm_cfg['lstm_units']} | "
        f"dense_units={lstm_cfg['lstm_dense_units']} | dropout={lstm_cfg['lstm_dropout']:.3f} | "
        f"recurrent_dropout={lstm_cfg['lstm_recurrent_dropout']:.3f} | "
        f"lr={lstm_cfg['lstm_learning_rate']:.5f} | batch={lstm_cfg['lstm_batch_size']} | "
        f"l2={lstm_cfg.get('lstm_l2_reg', 0.0):.5f} | stacked={lstm_cfg.get('lstm_stacked', False)} | "
        f"final_epochs={final_epoch_count}"
    )

    print("\n--- Validation split evaluation ---")
    context_mask, train_mask, val_mask, train_idx, val_idx = build_split_indices(
        X_model_raw, lap_model_sorted, train_laps, val_laps
    )
    preds_val, y_val_seq, val_seq_laps, *_, _ = fit_predict_lstm(
        X_model_raw.loc[train_mask],
        X_model_raw.loc[context_mask],
        y_model.loc[context_mask],
        lap_model_sorted.loc[context_mask],
        group_model.loc[context_mask],
        train_idx, val_idx,
        cat_cols, lstm_cfg, seed=seed,
    )
    val_metrics = metric_values(y_val_seq, preds_val)
    print(
        f"Val sequences: {len(y_val_seq)} | "
        f"RMSE={val_metrics['rmse']:.4f} | MAE={val_metrics['mae']:.4f} | R2={val_metrics['r2']:.4f}"
    )

    print("\n--- Training final LSTM model ---")
    (
        preds_holdout, y_holdout_seq, holdout_seq_laps,
        final_model, _, _, _, feature_names, final_epoch_count,
    ) = fit_final_lstm(
        X_model_raw, y_model, lap_model_sorted, group_model,
        X_holdout_raw, y_holdout, lap_holdout_sorted, group_holdout,
        cat_cols, lstm_cfg, seed=seed, final_epoch_count=final_epoch_count,
    )

    lstm_model_path, lstm_model_metadata_path = build_lstm_model_paths(repo_root, config, lstm_cfg)
    lstm_model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(lstm_model_path)

    holdout_ci = calc_holdout_ci(y_holdout_seq, preds_holdout, seed=seed)
    holdout_metrics = metric_values(y_holdout_seq, preds_holdout)

    results_for_cos = {
        "window": [1],
        "rmse": [val_metrics["rmse"]],
        "mae": [val_metrics["mae"]],
        "r2": [val_metrics["r2"]],
        "std": [val_metrics["std"]],
    }
    cos = summarize_cos(
        results_for_cos,
        val_metrics["mae"], val_metrics["rmse"],
        holdout_metrics["mae"], holdout_metrics["rmse"],
        val_metrics["std"], holdout_metrics["std"],
        float(config["alpha_cos"]), float(config["beta_cos"]),
    )

    lstm_model_metadata = {
        "target_gp_name": target_gp_name,
        "model": "lstm",
        "validation_protocol": "single_sequential_split",
        "tuning_strategy": LSTM_TUNING_STRATEGY,
        "search_space_version": LSTM_SEARCH_SPACE_VERSION,
        "model_path": str(lstm_model_path),
        "target_col": target_col,
        "lap_col": lap_col,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "encoded_feature_names": list(feature_names),
        "sequence_length": int(lstm_cfg["lstm_sequence_length"]),
        "sequence_length_source": lstm_cfg["lstm_sequence_length_source"],
        "lstm_window_ratio": lstm_window_ratio,
        "window_train_ratio": float(config["window_train_ratio"]),
        "modeling_lap_count": int(n_model_laps),
        "train_laps": len(train_laps),
        "val_laps": len(val_laps),
        "sequence_groups": group_cols,
        "training_block": "first_sequential_modeling_block",
        "holdout_usage": "final sequential holdout is not used for training, tuning, or early stopping",
        "preprocessing": "median_imputer_minmax_scaler_one_hot_full_rank",
        "lstm_config": lstm_cfg,
        "final_epoch_count": int(final_epoch_count),
        "val_metrics": val_metrics,
        "optuna_summary": optuna_summary,
    }
    lstm_model_metadata_path.write_text(json.dumps(lstm_model_metadata, indent=2), encoding="utf-8")
    print(f"Saved final LSTM model to: {lstm_model_path}")
    print(f"Saved metadata to: {lstm_model_metadata_path}")

    val_rmse_m, val_rmse_l, val_rmse_u = calc_stats([val_metrics["rmse"]])
    val_mae_m, val_mae_l, val_mae_u = calc_stats([val_metrics["mae"]])
    val_r2_m, val_r2_l, val_r2_u = calc_stats([val_metrics["r2"]])
    val_std_m, _, _ = calc_stats([val_metrics["std"]])

    split_info = {
        "total_laps": total_laps,
        "lap_min": lap_min,
        "lap_max": lap_max,
        "model_end_lap": model_end_lap,
        "holdout_start_lap": holdout_start_lap,
        "model_records": len(X_model_raw),
        "modeling_lap_count": int(n_model_laps),
        "train_laps": int(len(train_laps)),
        "val_laps": int(len(val_laps)),
        "val_sequences": int(len(y_val_seq)),
        "sequence_length": int(lstm_cfg["lstm_sequence_length"]),
        "holdout_records": len(X_holdout_raw),
        "holdout_sequences": int(len(y_holdout_seq)),
    }
    summary_metrics = {
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_r2": val_metrics["r2"],
        "val_residual_std": val_metrics["std"],
        "holdout_rmse": holdout_metrics["rmse"],
        "holdout_rmse_ci": holdout_ci["rmse"],
        "holdout_mae": holdout_metrics["mae"],
        "holdout_mae_ci": holdout_ci["mae"],
        "holdout_r2": holdout_metrics["r2"],
        "holdout_r2_ci": holdout_ci["r2"],
        "holdout_residual_std": holdout_metrics["std"],
        "cos_mae": cos["cos_mae"],
        "cos_mae_ci": cos["cos_mae_ci"],
        "cos_rmse": cos["cos_rmse"],
        "cos_rmse_ci": cos["cos_rmse_ci"],
    }
    log_mlflow_run(
        repo_root, config, "lstm", num_cols, cat_cols,
        split_info, results_for_cos, summary_metrics,
        extra_params={
            "preprocessing": "median_imputer_minmax_scaler_one_hot_full_rank",
            "validation_protocol": "single_sequential_split",
            "tuning_strategy": LSTM_TUNING_STRATEGY,
            "search_space_version": LSTM_SEARCH_SPACE_VERSION,
            "sequence_length": int(lstm_cfg["lstm_sequence_length"]),
            "sequence_length_source": lstm_cfg["lstm_sequence_length_source"],
            "lstm_window_ratio": lstm_window_ratio,
            "window_train_ratio": float(config["window_train_ratio"]),
            "sequence_groups": ", ".join(group_cols),
            "lstm_tuning_enabled": bool(lstm_cfg["lstm_tuning_enabled"]),
            "lstm_optuna_trials": int(lstm_cfg["lstm_optuna_trials"]),
            "lstm_units": int(lstm_cfg["lstm_units"]),
            "lstm_dense_units": int(lstm_cfg["lstm_dense_units"]),
            "lstm_dropout": float(lstm_cfg["lstm_dropout"]),
            "lstm_recurrent_dropout": float(lstm_cfg["lstm_recurrent_dropout"]),
            "lstm_learning_rate": float(lstm_cfg["lstm_learning_rate"]),
            "lstm_batch_size": int(lstm_cfg["lstm_batch_size"]),
            "lstm_epochs": int(lstm_cfg["lstm_epochs"]),
            "lstm_patience": int(lstm_cfg["lstm_patience"]),
            "lstm_l2_reg": float(lstm_cfg.get("lstm_l2_reg", 0.0)),
            "lstm_stacked": bool(lstm_cfg.get("lstm_stacked", False)),
            "lstm_final_epoch_count": int(final_epoch_count),
        },
        artifacts=[
            lstm_model_path,
            lstm_model_metadata_path,
            *(p for p in [lstm_params_path, lstm_trials_path] if p.exists()),
        ],
        validation_mode="single_split",
    )

    print("\n--- Validation split ---")
    print(f"Val sequences: {len(y_val_seq)} | LapNumber {int(np.min(val_seq_laps))}-{int(np.max(val_seq_laps))}")
    print(f"RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f} | R2: {val_metrics['r2']:.4f}")

    print("\n--- Sequential holdout ---")
    print(f"Holdout sequences: {len(y_holdout_seq)} | LapNumber {int(np.min(holdout_seq_laps))}-{int(np.max(holdout_seq_laps))}")
    print(f"RMSE: {holdout_metrics['rmse']:.4f} | 95% CI: [{holdout_ci['rmse'][0]:.4f}, {holdout_ci['rmse'][1]:.4f}]")
    print(f"MAE:  {holdout_metrics['mae']:.4f} | 95% CI: [{holdout_ci['mae'][0]:.4f}, {holdout_ci['mae'][1]:.4f}]")
    print(f"R2:   {holdout_metrics['r2']:.4f} | 95% CI: [{holdout_ci['r2'][0]:.4f}, {holdout_ci['r2'][1]:.4f}]")
    print(f"COS_MAE:  {cos['cos_mae']:.4f} | 95% CI: [{cos['cos_mae_ci'][0]:.4f}, {cos['cos_mae_ci'][1]:.4f}]")
    print(f"          MAE final/val={cos['mae_final']:.4f}/{cos['mae_sw']:.4f} | STD final/val={cos['std_final']:.4f}/{cos['std_sw']:.4f}")
    print(f"COS_RMSE: {cos['cos_rmse']:.4f} | 95% CI: [{cos['cos_rmse_ci'][0]:.4f}, {cos['cos_rmse_ci'][1]:.4f}]")
    print(f"          RMSE final/val={cos['rmse_final']:.4f}/{cos['rmse_sw']:.4f} | STD final/val={cos['std_final']:.4f}/{cos['std_sw']:.4f}")


if __name__ == "__main__":
    main()
