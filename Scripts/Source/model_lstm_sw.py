"""LSTM baseline with sliding-window validation and sequential holdout."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from modeling_utils import (
    align_one_hot,
    build_sequential_split,
    build_sliding_windows,
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
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard.
    raise ModuleNotFoundError(
        "TensorFlow is required for the LSTM model. Install project dependencies with "
        "`pip install -r Utils/requirements.txt` before running model_lstm_sw.py."
    ) from exc


DEFAULT_LSTM_CONFIG = {
    "lstm_units": 64,
    "lstm_dense_units": 32,
    "lstm_dropout": 0.2,
    "lstm_recurrent_dropout": 0.0,
    "lstm_learning_rate": 0.001,
    "lstm_batch_size": 32,
    "lstm_epochs": 100,
    "lstm_patience": 10,
    "lstm_reduce_lr_factor": 0.5,
    "lstm_reduce_lr_patience": 4,
    "lstm_min_learning_rate": 0.00001,
    "lstm_group_cols": [],
    "lstm_tuning_enabled": True,
    "lstm_optuna_trials": 15,
    "lstm_tuning_epochs": 40,
    "lstm_tuning_patience": 5,
    "lstm_min_final_epochs": 20,
    "lstm_models_subdir": "lstm/sw/models",
    "lstm_model_filename_template": "{safe_gp_name}_lstm_model_sw.keras",
    "lstm_model_metadata_filename_template": "{safe_gp_name}_lstm_model_sw_metadata.json",
}


def lstm_config(config: dict) -> dict:
    return {**DEFAULT_LSTM_CONFIG, **{key: value for key, value in config.items() if key.startswith("lstm_")}}


def build_lstm_model_paths(repo_root: Path, config: dict, lstm_cfg: dict) -> tuple[Path, Path]:
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    model_filename = str(lstm_cfg["lstm_model_filename_template"]).format(
        target_gp_name=target_gp_name,
        safe_gp_name=safe_name,
    )
    metadata_filename = str(lstm_cfg["lstm_model_metadata_filename_template"]).format(
        target_gp_name=target_gp_name,
        safe_gp_name=safe_name,
    )
    model_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / str(lstm_cfg["lstm_models_subdir"])
    return model_dir / model_filename, model_dir / metadata_filename


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def derive_lstm_sequence_length(laps: pd.Series, window_ratio: float, train_ratio: float) -> tuple[int, int, int, int]:
    unique_laps = np.sort(pd.to_numeric(laps, errors="coerce").dropna().unique())
    if len(unique_laps) < 2:
        raise ValueError("Insufficient modeling laps to derive the LSTM sequence length.")

    window_size = max(2, min(int(np.ceil(len(unique_laps) * window_ratio)), len(unique_laps)))
    train_size = max(1, int(np.floor(window_size * train_ratio)))
    if train_size >= window_size:
        train_size = window_size - 1
    sequence_length = train_size
    return sequence_length, len(unique_laps), window_size, train_size


def fit_feature_transformers(X_train: pd.DataFrame, X_context: pd.DataFrame, cat_cols: list[str]):
    X_train_enc, X_context_enc = align_one_hot(X_train, X_context, cat_cols, drop_first=False)

    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()

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
    """Create LSTM samples from the previous ``sequence_length`` temporal rows."""
    group_names = [f"__group_{index}__" for index in range(len(groups.columns))]
    sequence_groups = groups.reset_index(drop=True).copy()
    sequence_groups.columns = group_names
    sequence_frames = []
    targets = []
    target_laps = []
    target_index_set = set(target_indices)
    context = pd.concat(
        [
            X_scaled,
            y.rename("__target__"),
            laps.rename("__lap__"),
            sequence_groups,
        ],
        axis=1,
    )

    group_cols = group_names
    grouped = context.groupby(group_cols, sort=False, dropna=False) if group_cols else [(None, context)]
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


def count_lstm_sequences(
    laps: pd.Series,
    groups: pd.DataFrame,
    group_cols: list[str],
    target_indices: pd.Index,
    sequence_length: int,
) -> tuple[int, int]:
    sequence_groups = groups.reset_index(drop=True).copy()
    group_names = [f"__group_{index}__" for index in range(len(sequence_groups.columns))]
    sequence_groups.columns = group_names
    context = pd.concat([laps.reset_index(drop=True).rename("__lap__"), sequence_groups], axis=1)
    grouped = context.groupby(group_names, sort=False, dropna=False) if group_names else [(None, context)]
    target_index_set = set(target_indices)
    sequence_count = 0
    active_group_count = 0

    for _, group in grouped:
        group = group.sort_values("__lap__", kind="mergesort")
        group_sequence_count = 0
        for position, row_index in enumerate(group.index):
            if row_index in target_index_set and position >= sequence_length:
                group_sequence_count += 1
        if group_sequence_count:
            active_group_count += 1
            sequence_count += group_sequence_count

    return sequence_count, active_group_count


def make_lstm_model(sequence_length: int, n_features: int, lstm_cfg: dict):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, n_features)),
            tf.keras.layers.LSTM(
                int(lstm_cfg["lstm_units"]),
                dropout=float(lstm_cfg["lstm_dropout"]),
                recurrent_dropout=float(lstm_cfg["lstm_recurrent_dropout"]),
            ),
            tf.keras.layers.Dense(int(lstm_cfg["lstm_dense_units"]), activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(lstm_cfg["lstm_learning_rate"]))
    model.compile(optimizer=optimizer, loss="mse")
    return model


def suggest_lstm_config(trial, base_cfg: dict) -> dict:
    tuned_cfg = dict(base_cfg)
    tuned_cfg.update(
        {
            "lstm_units": trial.suggest_categorical("lstm_units", [16, 32, 64]),
            "lstm_dense_units": trial.suggest_categorical("lstm_dense_units", [8, 16, 32]),
            "lstm_dropout": trial.suggest_float("lstm_dropout", 0.0, 0.30),
            "lstm_learning_rate": trial.suggest_float("lstm_learning_rate", 5e-5, 1e-3, log=True),
            "lstm_batch_size": trial.suggest_categorical("lstm_batch_size", [16, 32]),
            "lstm_epochs": int(base_cfg["lstm_tuning_epochs"]),
            "lstm_patience": int(base_cfg["lstm_tuning_patience"]),
        }
    )
    return tuned_cfg


def training_callbacks(lstm_cfg: dict):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(lstm_cfg["lstm_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(lstm_cfg["lstm_reduce_lr_factor"]),
            patience=int(lstm_cfg["lstm_reduce_lr_patience"]),
            min_lr=float(lstm_cfg["lstm_min_learning_rate"]),
        ),
    ]


def fit_evaluate_lstm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lap_train: pd.Series,
    group_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    lap_eval: pd.Series,
    group_eval: pd.DataFrame,
    cat_cols: list[str],
    lstm_cfg: dict,
    seed: int,
    epochs: int | None = None,
    callbacks_enabled: bool = True,
):
    set_random_seed(seed)
    sequence_length = int(lstm_cfg["lstm_sequence_length"])

    X_context = pd.concat([X_train, X_eval], axis=0).reset_index(drop=True)
    y_context = pd.concat([y_train, y_eval], axis=0).reset_index(drop=True)
    lap_context = pd.concat([lap_train, lap_eval], axis=0).reset_index(drop=True)
    group_context = pd.concat([group_train, group_eval], axis=0).reset_index(drop=True)
    train_index = pd.RangeIndex(0, len(X_train))
    eval_index = pd.RangeIndex(len(X_train), len(X_context))

    X_context_scaled, imputer, feature_scaler, feature_names = fit_feature_transformers(
        X_train.reset_index(drop=True),
        X_context,
        cat_cols,
    )
    X_train_seq, y_train_seq_raw, _ = build_sequences(
        X_context_scaled,
        y_context,
        lap_context,
        group_context,
        train_index,
        sequence_length,
    )
    X_eval_seq, y_eval_seq, eval_laps = build_sequences(
        X_context_scaled,
        y_context,
        lap_context,
        group_context,
        eval_index,
        sequence_length,
    )

    if len(X_train_seq) == 0 or len(X_eval_seq) == 0:
        raise ValueError(
            "Unable to build LSTM sequences. Reduce lstm_sequence_length or check per-driver lap continuity."
        )

    target_scaler = MinMaxScaler()
    y_train_seq = target_scaler.fit_transform(y_train_seq_raw.reshape(-1, 1)).ravel()
    y_eval_seq_scaled = target_scaler.transform(y_eval_seq.reshape(-1, 1)).ravel()

    model = make_lstm_model(sequence_length, X_train_seq.shape[2], lstm_cfg)
    fit_kwargs = {
        "epochs": int(epochs or lstm_cfg["lstm_epochs"]),
        "batch_size": int(lstm_cfg["lstm_batch_size"]),
        "shuffle": False,
        "verbose": 0,
        "validation_data": (X_eval_seq, y_eval_seq_scaled),
    }
    if callbacks_enabled:
        fit_kwargs["callbacks"] = training_callbacks(lstm_cfg)
    history = model.fit(X_train_seq, y_train_seq, **fit_kwargs)

    preds_scaled = model.predict(X_eval_seq, verbose=0).ravel()
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    if "val_loss" in history.history and history.history["val_loss"]:
        best_epoch_count = int(np.argmin(history.history["val_loss"]) + 1)
    else:
        best_epoch_count = int(len(history.history.get("loss", [])) or fit_kwargs["epochs"])
    return (
        preds,
        y_eval_seq,
        eval_laps,
        model,
        imputer,
        feature_scaler,
        target_scaler,
        feature_names,
        best_epoch_count,
    )


def fit_predict_lstm_from_context(
    X_fit: pd.DataFrame,
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
        X_fit.reset_index(drop=True),
        X_context.reset_index(drop=True),
        cat_cols,
    )
    y_context = y_context.reset_index(drop=True)
    lap_context = lap_context.reset_index(drop=True)
    group_context = group_context.reset_index(drop=True)
    X_train_seq, y_train_seq_raw, train_seq_laps = build_sequences(
        X_context_scaled,
        y_context,
        lap_context,
        group_context,
        train_target_indices,
        sequence_length,
    )
    X_eval_seq, y_eval_seq, eval_laps = build_sequences(
        X_context_scaled,
        y_context,
        lap_context,
        group_context,
        eval_target_indices,
        sequence_length,
    )

    if len(X_train_seq) == 0 or len(X_eval_seq) == 0:
        raise ValueError("Unable to build LSTM sequences for this sliding window.")

    target_scaler = MinMaxScaler()
    y_train_seq = target_scaler.fit_transform(y_train_seq_raw.reshape(-1, 1)).ravel()
    y_eval_seq_scaled = target_scaler.transform(y_eval_seq.reshape(-1, 1)).ravel()

    model = make_lstm_model(sequence_length, X_train_seq.shape[2], lstm_cfg)
    fit_kwargs = {
        "epochs": int(epochs or lstm_cfg["lstm_epochs"]),
        "batch_size": int(lstm_cfg["lstm_batch_size"]),
        "shuffle": False,
        "verbose": 0,
        "validation_data": (X_eval_seq, y_eval_seq_scaled),
    }
    if callbacks_enabled:
        fit_kwargs["callbacks"] = training_callbacks(lstm_cfg)
    history = model.fit(X_train_seq, y_train_seq, **fit_kwargs)

    preds_scaled = model.predict(X_eval_seq, verbose=0).ravel()
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    if "val_loss" in history.history and history.history["val_loss"]:
        best_epoch_count = int(np.argmin(history.history["val_loss"]) + 1)
    else:
        best_epoch_count = int(len(history.history.get("loss", [])) or fit_kwargs["epochs"])
    return preds, y_eval_seq, eval_laps, model, imputer, feature_scaler, target_scaler, feature_names, best_epoch_count


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

    X_context = pd.concat([X_model, X_holdout], axis=0).reset_index(drop=True)
    y_context = pd.concat([y_model, y_holdout], axis=0).reset_index(drop=True)
    lap_context = pd.concat([lap_model, lap_holdout], axis=0).reset_index(drop=True)
    group_context = pd.concat([group_model, group_holdout], axis=0).reset_index(drop=True)
    model_index = pd.RangeIndex(0, len(X_model))
    holdout_index = pd.RangeIndex(len(X_model), len(X_context))

    X_context_scaled, imputer, feature_scaler, feature_names = fit_feature_transformers(
        X_model.reset_index(drop=True),
        X_context,
        cat_cols,
    )
    X_model_seq, y_model_seq_raw, model_seq_laps = build_sequences(
        X_context_scaled,
        y_context,
        lap_context,
        group_context,
        model_index,
        sequence_length,
    )
    X_holdout_seq, y_holdout_seq, holdout_laps = build_sequences(
        X_context_scaled,
        y_context,
        lap_context,
        group_context,
        holdout_index,
        sequence_length,
    )

    if len(X_model_seq) == 0 or len(X_holdout_seq) == 0:
        raise ValueError(
            "Unable to build final LSTM sequences. Reduce lstm_sequence_length or check per-driver lap continuity."
        )

    final_epoch_count = max(int(final_epoch_count), int(lstm_cfg["lstm_min_final_epochs"]))

    target_scaler = MinMaxScaler()
    y_model_seq = target_scaler.fit_transform(y_model_seq_raw.reshape(-1, 1)).ravel()

    model = make_lstm_model(sequence_length, X_model_seq.shape[2], lstm_cfg)
    callbacks = []
    model.fit(
        X_model_seq,
        y_model_seq,
        epochs=final_epoch_count,
        batch_size=int(lstm_cfg["lstm_batch_size"]),
        shuffle=False,
        callbacks=callbacks,
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


def tune_lstm_hyperparams(
    windows,
    unique_laps,
    lap_model_sorted,
    X_model_raw,
    y_model,
    group_model,
    cat_cols,
    base_cfg,
    seed: int,
):
    if not bool(base_cfg["lstm_tuning_enabled"]):
        print("LSTM Optuna tuning disabled; using YAML hyperparameters.")
        return base_cfg, None

    print("\n--- LSTM Optuna tuning ---")
    print(
        f"Trials={int(base_cfg['lstm_optuna_trials'])} | "
        f"sliding_windows={len(windows)} | "
        f"tuning_epochs={int(base_cfg['lstm_tuning_epochs'])} | "
        f"tuning_patience={int(base_cfg['lstm_tuning_patience'])}"
    )
    print("Objective: mean validation RMSE across feasible sliding windows inside the modeling block only.")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        trial_cfg = suggest_lstm_config(trial, base_cfg)
        trial_rmse = []
        try:
            for window_id, (start, split, end) in enumerate(windows, start=1):
                train_laps = unique_laps[start:split]
                val_laps = unique_laps[split:end]
                context_laps = unique_laps[:end]
                train_mask = lap_model_sorted.isin(train_laps)
                val_mask = lap_model_sorted.isin(val_laps)
                context_mask = lap_model_sorted.isin(context_laps)
                context_positions = pd.Series(np.arange(int(context_mask.sum())), index=X_model_raw.loc[context_mask].index)
                train_target_indices = pd.Index(context_positions.loc[X_model_raw.loc[train_mask].index])
                val_target_indices = pd.Index(context_positions.loc[X_model_raw.loc[val_mask].index])

                try:
                    preds, y_val_seq, _, *_ = fit_predict_lstm_from_context(
                        X_model_raw.loc[train_mask],
                        X_model_raw.loc[context_mask],
                        y_model.loc[context_mask],
                        lap_model_sorted.loc[context_mask],
                        group_model.loc[context_mask],
                        train_target_indices,
                        val_target_indices,
                        cat_cols,
                        trial_cfg,
                        seed=seed + trial.number + window_id,
                    )
                except ValueError:
                    tf.keras.backend.clear_session()
                    continue

                trial_rmse.append(float(np.sqrt(mean_squared_error(y_val_seq, preds))))
                tf.keras.backend.clear_session()
        except ValueError:
            tf.keras.backend.clear_session()
            return float("inf")

        if not trial_rmse:
            return float("inf")
        return float(np.mean(trial_rmse))

    study.optimize(objective, n_trials=int(base_cfg["lstm_optuna_trials"]), show_progress_bar=False)
    best_cfg = dict(base_cfg)
    best_cfg.update(study.best_params)
    print(f"Best LSTM Optuna RMSE: {study.best_value:.4f}")
    print(f"Best LSTM params: {study.best_params}")
    return best_cfg, {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": int(base_cfg["lstm_optuna_trials"]),
        "validation_strategy": "mean_rmse_across_feasible_sliding_windows",
        "sliding_windows": len(windows),
    }


def main():
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()
    lstm_cfg = lstm_config(config)

    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    num_cols, cat_cols = select_modeling_columns(df_base, config)
    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    group_cols = [col for col in list(lstm_cfg["lstm_group_cols"]) if col in df_base.columns]

    print("--- LSTM: SLIDING WINDOW + SEQUENTIAL HOLDOUT ---")
    print(f"Grand Prix: {target_gp_name}")
    print(f"Numerical features: {num_cols}")
    print(f"Categorical features: {cat_cols}")
    print(f"LSTM sequence groups: {group_cols if group_cols else 'none'}")

    (
        lap_series,
        lap_min,
        lap_max,
        model_idx,
        holdout_idx,
        holdout_start_lap,
        model_end_lap,
        total_laps,
    ) = build_sequential_split(df_base, valid_indices, float(config["holdout_ratio"]), lap_col)

    model_laps = lap_series.loc[model_idx]
    model_order_idx = model_laps.sort_values(kind="mergesort").index
    holdout_laps = lap_series.loc[holdout_idx]
    holdout_order_idx = holdout_laps.sort_values(kind="mergesort").index

    X_model_raw = X_raw.loc[model_order_idx].reset_index(drop=True)
    y_model = y_raw.loc[model_order_idx].reset_index(drop=True)
    lap_model_sorted = model_laps.loc[model_order_idx].reset_index(drop=True)
    group_model = df_base.loc[model_order_idx, group_cols].reset_index(drop=True)

    sequence_length, modeling_lap_count, derived_window_size, derived_train_size = derive_lstm_sequence_length(
        lap_model_sorted,
        float(config["window_ratio"]),
        float(config["window_train_ratio"]),
    )
    lstm_cfg["lstm_sequence_length"] = sequence_length
    lstm_cfg["lstm_sequence_length_source"] = "derived_from_sliding_window_train_size"

    X_holdout_raw = X_raw.loc[holdout_order_idx].reset_index(drop=True)
    y_holdout = y_raw.loc[holdout_order_idx].reset_index(drop=True)
    lap_holdout_sorted = holdout_laps.loc[holdout_order_idx].reset_index(drop=True)
    group_holdout = df_base.loc[holdout_order_idx, group_cols].reset_index(drop=True)

    unique_laps = np.sort(pd.to_numeric(lap_model_sorted, errors="coerce").dropna().unique())
    windows, window_size, train_size, val_size, step_size = build_sliding_windows(
        len(unique_laps),
        float(config["window_ratio"]),
        float(config["window_train_ratio"]),
        float(config["window_step_ratio"]),
    )
    if train_size != derived_train_size or window_size != derived_window_size:
        raise ValueError("Inconsistent LSTM sequence derivation and sliding-window configuration.")

    print(
        "Config: "
        f"holdout={config['holdout_ratio']} | "
        f"window_ratio={config['window_ratio']} | "
        f"window_train={config['window_train_ratio']} | step={config['window_step_ratio']} | "
        f"modeling_laps={modeling_lap_count} | "
        f"window_size={window_size} | train/val={train_size}/{val_size} | "
        f"sequence_length/train_size={lstm_cfg['lstm_sequence_length']} | "
        f"units={lstm_cfg['lstm_units']}"
    )

    lstm_cfg, optuna_summary = tune_lstm_hyperparams(
        windows,
        unique_laps,
        lap_model_sorted,
        X_model_raw,
        y_model,
        group_model,
        cat_cols,
        lstm_cfg,
        seed=int(config["random_seed"]),
    )
    print(
        "Selected LSTM config: "
        f"sequence_length/train_size={lstm_cfg['lstm_sequence_length']} | units={lstm_cfg['lstm_units']} | "
        f"dense_units={lstm_cfg['lstm_dense_units']} | dropout={lstm_cfg['lstm_dropout']} | "
        f"learning_rate={lstm_cfg['lstm_learning_rate']} | batch_size={lstm_cfg['lstm_batch_size']}"
    )

    print("\n--- Sequential split ---")
    print(f"Total laps: {total_laps} (LapNumber {lap_min}-{lap_max})")
    print(f"Modeling block: laps {lap_min}-{model_end_lap} | records={len(X_model_raw)}")
    print(f"Holdout block: laps {holdout_start_lap}-{lap_max} | records={len(X_holdout_raw)}")
    print(f"Sliding windows: {len(windows)} | window={window_size} | train/val={train_size}/{val_size} | step={step_size}")

    results = {"window": [], "rmse": [], "mae": [], "r2": [], "std": [], "eval_sequences": [], "train_sequences": []}
    epoch_counts = []

    print("\n--- Sliding-window validation ---")
    for i, (start, split, end) in enumerate(windows, start=1):
        train_laps = unique_laps[start:split]
        val_laps = unique_laps[split:end]
        context_laps = unique_laps[:end]
        train_mask = lap_model_sorted.isin(train_laps)
        val_mask = lap_model_sorted.isin(val_laps)
        context_mask = lap_model_sorted.isin(context_laps)
        context_positions = pd.Series(np.arange(int(context_mask.sum())), index=X_model_raw.loc[context_mask].index)
        train_target_indices = pd.Index(context_positions.loc[X_model_raw.loc[train_mask].index])
        val_target_indices = pd.Index(context_positions.loc[X_model_raw.loc[val_mask].index])
        train_sequence_count, _ = count_lstm_sequences(
            lap_model_sorted.loc[context_mask],
            group_model.loc[context_mask],
            group_cols,
            train_target_indices,
            int(lstm_cfg["lstm_sequence_length"]),
        )

        try:
            preds, y_val_seq, _, _, _, _, _, _, best_epoch_count = fit_predict_lstm_from_context(
                X_model_raw.loc[train_mask],
                X_model_raw.loc[context_mask],
                y_model.loc[context_mask],
                lap_model_sorted.loc[context_mask],
                group_model.loc[context_mask],
                train_target_indices,
                val_target_indices,
                cat_cols,
                lstm_cfg,
                seed=int(config["random_seed"]) + i,
            )
        except ValueError:
            print(
                f"Window {i:02d} skipped | train laps {int(train_laps[0])}-{int(train_laps[-1])} | "
                f"val laps {int(val_laps[0])}-{int(val_laps[-1])} | "
                "insufficient grouped sequence history"
            )
            continue

        metrics = metric_values(y_val_seq, preds)
        for key in ("rmse", "mae", "r2", "std"):
            results[key].append(metrics[key])
        results["window"].append(i)
        results["eval_sequences"].append(int(len(y_val_seq)))
        results["train_sequences"].append(int(train_sequence_count))
        epoch_counts.append(int(best_epoch_count))
        print(
            f"Window {i:02d} | train laps {int(train_laps[0])}-{int(train_laps[-1])} | "
            f"val laps {int(val_laps[0])}-{int(val_laps[-1])} | "
            f"train_seq={train_sequence_count} | val_seq={len(y_val_seq)} | "
            f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f}"
        )

    if not results["rmse"]:
        raise ValueError("No feasible LSTM sliding windows were generated. Reduce window_train_ratio or sequence length.")

    rmse_m, rmse_l, rmse_u = calc_stats(results["rmse"])
    mae_m, mae_l, mae_u = calc_stats(results["mae"])
    r2_m, r2_l, r2_u = calc_stats(results["r2"])
    std_m, _, _ = calc_stats(results["std"])
    final_epoch_count = max(
        int(round(float(np.median(epoch_counts)))) if epoch_counts else int(lstm_cfg["lstm_epochs"]),
        int(lstm_cfg["lstm_min_final_epochs"]),
    )
    print(f"Final epoch count calibrated from feasible sliding windows: {final_epoch_count}")

    (
        preds_holdout,
        y_holdout_seq,
        holdout_seq_laps,
        final_model,
        _,
        _,
        _,
        feature_names,
        final_epoch_count,
    ) = fit_final_lstm(
        X_model_raw,
        y_model,
        lap_model_sorted,
        group_model,
        X_holdout_raw,
        y_holdout,
        lap_holdout_sorted,
        group_holdout,
        cat_cols,
        lstm_cfg,
        seed=int(config["random_seed"]),
        final_epoch_count=final_epoch_count,
    )

    lstm_model_path, lstm_model_metadata_path = build_lstm_model_paths(repo_root, config, lstm_cfg)
    lstm_model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(lstm_model_path)
    lstm_model_metadata = {
        "target_gp_name": target_gp_name,
        "model": "lstm",
        "model_path": str(lstm_model_path),
        "target_col": target_col,
        "lap_col": lap_col,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "encoded_feature_names": list(feature_names),
        "sequence_length": int(lstm_cfg["lstm_sequence_length"]),
        "sequence_length_source": lstm_cfg["lstm_sequence_length_source"],
        "sequence_window_ratio": float(config["window_ratio"]),
        "sequence_window_train_ratio": float(config["window_train_ratio"]),
        "modeling_lap_count": int(modeling_lap_count),
        "sequence_groups": group_cols,
        "training_block": "first_sequential_modeling_block",
        "sliding_window_validation": "feasible sliding windows inside the first sequential modeling block",
        "holdout_usage": "final sequential holdout is not used for training, tuning, or early stopping",
        "preprocessing": "median_imputer_minmax_scaler_one_hot_full_rank",
        "lstm_config": lstm_cfg,
        "final_epoch_count": int(final_epoch_count),
        "sliding_window_metrics": {
            "rmse_mean": rmse_m,
            "rmse_ci": (rmse_l, rmse_u),
            "mae_mean": mae_m,
            "mae_ci": (mae_l, mae_u),
            "r2_mean": r2_m,
            "r2_ci": (r2_l, r2_u),
            "residual_std_mean": std_m,
        },
        "optuna_summary": optuna_summary,
    }
    lstm_model_metadata_path.write_text(json.dumps(lstm_model_metadata, indent=2), encoding="utf-8")
    print(f"Saved final LSTM model to: {lstm_model_path}")
    print(f"Saved final LSTM model metadata to: {lstm_model_metadata_path}")
    print(f"Final LSTM epoch count calibrated inside modeling block: {int(final_epoch_count)}")

    holdout_ci = calc_holdout_ci(y_holdout_seq, preds_holdout, seed=int(config["random_seed"]))
    holdout_metrics = metric_values(y_holdout_seq, preds_holdout)
    cos = summarize_cos(
        results,
        mae_m,
        rmse_m,
        holdout_metrics["mae"],
        holdout_metrics["rmse"],
        std_m,
        holdout_metrics["std"],
        float(config["alpha_cos"]),
        float(config["beta_cos"]),
    )

    split_info = {
        "total_laps": total_laps,
        "lap_min": lap_min,
        "lap_max": lap_max,
        "model_end_lap": model_end_lap,
        "holdout_start_lap": holdout_start_lap,
        "model_records": len(X_model_raw),
        "modeling_lap_count": int(modeling_lap_count),
        "sequence_length": int(lstm_cfg["lstm_sequence_length"]),
        "sequence_window_ratio": float(config["window_ratio"]),
        "sequence_window_train_ratio": float(config["window_train_ratio"]),
        "holdout_records": len(X_holdout_raw),
        "holdout_sequences": len(y_holdout_seq),
        "sliding_windows": len(windows),
        "feasible_sliding_windows": len(results["window"]),
        "window_size": window_size,
        "window_train_size": train_size,
        "window_validation_size": val_size,
        "window_step_size": step_size,
    }
    summary_metrics = {
        "sw_rmse_mean": rmse_m,
        "sw_rmse_ci": (rmse_l, rmse_u),
        "sw_mae_mean": mae_m,
        "sw_mae_ci": (mae_l, mae_u),
        "sw_r2_mean": r2_m,
        "sw_r2_ci": (r2_l, r2_u),
        "sw_residual_std_mean": std_m,
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
        repo_root,
        config,
        "lstm",
        num_cols,
        cat_cols,
        split_info,
        results,
        summary_metrics,
        extra_params={
            "preprocessing": "median_imputer_minmax_scaler_one_hot_full_rank",
            "sequence_length": int(lstm_cfg["lstm_sequence_length"]),
            "sequence_length_source": lstm_cfg["lstm_sequence_length_source"],
            "sequence_window_ratio": float(config["window_ratio"]),
            "sequence_window_train_ratio": float(config["window_train_ratio"]),
            "sequence_groups": ", ".join(group_cols),
            "lstm_tuning_enabled": bool(lstm_cfg["lstm_tuning_enabled"]),
            "lstm_optuna_trials": int(lstm_cfg["lstm_optuna_trials"]),
            "lstm_validation_strategy": "feasible_sliding_windows_inside_modeling_block",
            "lstm_min_final_epochs": int(lstm_cfg["lstm_min_final_epochs"]),
            "lstm_units": int(lstm_cfg["lstm_units"]),
            "lstm_dense_units": int(lstm_cfg["lstm_dense_units"]),
            "lstm_dropout": float(lstm_cfg["lstm_dropout"]),
            "lstm_recurrent_dropout": float(lstm_cfg["lstm_recurrent_dropout"]),
            "lstm_learning_rate": float(lstm_cfg["lstm_learning_rate"]),
            "lstm_batch_size": int(lstm_cfg["lstm_batch_size"]),
            "lstm_epochs": int(lstm_cfg["lstm_epochs"]),
            "lstm_patience": int(lstm_cfg["lstm_patience"]),
            "lstm_reduce_lr_factor": float(lstm_cfg["lstm_reduce_lr_factor"]),
            "lstm_reduce_lr_patience": int(lstm_cfg["lstm_reduce_lr_patience"]),
            "lstm_min_learning_rate": float(lstm_cfg["lstm_min_learning_rate"]),
            "lstm_final_epoch_count": int(final_epoch_count),
        },
        artifacts=[lstm_model_path, lstm_model_metadata_path],
    )

    print("\n--- Sliding-window summary (indicative CI) ---")
    print("NOTE: sliding windows overlap; these confidence intervals are descriptive.")
    print(f"Feasible windows: {len(results['window'])}/{len(windows)}")
    print(f"RMSE: {rmse_m:.4f} | 95% CI: [{rmse_l:.4f}, {rmse_u:.4f}]")
    print(f"MAE:  {mae_m:.4f} | 95% CI: [{mae_l:.4f}, {mae_u:.4f}]")
    print(f"R2:   {r2_m:.4f} | 95% CI: [{r2_l:.4f}, {r2_u:.4f}]")

    print("\n--- Sequential holdout ---")
    print(f"Evaluated holdout sequences: {len(y_holdout_seq)}")
    print(f"Holdout sequence LapNumber range: {int(np.min(holdout_seq_laps))}-{int(np.max(holdout_seq_laps))}")
    print(f"RMSE: {holdout_metrics['rmse']:.4f} | 95% CI: [{holdout_ci['rmse'][0]:.4f}, {holdout_ci['rmse'][1]:.4f}]")
    print(f"MAE:  {holdout_metrics['mae']:.4f} | 95% CI: [{holdout_ci['mae'][0]:.4f}, {holdout_ci['mae'][1]:.4f}]")
    print(f"R2:   {holdout_metrics['r2']:.4f} | 95% CI: [{holdout_ci['r2'][0]:.4f}, {holdout_ci['r2'][1]:.4f}]")
    print(f"COS_MAE:  {cos['cos_mae']:.4f} | 95% CI: [{cos['cos_mae_ci'][0]:.4f}, {cos['cos_mae_ci'][1]:.4f}]")
    print(f"          MAE final/SW={cos['mae_final']:.4f}/{cos['mae_sw']:.4f} | STD final/SW={cos['std_final']:.4f}/{cos['std_sw']:.4f}")
    print(f"COS_RMSE: {cos['cos_rmse']:.4f} | 95% CI: [{cos['cos_rmse_ci'][0]:.4f}, {cos['cos_rmse_ci'][1]:.4f}]")
    print(f"          RMSE final/SW={cos['rmse_final']:.4f}/{cos['rmse_sw']:.4f} | STD final/SW={cos['std_final']:.4f}/{cos['std_sw']:.4f}")


if __name__ == "__main__":
    main()
