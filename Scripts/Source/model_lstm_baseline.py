"""Baseline-LapTime_prev LSTM: previous-lap baseline + LSTM residual.

This is the "previous-lap baseline" LSTM (an extra model, tested alongside the core
experiment). The baseline is the driver's previous lap time: the network is trained on
the residual ``LapTime_seconds - LapTime_prev`` (``lstm_target_mode = residual_from_laptime_prev``)
and the final prediction is ``LapTime_prev + lstm_residual_prediction``. ``LapTime_prev``
is deliberately NOT a network input feature (the configured ``lstm_feature_mode`` is
``auxiliary_embedding``, which drops ``LapTime_prev`` and keeps Driver/Team embeddings);
the previous-lap signal enters only through the residual target, not as a feature.

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
        "`pip install -r Utils/requirements.txt` before running model_lstm_baseline.py."
    ) from exc


LSTM_SEARCH_SPACE_VERSION = "v11"
LSTM_TUNING_STRATEGY = "single_sequential_split_v1"

# Feature autorregressiva (alvo defasado em 1 volta). É removida nos modos
# auxiliary do LSTM porque a autorregressão já vem pela ordem temporal da sequência.
LSTM_AUTOREGRESSIVE_FEATURE = "LapTime_prev"

# Modos de alvo que treinam o LSTM sobre o resíduo (LapTime - baseline). O baseline pode
# ser o valor da volta anterior (residual_from_laptime_prev) ou as previsões out-of-sample
# de um modelo tabular (residual_from_tabular, usado pelo híbrido em model_lstm_hybrid.py).
# Em ambos a reconstrução é hybrid_prediction = baseline + resíduo previsto.
RESIDUAL_TARGET_MODES = {"residual_from_laptime_prev", "residual_from_tabular"}

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
    "lstm_optuna_trials": 40,
    "lstm_tuning_epochs": 40,
    "lstm_tuning_patience": 5,
    "lstm_min_final_epochs": 10,
    "lstm_final_val_ratio": 0.15,
    "lstm_l2_reg": 0.0,
    "lstm_stacked": False,
    "lstm_huber_delta": 1.0,
    "lstm_embedding_cols": ["Driver", "Team"],
    "lstm_embedding_max_dim": 8,
    "lstm_reduce_lr_factor": 0.5,
    "lstm_reduce_lr_patience": 4,
    "lstm_min_learning_rate": 1e-5,
    "lstm_feature_mode": "auxiliary",
    "lstm_include_target_row_features": True,
    "lstm_target_mode": "residual_from_laptime_prev",
    "lstm_models_subdir": "lstm/models",
    "lstm_model_filename_template": "{safe_gp_name}_lstm_model.keras",
    "lstm_model_metadata_filename_template": "{safe_gp_name}_lstm_model_metadata.json",
    "use_saved_lstm_params": False,
    "lstm_params_subdir": "lstm/params",
    "lstm_params_filename_template": "{safe_gp_name}_lstm_params.json",
    "lstm_trials_filename_template": "{safe_gp_name}_lstm_optuna_trials.csv",
}


def lstm_config(config: dict) -> dict:
    lstm_specific = {k: v for k, v in config.items() if k.startswith("lstm_")}
    if "use_saved_lstm_params" in config:
        lstm_specific["use_saved_lstm_params"] = config["use_saved_lstm_params"]
    return {**DEFAULT_LSTM_CONFIG, **lstm_specific}


def build_lstm_model_paths(repo_root: Path, config: dict, lstm_cfg: dict) -> tuple[Path, Path]:
    target_gp_name = str(config["target_gp_name"])
    safe_name = f"{safe_gp_name(target_gp_name)}_{lstm_cfg['lstm_feature_mode']}"
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
    safe_name = f"{safe_gp_name(target_gp_name)}_{lstm_cfg['lstm_feature_mode']}"
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


def _embedding_dim(cardinality: int, max_dim: int) -> int:
    """Heurística de dimensão do embedding: min(max_dim, (card + 1) // 2)."""
    return int(max(1, min(int(max_dim), (int(cardinality) + 1) // 2)))


def resolve_embedding_cols(lstm_cfg: dict, cat_cols: list[str]) -> tuple[list[str], int]:
    """Categorical columns sent through Embedding layers in embedding feature modes."""
    if str(lstm_cfg.get("lstm_feature_mode", "")).lower() not in ("auxiliary_embedding", "full_embedding"):
        return [], int(lstm_cfg.get("lstm_embedding_max_dim", 8))
    embed_cols = [c for c in list(lstm_cfg.get("lstm_embedding_cols", [])) if c in cat_cols]
    return embed_cols, int(lstm_cfg.get("lstm_embedding_max_dim", 8))


def fit_ordinal_encoders(X_train: pd.DataFrame, X_context: pd.DataFrame, embed_cols: list[str]):
    """Codifica categóricas como inteiros para camadas Embedding.

    Índice 0 fica reservado a desconhecido/missing; categorias vistas no treino
    começam em 1. O encoder é ajustado só no treino e aplicado ao contexto.
    """
    from sklearn.preprocessing import OrdinalEncoder

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_cat = X_train[embed_cols].fillna("Missing").astype(str)
    X_context_cat = X_context[embed_cols].fillna("Missing").astype(str)
    encoder.fit(X_train_cat)
    codes = encoder.transform(X_context_cat) + 1.0  # unknown (-1) -> 0
    codes_df = pd.DataFrame(codes, index=X_context.index, columns=embed_cols).astype(np.float32)
    spec = [
        {"name": col, "cardinality": int(len(cats)), "dim": None}
        for col, cats in zip(embed_cols, encoder.categories_)
    ]
    return codes_df, encoder, spec


def fit_feature_transformers(
    X_train: pd.DataFrame,
    X_context: pd.DataFrame,
    cat_cols: list[str],
    embed_cols: list[str] | None = None,
    max_dim: int = 8,
):
    """Ajusta transformadores e devolve o frame de contexto pronto.

    Sem embed_cols: caminho one-hot puro (comportamento original).
    Com embed_cols: numéricas escaladas + one-hot (não escalado) das demais
    categóricas + códigos inteiros das categóricas a embeddar, nesta ordem de
    colunas. feature_meta carrega embed_spec e dense_channels para a arquitetura.
    """
    embed_cols = [c for c in (embed_cols or []) if c in cat_cols]

    if not embed_cols:
        X_train_enc, X_context_enc = align_one_hot(X_train, X_context, cat_cols, drop_first=False)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_imp = imputer.fit_transform(X_train_enc)
        X_context_imp = imputer.transform(X_context_enc)
        scaler.fit(X_train_imp)
        X_context_scaled = scaler.transform(X_context_imp)
        feature_names = list(X_train_enc.columns)
        feature_meta = {"embed_spec": [], "dense_channels": len(feature_names)}
        return (
            pd.DataFrame(X_context_scaled, index=X_context.index, columns=feature_names),
            imputer,
            scaler,
            feature_names,
            feature_meta,
        )

    onehot_cols = [c for c in cat_cols if c not in embed_cols]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    if num_cols:
        X_train_num = imputer.fit_transform(X_train[num_cols])
        scaler.fit(X_train_num)
        X_context_num = scaler.transform(imputer.transform(X_context[num_cols]))
        num_df = pd.DataFrame(X_context_num, index=X_context.index, columns=num_cols)
    else:
        num_df = pd.DataFrame(index=X_context.index)

    if onehot_cols:
        oh_train, oh_context = align_one_hot(
            X_train[onehot_cols], X_context[onehot_cols], onehot_cols, drop_first=False
        )
        onehot_df = oh_context.astype(np.float32)
        onehot_names = list(oh_train.columns)
    else:
        onehot_df = pd.DataFrame(index=X_context.index)
        onehot_names = []

    codes_df, _encoder, spec = fit_ordinal_encoders(X_train, X_context, embed_cols)
    for s in spec:
        s["dim"] = _embedding_dim(s["cardinality"], max_dim)

    combined = pd.concat([num_df, onehot_df, codes_df], axis=1)
    feature_names = list(num_df.columns) + onehot_names + embed_cols
    feature_meta = {
        "embed_spec": spec,
        "dense_channels": len(num_df.columns) + len(onehot_names),
    }
    return combined, imputer, scaler, feature_names, feature_meta


def build_sequences(
    X_scaled: pd.DataFrame,
    y: pd.Series,
    laps: pd.Series,
    groups: pd.DataFrame,
    target_indices: pd.Index,
    sequence_length: int,
    include_target_row_features: bool = True,
):
    """Build (sequence_length × features) input tensors for each target index."""
    group_names = [f"__group_{i}__" for i in range(len(groups.columns))]
    sequence_groups = groups.reset_index(drop=True).copy()
    sequence_groups.columns = group_names
    context = pd.concat([X_scaled, y.rename("__target__"), laps.rename("__lap__"), sequence_groups], axis=1)
    target_index_set = set(target_indices)
    sequence_frames, targets, target_laps, target_groups, target_row_indices = [], [], [], [], []

    grouped = context.groupby(group_names, sort=False, dropna=False) if group_names else [(None, context)]
    for group_key, group in grouped:
        group = group.sort_values("__lap__", kind="mergesort")
        ordered_indices = list(group.index)
        for position, row_index in enumerate(ordered_indices):
            first_position = position - sequence_length + 1 if include_target_row_features else position - sequence_length
            last_position = position + 1 if include_target_row_features else position
            if row_index not in target_index_set or first_position < 0:
                continue
            sequence_indices = ordered_indices[first_position:last_position]
            sequence_frames.append(X_scaled.loc[sequence_indices].to_numpy(dtype=np.float32))
            targets.append(float(y.loc[row_index]))
            target_laps.append(float(laps.loc[row_index]))
            target_groups.append(group_key)
            target_row_indices.append(row_index)

    if not sequence_frames:
        n_features = X_scaled.shape[1]
        return (
            np.empty((0, sequence_length, n_features), dtype=np.float32),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            [],
            np.empty((0,), dtype=int),
        )
    return (
        np.stack(sequence_frames),
        np.asarray(targets, dtype=float),
        np.asarray(target_laps, dtype=float),
        target_groups,
        np.asarray(target_row_indices, dtype=int),
    )


def resolve_lstm_features(mode: str, num_cols: list[str], cat_cols: list[str], target_col: str):
    """Seleciona as features do LSTM conforme o experimento escolhido.

    - laptime_only: só a série temporal do próprio tempo de volta (univariado, AR).
    - auxiliary: demais variáveis (num + cat) como séries auxiliares, sem LapTime_prev.
    - auxiliary_numeric: igual a auxiliary, mas sem as categóricas.
    - auxiliary_embedding: igual a auxiliary; as categóricas de alta cardinalidade
      (lstm_embedding_cols) entram por camadas Embedding, o restante via one-hot.
      A separação embed/one-hot é resolvida em fit_feature_transformers.
    - full: todas as features, como nos modelos tabulares (referência).
    - full_embedding: igual a full, mas usando Embedding para lstm_embedding_cols.
    """
    if mode == "laptime_only":
        if LSTM_AUTOREGRESSIVE_FEATURE not in num_cols:
            raise ValueError(
                f"laptime_only requires {LSTM_AUTOREGRESSIVE_FEATURE!r} when target-row features "
                "are included, otherwise the target LapTime_seconds would leak into the sequence."
            )
        return [LSTM_AUTOREGRESSIVE_FEATURE], []
    if mode in ("auxiliary", "auxiliary_embedding"):
        return [c for c in num_cols if c != LSTM_AUTOREGRESSIVE_FEATURE], list(cat_cols)
    if mode == "auxiliary_numeric":
        return [c for c in num_cols if c != LSTM_AUTOREGRESSIVE_FEATURE], []
    if mode in ("full", "full_embedding"):
        return list(num_cols), list(cat_cols)
    raise ValueError(
        f"lstm_feature_mode inválido: {mode!r}. "
        "Use 'laptime_only', 'auxiliary', 'auxiliary_embedding', 'auxiliary_numeric', "
        "'full' ou 'full_embedding'."
    )


def _lstm_recurrent_head(x, lstm_cfg: dict, regularizer):
    """Aplica a pilha LSTM + BatchNorm + Dense -> Dense(1) sobre o tensor x."""
    stacked = bool(lstm_cfg.get("lstm_stacked", False))
    units = int(lstm_cfg["lstm_units"])
    dropout = float(lstm_cfg["lstm_dropout"])
    recurrent_dropout = float(lstm_cfg["lstm_recurrent_dropout"])

    x = tf.keras.layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        return_sequences=stacked,
    )(x)
    if stacked:
        x = tf.keras.layers.LSTM(
            max(16, units // 2),
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(
        int(lstm_cfg["lstm_dense_units"]),
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    return tf.keras.layers.Dense(1)(x)


def make_lstm_model(sequence_length: int, n_features: int, lstm_cfg: dict, feature_meta: dict | None = None):
    l2_reg = float(lstm_cfg.get("lstm_l2_reg", 0.0))
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    embed_spec = list((feature_meta or {}).get("embed_spec", []))
    inp = tf.keras.layers.Input(shape=(sequence_length, n_features))

    if embed_spec:
        # Últimas len(embed_spec) colunas são códigos inteiros; o início é o bloco denso.
        dense_channels = int(feature_meta["dense_channels"])
        blocks = []
        if dense_channels > 0:
            blocks.append(tf.keras.layers.Lambda(lambda t: t[:, :, :dense_channels])(inp))
        for i, s in enumerate(embed_spec):
            code = tf.keras.layers.Lambda(
                lambda t, k=i: tf.cast(t[:, :, dense_channels + k], tf.int32)
            )(inp)
            blocks.append(
                tf.keras.layers.Embedding(int(s["cardinality"]) + 1, int(s["dim"]))(code)
            )
        x = tf.keras.layers.Concatenate(axis=-1)(blocks) if len(blocks) > 1 else blocks[0]
        out = _lstm_recurrent_head(x, lstm_cfg, regularizer)
    else:
        out = _lstm_recurrent_head(inp, lstm_cfg, regularizer)

    model = tf.keras.Model(inputs=inp, outputs=out)
    # Huber é mais robusto que MSE a outliers grandes (pit/safety car/outlaps).
    # delta em unidades do alvo padronizado (~1 desvio); delta alto -> comportamento ~MSE.
    huber_delta = float(lstm_cfg.get("lstm_huber_delta", 1.0))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lstm_cfg["lstm_learning_rate"])),
        loss=tf.keras.losses.Huber(delta=huber_delta),
    )
    return model


def suggest_lstm_config(trial, base_cfg: dict) -> dict:
    # Search space v11:
    # - Refined around the best Bahrain hybrid trials from v10, which consistently
    #   favored a small LSTM head, moderate/high dropout, lr near 1e-3 and low L2.
    # - Adds Huber delta to let Optuna tune how aggressively large residual errors
    #   are down-weighted.
    # - lstm_sequence_length remains fixed by lstm_window_ratio in YAML.
    tuned = dict(base_cfg)
    tuned.update(
        {
            "lstm_units": trial.suggest_categorical("lstm_units", [8, 16, 24, 32]),
            "lstm_dense_units": trial.suggest_categorical("lstm_dense_units", [24, 32, 48]),
            "lstm_dropout": trial.suggest_float("lstm_dropout", 0.22, 0.42),
            "lstm_recurrent_dropout": trial.suggest_float("lstm_recurrent_dropout", 0.05, 0.18),
            "lstm_learning_rate": trial.suggest_float("lstm_learning_rate", 7e-4, 1.8e-3, log=True),
            "lstm_batch_size": trial.suggest_categorical("lstm_batch_size", [16, 32]),
            "lstm_l2_reg": trial.suggest_float("lstm_l2_reg", 4e-4, 1.5e-3),
            "lstm_huber_delta": trial.suggest_float("lstm_huber_delta", 0.5, 2.0),
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
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(lstm_cfg.get("lstm_reduce_lr_factor", 0.5)),
            patience=int(lstm_cfg.get("lstm_reduce_lr_patience", 4)),
            min_lr=float(lstm_cfg.get("lstm_min_learning_rate", 1e-5)),
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
    baseline_context: pd.Series | None = None,
):
    set_random_seed(seed)
    sequence_length = int(lstm_cfg["lstm_sequence_length"])
    include_target_row_features = bool(lstm_cfg.get("lstm_include_target_row_features", True))
    target_mode = str(lstm_cfg.get("lstm_target_mode", "absolute")).lower()
    embed_cols, max_dim = resolve_embedding_cols(lstm_cfg, cat_cols)

    X_train_r = X_train.reset_index(drop=True)
    X_context_r = X_context.reset_index(drop=True)
    X_context_scaled, imputer, feature_scaler, feature_names, feature_meta = fit_feature_transformers(
        X_train_r,
        X_context_r,
        cat_cols,
        embed_cols=embed_cols,
        max_dim=max_dim,
    )
    y_ctx = y_context.reset_index(drop=True)
    lap_ctx = lap_context.reset_index(drop=True)
    group_ctx = group_context.reset_index(drop=True)
    # Baseline for the residual target. An explicit series takes precedence (used by
    # auxiliary modes, where LapTime_prev is deliberately absent from the network
    # input); otherwise fall back to the autoregressive column carried inside X.
    if baseline_context is not None:
        baseline_ctx = pd.to_numeric(baseline_context.reset_index(drop=True), errors="coerce")
    elif target_mode == "residual_from_laptime_prev" and LSTM_AUTOREGRESSIVE_FEATURE in X_context_r.columns:
        baseline_ctx = pd.to_numeric(X_context_r[LSTM_AUTOREGRESSIVE_FEATURE], errors="coerce")
    else:
        baseline_ctx = None
    if target_mode in RESIDUAL_TARGET_MODES:
        if baseline_ctx is None:
            raise ValueError(
                f"{target_mode} requires an explicit baseline_context series"
                + (
                    f" or {LSTM_AUTOREGRESSIVE_FEATURE!r} as an input feature."
                    if target_mode == "residual_from_laptime_prev"
                    else "."
                )
            )
        y_model_target = y_ctx - baseline_ctx
    elif target_mode == "absolute":
        y_model_target = y_ctx
    else:
        raise ValueError(
            "Unsupported lstm_target_mode. Use 'absolute', 'residual_from_laptime_prev', "
            "or 'residual_from_tabular'."
        )

    X_train_seq, y_train_raw, _, _, _ = build_sequences(
        X_context_scaled,
        y_model_target,
        lap_ctx,
        group_ctx,
        train_target_indices,
        sequence_length,
        include_target_row_features=include_target_row_features,
    )
    X_eval_seq, y_eval_target, eval_laps, _, eval_row_indices = build_sequences(
        X_context_scaled,
        y_model_target,
        lap_ctx,
        group_ctx,
        eval_target_indices,
        sequence_length,
        include_target_row_features=include_target_row_features,
    )

    if len(X_train_seq) == 0 or len(X_eval_seq) == 0:
        raise ValueError("Unable to build LSTM sequences. Check sequence_length vs. lap continuity.")

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_eval_scaled = target_scaler.transform(y_eval_target.reshape(-1, 1)).ravel()

    model = make_lstm_model(sequence_length, X_train_seq.shape[2], lstm_cfg, feature_meta)
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
    preds_target = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    y_eval_seq = y_ctx.loc[eval_row_indices].to_numpy(dtype=float)
    if target_mode in RESIDUAL_TARGET_MODES:
        preds = preds_target + baseline_ctx.loc[eval_row_indices].to_numpy(dtype=float)
    else:
        preds = preds_target
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
    baseline_model_series: pd.Series | None = None,
    baseline_holdout_series: pd.Series | None = None,
):
    set_random_seed(seed)
    sequence_length = int(lstm_cfg["lstm_sequence_length"])
    include_target_row_features = bool(lstm_cfg.get("lstm_include_target_row_features", True))
    target_mode = str(lstm_cfg.get("lstm_target_mode", "absolute")).lower()
    embed_cols, max_dim = resolve_embedding_cols(lstm_cfg, cat_cols)

    X_model_r = X_model.reset_index(drop=True)
    y_model_r = y_model.reset_index(drop=True)
    lap_model_r = lap_model.reset_index(drop=True)
    group_model_r = group_model.reset_index(drop=True)

    X_holdout_r = X_holdout.reset_index(drop=True)
    y_holdout_r = y_holdout.reset_index(drop=True)
    lap_holdout_r = lap_holdout.reset_index(drop=True)
    group_holdout_r = group_holdout.reset_index(drop=True)

    # Reserve the modeling tail only to calibrate the final epoch count. After
    # calibration, a fresh model is trained on the full modeling block so the
    # final estimator uses all non-holdout data.
    final_val_ratio = float(lstm_cfg.get("lstm_final_val_ratio", 0.15))
    unique_laps = np.sort(pd.to_numeric(lap_model_r, errors="coerce").dropna().unique())
    n_model_laps = len(unique_laps)
    n_val_laps = int(np.floor(n_model_laps * final_val_ratio))
    use_early_stopping = (
        final_val_ratio > 0.0
        and n_val_laps > sequence_length
        and (n_model_laps - n_val_laps) > sequence_length
    )

    calibration_train_laps = set(
        (unique_laps[: n_model_laps - n_val_laps] if use_early_stopping else unique_laps).tolist()
    )
    calibration_train_mask = lap_model_r.isin(calibration_train_laps)

    # Continuous series = modeling block + holdout, used only as context. Holdout
    # rows are forecast targets; their lookback may reach the modeling tail, which
    # mirrors the validation split. The holdout is never used as a training target
    # or for transformer fitting.
    n_model = len(X_model_r)
    X_full_r = pd.concat([X_model_r, X_holdout_r], ignore_index=True)
    y_full_r = pd.concat([y_model_r, y_holdout_r], ignore_index=True)
    lap_full_r = pd.concat([lap_model_r, lap_holdout_r], ignore_index=True)
    group_full_r = pd.concat([group_model_r, group_holdout_r], ignore_index=True)

    if target_mode in RESIDUAL_TARGET_MODES:
        # Explicit baseline series take precedence (auxiliary modes keep LapTime_prev
        # out of the network input but still need it to define the residual target; the
        # hybrid passes out-of-sample tabular predictions). residual_from_laptime_prev
        # may fall back to the autoregressive column; residual_from_tabular may not.
        if baseline_model_series is not None and baseline_holdout_series is not None:
            baseline_model = pd.to_numeric(baseline_model_series.reset_index(drop=True), errors="coerce")
            baseline_holdout = pd.to_numeric(baseline_holdout_series.reset_index(drop=True), errors="coerce")
        elif target_mode == "residual_from_laptime_prev" and LSTM_AUTOREGRESSIVE_FEATURE in X_full_r.columns:
            baseline_model = pd.to_numeric(X_model_r[LSTM_AUTOREGRESSIVE_FEATURE], errors="coerce")
            baseline_holdout = pd.to_numeric(X_holdout_r[LSTM_AUTOREGRESSIVE_FEATURE], errors="coerce")
        else:
            raise ValueError(
                f"{target_mode} requires explicit baseline series"
                + (
                    f" or {LSTM_AUTOREGRESSIVE_FEATURE!r} as an input feature."
                    if target_mode == "residual_from_laptime_prev"
                    else "."
                )
            )
        baseline_full = pd.concat([baseline_model, baseline_holdout], ignore_index=True)
        y_model_target = y_model_r - baseline_model
        y_full_target = y_full_r - baseline_full
    elif target_mode == "absolute":
        baseline_full = pd.Series(0.0, index=X_full_r.index)
        y_model_target = y_model_r
        y_full_target = y_full_r
    else:
        raise ValueError("Unsupported lstm_target_mode. Use 'absolute' or 'residual_from_laptime_prev'.")

    if use_early_stopping:
        X_calibration_scaled, _, _, _, calibration_meta = fit_feature_transformers(
            X_model_r.loc[calibration_train_mask],
            X_model_r,
            cat_cols,
            embed_cols=embed_cols,
            max_dim=max_dim,
        )

        calibration_train_targets = X_model_r.index[calibration_train_mask.to_numpy()]
        calibration_val_targets = X_model_r.index[(~calibration_train_mask).to_numpy()]
        X_calibration_train_seq, y_calibration_train_raw, _, _, _ = build_sequences(
            X_calibration_scaled,
            y_model_target,
            lap_model_r,
            group_model_r,
            calibration_train_targets,
            sequence_length,
            include_target_row_features=include_target_row_features,
        )
        X_calibration_val_seq, y_calibration_val_raw, _, _, _ = build_sequences(
            X_calibration_scaled,
            y_model_target,
            lap_model_r,
            group_model_r,
            calibration_val_targets,
            sequence_length,
            include_target_row_features=include_target_row_features,
        )

        if len(X_calibration_train_seq) > 0 and len(X_calibration_val_seq) > 0:
            target_scaler_calibration = StandardScaler()
            y_calibration_train_scaled = target_scaler_calibration.fit_transform(
                y_calibration_train_raw.reshape(-1, 1)
            ).ravel()
            y_calibration_val_scaled = target_scaler_calibration.transform(
                y_calibration_val_raw.reshape(-1, 1)
            ).ravel()
            calibration_model = make_lstm_model(
                sequence_length,
                X_calibration_train_seq.shape[2],
                lstm_cfg,
                calibration_meta,
            )
            max_epochs = max(int(lstm_cfg["lstm_epochs"]), int(final_epoch_count))
            print(
                f"  Calibrating final epoch count: train_laps={n_model_laps - n_val_laps} "
                f"val_laps={n_val_laps} max_epochs={max_epochs} patience={int(lstm_cfg['lstm_patience'])}"
            )
            history = calibration_model.fit(
                X_calibration_train_seq,
                y_calibration_train_scaled,
                epochs=max_epochs,
                batch_size=int(lstm_cfg["lstm_batch_size"]),
                shuffle=False,
                verbose=0,
                validation_data=(X_calibration_val_seq, y_calibration_val_scaled),
                callbacks=training_callbacks(lstm_cfg),
            )
            final_epoch_count = int(np.argmin(history.history["val_loss"]) + 1)
            print(f"  Calibrated final epoch count: {final_epoch_count}")
            tf.keras.backend.clear_session()
        else:
            final_epoch_count = max(int(final_epoch_count), int(lstm_cfg["lstm_min_final_epochs"]))
            print(
                "  Skipping final epoch calibration because the modeling-tail split "
                f"did not produce both train and validation sequences; epochs={final_epoch_count}"
            )
    else:
        final_epoch_count = max(int(final_epoch_count), int(lstm_cfg["lstm_min_final_epochs"]))
        print(f"  Final epoch count from Optuna/YAML: {final_epoch_count}")

    X_full_scaled, imputer, feature_scaler, feature_names, feature_meta = fit_feature_transformers(
        X_model_r, X_full_r, cat_cols, embed_cols=embed_cols, max_dim=max_dim
    )

    train_targets = X_model_r.index
    holdout_targets = pd.RangeIndex(n_model, len(X_full_r))

    X_train_seq, y_train_raw, _, _, train_row_indices = build_sequences(
        X_full_scaled,
        y_full_target,
        lap_full_r,
        group_full_r,
        train_targets,
        sequence_length,
        include_target_row_features=include_target_row_features,
    )
    X_holdout_seq, y_holdout_target, holdout_laps, _, holdout_row_indices = build_sequences(
        X_full_scaled,
        y_full_target,
        lap_full_r,
        group_full_r,
        holdout_targets,
        sequence_length,
        include_target_row_features=include_target_row_features,
    )
    if len(X_train_seq) == 0 or len(X_holdout_seq) == 0:
        raise ValueError("Unable to build final LSTM sequences. Check sequence_length vs. lap continuity.")

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()

    model = make_lstm_model(sequence_length, X_train_seq.shape[2], lstm_cfg, feature_meta)
    print(
        f"  Training final model on full modeling block: sequences={len(X_train_seq)} "
        f"epochs={final_epoch_count}"
    )
    model.fit(
        X_train_seq,
        y_train_scaled,
        epochs=final_epoch_count,
        batch_size=int(lstm_cfg["lstm_batch_size"]),
        shuffle=False,
        verbose=0,
    )

    train_preds_scaled = model.predict(X_train_seq, verbose=0).ravel()
    train_preds_target = target_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).ravel()
    y_train_actual = y_full_r.loc[train_row_indices].to_numpy(dtype=float)
    if target_mode in RESIDUAL_TARGET_MODES:
        train_preds = train_preds_target + baseline_full.loc[train_row_indices].to_numpy(dtype=float)
    else:
        train_preds = train_preds_target
    train_metrics = metric_values(y_train_actual, train_preds)
    print(
        f"  Final model in-sample modeling sequences: RMSE={train_metrics['rmse']:.4f} | "
        f"MAE={train_metrics['mae']:.4f} | R2={train_metrics['r2']:.4f}"
    )

    preds_scaled = model.predict(X_holdout_seq, verbose=0).ravel()
    preds_target = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    y_holdout_seq = y_full_r.loc[holdout_row_indices].to_numpy(dtype=float)
    if target_mode in RESIDUAL_TARGET_MODES:
        holdout_baseline = baseline_full.loc[holdout_row_indices].to_numpy(dtype=float)
        preds = preds_target + holdout_baseline
        baseline_metrics = metric_values(y_holdout_seq, holdout_baseline)
        print(
            f"  Holdout {target_mode} baseline: RMSE={baseline_metrics['rmse']:.4f} | "
            f"MAE={baseline_metrics['mae']:.4f} | R2={baseline_metrics['r2']:.4f}"
        )
    else:
        preds = preds_target
    return preds, y_holdout_seq, holdout_laps, model, imputer, feature_scaler, target_scaler, feature_names, final_epoch_count, feature_meta, holdout_row_indices


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
    baseline_model: pd.Series | None = None,
):
    n_trials = int(base_cfg["lstm_optuna_trials"])
    tuning_enabled = bool(base_cfg["lstm_tuning_enabled"])
    use_saved = bool(base_cfg.get("use_saved_lstm_params", False))

    if use_saved:
        if params_path is not None and params_path.exists():
            print(f"Found saved LSTM parameters: {params_path}")
            with params_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            params_match = (
                loaded.get("search_space_version") == LSTM_SEARCH_SPACE_VERSION
                and loaded.get("tuning_strategy") == LSTM_TUNING_STRATEGY
                and int(loaded.get("n_trials", 0)) == n_trials
            )
            if params_match:
                print("Using saved LSTM parameters (search space version, strategy, and n_trials match).")
                best_cfg = dict(base_cfg)
                best_cfg.update(loaded["best_params"])
                best_epoch_count = int(loaded.get("best_epoch_count", int(base_cfg["lstm_tuning_epochs"])))
                return best_cfg, best_epoch_count, loaded

            mismatch_msg = (
                "Saved LSTM parameters do not match current search space version, strategy, or n_trials. "
                f"Expected version={LSTM_SEARCH_SPACE_VERSION}, strategy={LSTM_TUNING_STRATEGY}, "
                f"n_trials={n_trials}; got version={loaded.get('search_space_version')}, "
                f"strategy={loaded.get('tuning_strategy')}, n_trials={loaded.get('n_trials')}."
            )
            if not tuning_enabled:
                raise ValueError(mismatch_msg)
            print(f"{mismatch_msg} Running Optuna again.")
        elif not tuning_enabled:
            raise FileNotFoundError(
                f"use_saved_lstm_params=true and lstm_tuning_enabled=false, but no saved "
                f"LSTM parameter file was found at: {params_path}"
            )

    if not tuning_enabled:
        print("LSTM Optuna tuning disabled and saved parameters were not requested; using YAML hyperparameters.")
        return base_cfg, int(base_cfg["lstm_epochs"]), None

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
                baseline_context=baseline_model.loc[context_mask] if baseline_model is not None else None,
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
    # A época de re-treino vem do TRIAL VENCEDOR (mesma config de best_params),
    # não da mediana dos trials: best_epoch é dependente dos hiperparâmetros, então
    # misturar configs e aplicar à vencedora descasa o orçamento de épocas.
    best_epoch_count = int(
        study.best_trial.user_attrs.get("best_epoch_count", int(base_cfg["lstm_tuning_epochs"]))
    )
    print(f"Best LSTM Optuna RMSE: {study.best_value:.4f}")
    print(f"Best LSTM params: {study.best_params}")
    print(f"Best-trial epoch count (trial {study.best_trial.number}): {best_epoch_count}")

    optuna_summary = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_epoch_count": best_epoch_count,
        "best_epoch_count_source": "best_trial",
        "best_trial_number": int(study.best_trial.number),
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

    return best_cfg, best_epoch_count, optuna_summary


def main():
    target_gp_name, config, repo_root, laps_cleaned = load_cleaned_data(Path(__file__))
    df_base = laps_cleaned.copy()
    lstm_cfg = lstm_config(config)

    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])
    num_cols, cat_cols = select_modeling_columns(df_base, config)

    feature_mode = str(lstm_cfg.get("lstm_feature_mode", "auxiliary")).lower()
    lstm_cfg["lstm_feature_mode"] = feature_mode
    num_cols, cat_cols = resolve_lstm_features(feature_mode, num_cols, cat_cols, target_col)

    X_raw, y_raw, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    # Baseline for the residual target, kept separate from the input features so
    # auxiliary modes (which exclude LapTime_prev from the network input) can still
    # use residual_from_laptime_prev. No leakage: LapTime_prev is the previous lap.
    target_mode = str(lstm_cfg.get("lstm_target_mode", "absolute")).lower()
    if target_mode == "residual_from_tabular":
        raise ValueError(
            "lstm_target_mode='residual_from_tabular' is the hybrid mode; run "
            "model_lstm_hybrid.py instead of model_lstm.py (it supplies the tabular baseline)."
        )
    if target_mode == "residual_from_laptime_prev":
        if LSTM_AUTOREGRESSIVE_FEATURE not in df_base.columns:
            raise ValueError(
                f"lstm_target_mode={target_mode!r} requires {LSTM_AUTOREGRESSIVE_FEATURE!r} in the data."
            )
        baseline_raw = pd.to_numeric(df_base.loc[valid_indices, LSTM_AUTOREGRESSIVE_FEATURE], errors="coerce")
    else:
        baseline_raw = None

    group_cols = [col for col in list(lstm_cfg["lstm_group_cols"]) if col in df_base.columns]

    print("--- LSTM: SINGLE SEQUENTIAL SPLIT + SEQUENTIAL HOLDOUT ---")
    print(f"Grand Prix: {target_gp_name}")
    print(f"Feature mode: {feature_mode}")
    print(f"Numerical features: {num_cols}")
    print(f"Categorical features: {cat_cols}")
    print(f"Include target-row features in each sequence: {bool(lstm_cfg.get('lstm_include_target_row_features', True))}")
    print(f"LSTM target mode: {str(lstm_cfg.get('lstm_target_mode', 'absolute')).lower()}")
    embed_cols, embed_max_dim = resolve_embedding_cols(lstm_cfg, cat_cols)
    if embed_cols:
        onehot_cols = [c for c in cat_cols if c not in embed_cols]
        dims = {c: _embedding_dim(int(df_base[c].fillna("Missing").astype(str).nunique()), embed_max_dim) for c in embed_cols}
        print(f"Embedding (cat -> dim aprox.): {dims} | One-hot: {onehot_cols}")
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

    baseline_model = (
        baseline_raw.loc[model_order_idx].reset_index(drop=True) if baseline_raw is not None else None
    )
    baseline_holdout = (
        baseline_raw.loc[holdout_order_idx].reset_index(drop=True) if baseline_raw is not None else None
    )

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

    if bool(lstm_cfg["lstm_tuning_enabled"]) or bool(lstm_cfg.get("use_saved_lstm_params", False)):
        lstm_cfg, optuna_best_epoch, optuna_summary = tune_lstm_hyperparams(
            X_model_raw, y_model, lap_model_sorted, group_model,
            train_laps, val_laps, cat_cols, lstm_cfg, seed=seed,
            params_path=lstm_params_path,
            trials_path=lstm_trials_path,
            baseline_model=baseline_model,
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
        baseline_context=baseline_model.loc[context_mask] if baseline_model is not None else None,
    )
    val_metrics = metric_values(y_val_seq, preds_val)
    print(
        f"Val sequences: {len(y_val_seq)} | "
        f"RMSE={val_metrics['rmse']:.4f} | MAE={val_metrics['mae']:.4f} | R2={val_metrics['r2']:.4f}"
    )

    print("\n--- Training final LSTM model ---")
    (
        preds_holdout, y_holdout_seq, holdout_seq_laps,
        final_model, _, _, _, feature_names, final_epoch_count, feature_meta, _,
    ) = fit_final_lstm(
        X_model_raw, y_model, lap_model_sorted, group_model,
        X_holdout_raw, y_holdout, lap_holdout_sorted, group_holdout,
        cat_cols, lstm_cfg, seed=seed, final_epoch_count=final_epoch_count,
        baseline_model_series=baseline_model, baseline_holdout_series=baseline_holdout,
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
        r2_m=val_metrics["r2"], r2_holdout=holdout_metrics["r2"],
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
        "feature_mode": feature_mode,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "embedding_cols": embed_cols,
        "onehot_cols": [c for c in cat_cols if c not in embed_cols],
        "embedding_spec": feature_meta.get("embed_spec", []),
        "encoded_feature_names": list(feature_names),
        "sequence_length": int(lstm_cfg["lstm_sequence_length"]),
        "sequence_length_source": lstm_cfg["lstm_sequence_length_source"],
        "include_target_row_features": bool(lstm_cfg.get("lstm_include_target_row_features", True)),
        "target_mode": str(lstm_cfg.get("lstm_target_mode", "absolute")).lower(),
        "lstm_window_ratio": lstm_window_ratio,
        "window_train_ratio": float(config["window_train_ratio"]),
        "modeling_lap_count": int(n_model_laps),
        "train_laps": len(train_laps),
        "val_laps": len(val_laps),
        "sequence_groups": group_cols,
        "training_block": "first_sequential_modeling_block",
        "holdout_usage": "holdout laps are forecast targets only; their lookback may reach the modeling tail. Holdout is never used for training, tuning, or early stopping",
        "preprocessing": "median_imputer_standard_scaler_one_hot_full_rank_or_embeddings",
        "lstm_config": lstm_cfg,
        "final_epoch_count": int(final_epoch_count),
        "final_training_strategy": "epoch_calibration_on_modeling_tail_then_full_modeling_retrain",
        "final_val_ratio": float(lstm_cfg.get("lstm_final_val_ratio", 0.15)),
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
        "cos_r2": cos["cos_r2"],
        "cos_r2_ci": cos["cos_r2_ci"],
    }
    log_mlflow_run(
        repo_root, config, "lstm", num_cols, cat_cols,
        split_info, results_for_cos, summary_metrics,
        extra_params={
            "preprocessing": "median_imputer_standard_scaler_one_hot_full_rank_or_embeddings",
            "feature_mode": feature_mode,
            "validation_protocol": "single_sequential_split",
            "tuning_strategy": LSTM_TUNING_STRATEGY,
            "search_space_version": LSTM_SEARCH_SPACE_VERSION,
            "sequence_length": int(lstm_cfg["lstm_sequence_length"]),
            "sequence_length_source": lstm_cfg["lstm_sequence_length_source"],
            "include_target_row_features": bool(lstm_cfg.get("lstm_include_target_row_features", True)),
            "target_mode": str(lstm_cfg.get("lstm_target_mode", "absolute")).lower(),
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
    print(f"COS_R2:   {cos['cos_r2']:.4f} | 95% CI: [{cos['cos_r2_ci'][0]:.4f}, {cos['cos_r2_ci'][1]:.4f}]")
    print(f"          R2 final/val={cos['r2_final']:.4f}/{cos['r2_sw']:.4f} | STD final/val={cos['std_final']:.4f}/{cos['std_sw']:.4f}")


if __name__ == "__main__":
    main()
