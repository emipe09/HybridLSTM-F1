# Hyperparameter Search Spaces (XGBoost & LSTM)

Supplementary material omitted from the paper for space. Every value below is taken
directly from the source code and the per-circuit YAML files — nothing here is invented.
Source of truth: [`configs/*.yaml`](../configs), [`Scripts/Source/xgb_utils.py`](../Scripts/Source/xgb_utils.py),
[`Scripts/Source/model_xgb_ew.py`](../Scripts/Source/model_xgb_ew.py),
[`Scripts/Source/model_lstm_baseline.py`](../Scripts/Source/model_lstm_baseline.py).

---

## 1. XGBoost (`model_xgb_ew.py`)

### 1.1 Tuning protocol
- **Optimizer:** Optuna, TPE sampler (`xgb_optuna_sampler: "tpe"`), seeded with `random_seed: 42`.
- **Trials:** `optuna_trials: 100` per fold.
- **Strategy** (`XGB_EW_TUNING_STRATEGY = "per_fold_all_folds_median_params_v1"`): an independent
  Optuna study runs on *each* expanding-window fold; the best trial per fold is selected by
  validation RMSE; the final hyperparameters are the **median across folds**
  (`aggregate_window_params`), and `n_estimators` is the **median early-stopping iteration**
  across folds (`median_n_estimators`).
- **Training:** `num_boost_round=5000` with `early_stopping_rounds=100` on validation RMSE.
- **Base params** (`BASE_XGB_PARAMS`): `objective=reg:squarederror`, `tree_method=hist`,
  `eval_metric=rmse`, `nthread=-1`.
- **Search-space version:** `XGB_SEARCH_SPACE_VERSION = "gp_final_random_v1"`.

### 1.2 Per-circuit search space (min – max)
Bounds come from the `xgb_*_min` / `xgb_*_max` keys in each YAML. Where a key is absent the
code falls back to `DEFAULT_XGB_SEARCH_SPACE` in `xgb_utils.py`.

| Parameter (type) | Bahrain | Saudi Arabia | USA | Italy | Hungary |
|---|---|---|---|---|---|
| `learning_rate` (float, log) | 0.006667 – 0.300 | 0.006667 – 0.300 | 0.0167 – 0.500 | 0.0167 – 0.300 | 0.006667 – 0.300 |
| `max_depth` (int) | 3 – 7 | 1 – 7 | 3 – 9 | 1 – 7 | 1 – 5 |
| `min_child_weight` (int) | 4 – 6 | 4 – 6 | 1 – 6 | 4 – 6 | 4 – 6 |
| `subsample` (float) | 0.656 – 0.844 | 0.678 – 0.872 | 0.719 – 1.000 | 0.678 – 0.872 | 0.678 – 0.872 |
| `colsample_bytree` (float) | 0.744 – 0.956 | 0.722 – 0.928 | 0.809 – 1.000 | 0.722 – 0.928 | 0.722 – 0.928 |
| `gamma` (float) | 0.0 – 0.700 | 0.0 – 0.700 | 0.0 – 0.700 | 0.0 – 0.700 | 0.0 – 0.700 |
| `reg_alpha` (float, log) | 0.000333 – 0.0300 | 0.000333 – 0.0300 | 0.00000333 – 0.0300 | 0.000333 – 0.0300 | 0.000333 – 0.003000 |
| `reg_lambda` (float, log) | 0.167 – 3.000 | 0.167 – 3.000 | 0.0333 – 3.000 | 0.167 – 3.000 | 0.167 – 3.000 |

`DEFAULT_XGB_SEARCH_SPACE` (fallback): `learning_rate` 0.01–0.10 (log), `max_depth` 2–6,
`min_child_weight` 5–30, `subsample` 0.6–0.9, `colsample_bytree` 0.6–0.9, `gamma` 0.2–8.0,
`reg_alpha` 1e-4–10.0 (log), `reg_lambda` 0.1–30.0 (log).

### 1.3 Final reported XGBoost-EW hyperparameters
Median of the best Optuna parameters across all expanding-window folds (source:
`Scripts/Results/xgboost/ew/params/{safe_gp_name}_xgb_params_ew.json`, git-ignored). Shared:
seed 42, TPE sampler, 100 Optuna trials/fold, `n_estimators` = median early-stopping iteration.

| Grand Prix | EW window | n_estimators | learning_rate | max_depth | min_child_weight | subsample | colsample_bytree | gamma | reg_alpha | reg_lambda |
|---|---|---|---|---|---|---|---|---|---|---|
| Bahrain | 30% | 249 | 0.164969 | 3 | 5 | 0.684174 | 0.904215 | 0.203528 | 0.001370 | 0.510169 |
| Saudi Arabia | 50% | 152 | 0.199282 | 3 | 4 | 0.756074 | 0.833232 | 0.316344 | 0.003249 | 0.283445 |
| USA | 5% | 85 | 0.201475 | 6 | 4 | 0.784062 | 0.887642 | 0.213003 | 0.000624 | 0.425945 |
| Italy | 50% | 395 | 0.061874 | 1 | 5 | 0.784588 | 0.779387 | 0.148427 | 0.002628 | 0.265021 |
| Hungary | 40% | 1050 | 0.042898 | 1 | 6 | 0.769281 | 0.909664 | 0.504674 | 0.001244 | 0.531199 |

### 1.4 Selected expanding-window ratios
From the window-size sweep (5%–50%), the per-model ratio kept by validation performance:

| | Bahrain | Saudi | USA | Italy | Hungary |
|---|---|---|---|---|---|
| `lr_ew_window_ratio` | 0.05 | 0.10 | 0.45 | 0.05 | 0.45 |
| `xgb_ew_window_ratio` | 0.30 | 0.50 | 0.05 | 0.50 | 0.40 |

---

## 2. Hybrid LR-LSTM (`model_lstm_hybrid.py` + `model_lstm_baseline.py`)

The hybrid keeps **LR-EW** as the tabular baseline (out-of-fold expanding-window predictions,
no leakage — see `baseline_utils.py`) and trains an LSTM to predict the residual
`LapTime_seconds − baseline_prediction`; final prediction = `baseline + lstm_residual`.

### 2.1 Architecture
`LSTM → BatchNormalization → Dense(relu) → Dense(1)`, with **entity embeddings** for the
categorical columns `lstm_embedding_cols: [Driver, Team]`
(`embedding_dim = min(max_dim, (cardinality+1)//2)`, `lstm_embedding_max_dim: 8`).
- **Loss:** Huber (`delta = lstm_huber_delta`). **Optimizer:** Adam.
- **Sequences:** grouped by `lstm_group_cols: [Year, Driver]`; sequence length =
  `ceil(n_race_laps · lstm_window_ratio)`; `lstm_include_target_row_features: true`.
- **Feature mode:** hybrid uses `full_embedding`; the standalone baseline LSTM uses `auxiliary_embedding`.
- **Callbacks:** `EarlyStopping(val_loss)` + `ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5)`.

### 2.2 Tuning protocol
- **Optimizer:** Optuna, `lstm_optuna_trials: 50`, strategy `single_sequential_split_v1`
  (first `window_train_ratio` = 80% trains, remainder validates).
- **Search-space version:** `LSTM_SEARCH_SPACE_VERSION = "v11"`, `suggest_lstm_config`:

  | Hyperparameter | Search range |
  |---|---|
  | `lstm_units` | {8, 16, 24, 32} |
  | `lstm_dense_units` | {24, 32, 48} |
  | `lstm_dropout` | 0.22 – 0.42 |
  | `lstm_recurrent_dropout` | 0.05 – 0.18 |
  | `lstm_learning_rate` | 7e-4 – 1.8e-3 (log) |
  | `lstm_batch_size` | {16, 32} |
  | `lstm_l2_reg` | 4e-4 – 1.5e-3 |
  | `lstm_huber_delta` | 0.5 – 2.0 |

  Tuning uses `lstm_tuning_epochs: 40`, `lstm_tuning_patience: 5`.

### 2.3 Final per-circuit hyperparameters
Persisted in the YAML files (selected from validation RMSE, never the holdout):

| Hyperparameter | Bahrain | Saudi | USA | Italy | Hungary |
|---|---|---|---|---|---|
| `lstm_units` | 24 | 24 | 8 | 16 | 32 |
| `lstm_dense_units` | 48 | 24 | 48 | 32 | 48 |
| `lstm_dropout` | 0.4132 | 0.3466 | 0.3239 | 0.2250 | 0.2914 |
| `lstm_recurrent_dropout` | 0.0829 | 0.0875 | 0.1611 | 0.0849 | 0.0865 |
| `lstm_learning_rate` | 0.000718 | 0.001543 | 0.001654 | 0.001116 | 0.001169 |
| `lstm_batch_size` | 16 | 32 | 32 | 32 | 32 |
| `lstm_epochs` | 14 | 4 | 7 | 13 | 4 |
| `lstm_patience` | 12 | 8 | 8 | 8 | 8 |
| `lstm_l2_reg` | 0.000693 | 0.000839 | 0.000753 | 0.001139 | 0.000482 |
| `lstm_huber_delta` | 1.554 | 1.240 | 1.237 | 1.831 | 1.980 |
| `lstm_window_ratio` (baseline) | 0.05 | 0.50 | 0.05 | 0.50 | 0.15 |
| `lstm_window_ratio_sweep` (hybrid) | [0.30] | [0.03] | [0.10] | [0.05] | [0.03] |

Shared: `lstm_embedding_max_dim: 8`, `lstm_reduce_lr_factor: 0.50`,
`lstm_reduce_lr_patience: 4`, `lstm_min_learning_rate: 1e-5`, `random_seed: 42`.

> These tables also supply the LSTM/embedding/hyperparameter-tuning details flagged as
> *to be added* in the paper draft (Section 3.3 placeholder).
