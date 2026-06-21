# Modeling Scripts

Reproducible modeling scripts and shared utilities for the experiment.

The experiment is built around three models:

- **LR-EW** (`model_lr_ew.py`) — Linear Regression, expanding-window validation.
- **XGBoost-EW** (`model_xgb_ew.py`) — XGBoost, expanding-window validation.
- **`LSTM_hybrid`** (`model_lstm_hybrid.py`) — Linear Regression (LR-EW) baseline
  plus an LSTM that predicts the residual.

Two extra models are kept because they were tested during the research:

- **SW models** (`model_lr_sw.py`, `model_xgb_sw.py`) — sliding-window variants.
- **Baseline-LapTime_prev LSTM** (`model_lstm_baseline.py`) — an LSTM whose baseline
  is the **previous lap time**: it learns the residual `LapTime_seconds - LapTime_prev`
  and **does not use `LapTime_prev` as a network feature**.

## YAML Configuration

Scripts read their settings from `configs/*.yaml` via the `CONFIG_PATH` environment
variable. When `CONFIG_PATH` is not set, the script loads the YAML for the selected
`TARGET_GP_NAME`. The YAML is the single source of truth (target, paths, features,
split ratios, model settings, tracking). Relative paths resolve from the repo root.

```bash
CONFIG_PATH="configs/bahrain.yaml" python Scripts/Source/model_lr_ew.py
CONFIG_PATH="configs/bahrain.yaml" python Scripts/Source/model_xgb_ew.py
CONFIG_PATH="configs/bahrain.yaml" python Scripts/Source/model_lstm_hybrid.py
```

```powershell
$env:CONFIG_PATH = "configs/bahrain.yaml"
python Scripts/Source/model_lr_ew.py
```

## MLflow Tracking

The model scripts log metadata to MLflow when `mlflow_enabled` is true. The default
tracking directory is `Scripts/Results/mlruns` (generated output, not committed).

- LR and XGBoost runs record the Grand Prix, feature lists, split ratios, window
  sizes, seed, per-window and final holdout metrics (RMSE, MAE, R2, residual STD),
  the COS metrics, and JSON artifacts.
- LSTM and `LSTM_hybrid` runs record the Grand Prix, feature lists, split ratios,
  the single sequential-split validation metrics, final holdout metrics, the selected
  sequence configuration, and the saved Keras model metadata. `LSTM_hybrid` runs
  additionally record the Linear Regression (LR-EW) tabular baseline, the selected
  `lstm_window_ratio`, and the baseline holdout metrics.

Start the local UI from the repo root:

```bash
python -m mlflow ui --backend-store-uri Scripts/Results/mlruns
```

## Core models

### `model_lr_ew.py` and `model_xgb_ew.py`

Expanding-window (EW) Linear Regression and XGBoost — the final reported validation
protocol for every circuit. The training set grows cumulatively across folds and each
fold validates on the next fixed-size chunk. Per-circuit window ratios are
`lr_ew_window_ratio` and `xgb_ew_window_ratio` in the YAML. XGBoost-EW runs an
independent Optuna study per fold and aggregates the best hyperparameters by median
across folds (`n_estimators` as the median early-stopping iteration). Preprocessing
(median imputation, scaling, one-hot fitted only on each training split), COS metrics,
bootstrap holdout CIs, and the untouched sequential holdout apply to both.

### `model_lstm_hybrid.py`

Runs the `LSTM_hybrid` model. By design it uses Linear Regression (LR-EW) as the
tabular expanding-window baseline so the model keeps a strong linear component, plus an
LSTM that predicts the baseline residual to capture the remaining complex
relationships. The final prediction is `baseline_prediction + lstm_residual_prediction`.

- Reuses the `model_lstm_baseline.py` core unchanged (network, sequences, Optuna,
  epoch calibration) by forcing `lstm_target_mode = 'residual_from_tabular'`.
- The baseline uses an out-of-fold (expanding-window) prediction series over the
  modeling block for both the validation split and the final residual targets, so every
  baseline value is produced by a model trained only on earlier laps; the holdout is
  never used to train or select the baseline. Standalone LR-EW per-row prediction
  exports are reused when present (via `baseline_utils.py`).
- Sweeps `lstm_window_ratio_sweep` and keeps the configuration with the best
  validation RMSE (never the holdout).
- Artifacts are saved under `Scripts/Results/lstm_hybrid/`.

## Extra models

### `model_lr_sw.py` and `model_xgb_sw.py`

Sliding-window (SW) variants kept for reference. Same preprocessing, COS metrics, and
untouched sequential holdout as the EW scripts; validation slides a fixed-size window
across the modeling block. XGBoost-SW runs independent Optuna studies inside each
window. EW (above) is the protocol selected for the final results.

### `model_lstm_baseline.py`

The baseline-LapTime_prev LSTM. The baseline is the driver's **previous lap time**: the
network learns the residual `LapTime_seconds - LapTime_prev`
(`lstm_target_mode = residual_from_laptime_prev`) and the final prediction is
`LapTime_prev + lstm_residual_prediction`. **`LapTime_prev` is not a network feature**
— the configured `lstm_feature_mode` is `auxiliary_embedding`, which drops `LapTime_prev`
while keeping Driver/Team embeddings; the previous-lap signal enters only through the
residual target.

- Single sequential split inside the modeling block (first `window_train_ratio` trains,
  the rest validates); no sliding/expanding windows.
- Sequences grouped by `lstm_group_cols` (`[Year, Driver]`), sorted by `LapNumber`,
  with `sequence_length = ceil(n_race_laps * lstm_window_ratio)`.
- Optuna tuning on the single split with `EarlyStopping` calibrating the epoch count;
  final retraining on the full modeling block; bootstrap CIs and COS metrics on the
  untouched holdout.

## Analysis and figures

### `model_interpretability.py`

Loads the saved final Linear Regression and XGBoost models for the selected Grand Prix.
LR is interpreted through standardized encoded coefficients; XGBoost through native
feature importance and SHAP values. Outputs (coefficient CSV/PNG, feature-importance
and SHAP CSV/PNG, force plots, manifest JSON) are written under
`Scripts/Results/model_interpretability/`.

```bash
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_lr_sw.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_xgb_sw.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_interpretability.py
```

### `plot_driver_holdout_timeseries.py`

Rebuilds the sequential split, predicts the holdout laps with the LR or XGBoost final
model, and plots a single driver's actual vs predicted lap-time series with an
approximate 95% prediction band.

```bash
CONFIG_PATH=configs/hungary.yaml python Scripts/Source/plot_driver_holdout_timeseries.py --driver VER
```

## Runner

### `run_experiment.py`

Runs the final reported experiment (LR-EW + XGBoost-EW) for all circuits, or a subset
via `--circuit`, using the window ratios encoded in each YAML. Add `--with-hybrid` to
also run `LSTM_hybrid` after the tabular models, so the reused EW baseline predictions
are already available.

```bash
python Scripts/Source/run_experiment.py
python Scripts/Source/run_experiment.py --circuit bahrain italy
python Scripts/Source/run_experiment.py --with-hybrid
python Scripts/Source/run_experiment.py --continue-on-error
```

## Shared utilities

- `modeling_utils.py` — configuration loading, temporal splitting, one-hot alignment,
  metric calculation, bootstrap confidence intervals, and COS reporting helpers.
- `baseline_utils.py` — hybrid tabular-baseline helpers (out-of-fold expanding-window
  predictions, block predictions, saved-prediction reuse, XGBoost-EW hyperparameter
  resolution) with leakage control.
- `xgb_utils.py` — XGBoost matrix building and base parameters.

## COS Metrics

The LR and XGBoost scripts report (with `SW_or_EW` denoting the validation block and
`final` the sequential holdout):

```text
COS_MAE  = 0.5 * (MAE_SW_or_EW / MAE_final)  + 0.5 * (STD_SW_or_EW / STD_final)
COS_RMSE = 0.5 * (RMSE_SW_or_EW / RMSE_final) + 0.5 * (STD_SW_or_EW / STD_final)
```

For the LSTM and `LSTM_hybrid` models the single validation split plays the SW/EW role.
The COS confidence intervals are indicative because the windows overlap (SW) or the
training sets grow across folds (EW).
