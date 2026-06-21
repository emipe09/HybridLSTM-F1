# Modeling Scripts

Reproducible modeling scripts, feature-selection scripts, and shared modeling
utilities are kept here.

## YAML Configuration

The scripts can read experiment settings from `configs/*.yaml` through the
`CONFIG_PATH` environment variable:

```powershell
$env:CONFIG_PATH = "configs/bahrain.yaml"
.\.venv\Scripts\python.exe Scripts/Source/model_lr_ew.py
.\.venv\Scripts\python.exe Scripts/Source/model_xgb_ew.py
.\.venv\Scripts\python.exe Scripts/Source/model_lstm.py
.\.venv\Scripts\python.exe Scripts/Source/model_lstm_hybrid.py
.\.venv\Scripts\python.exe Scripts/Source/correlation_ablation_lr.py
```

```bash
CONFIG_PATH="configs/bahrain.yaml" .venv/bin/python Scripts/Source/model_lr_ew.py
CONFIG_PATH="configs/bahrain.yaml" .venv/bin/python Scripts/Source/model_xgb_ew.py
CONFIG_PATH="configs/bahrain.yaml" .venv/bin/python Scripts/Source/model_lstm.py
CONFIG_PATH="configs/bahrain.yaml" .venv/bin/python Scripts/Source/model_lstm_hybrid.py
```

When `CONFIG_PATH` is not defined, the scripts load the corresponding YAML file
for the selected `TARGET_GP_NAME`. There are no built-in modeling defaults; the
YAML file is the source of truth for target, paths, features, split ratios,
model settings, and tracking settings.

Directory and file locations are also configured in YAML. The main path keys are:

- `data_dir`
- `model_data_dir`
- `results_dir`
- `cleaned_data_filename_template`
- `xgb_params_subdir`
- `xgb_params_filename_template`
- `mlflow_enabled`
- `mlflow_tracking_uri`
- `mlflow_experiment_name`

Relative paths are resolved from the repository root.

## MLflow Tracking

The model scripts log experiment metadata to MLflow when `mlflow_enabled` is
true. The default tracking directory is `Scripts/Results/mlruns`, which is
generated output and should not be committed.

Linear Regression and XGBoost runs record:

- selected Grand Prix, feature lists, split ratios, window sizes, and random seed
- sliding-window RMSE, MAE, R2, residual standard deviation, and per-window values
- final sequential-holdout RMSE, MAE, R2, residual standard deviation, and confidence intervals
- `COS_MAE`, `COS_RMSE`, and their descriptive confidence intervals
- JSON artifacts for the resolved configuration, per-window results, and summary metrics

LSTM and `LSTM_hybrid` runs record the same selected Grand Prix, feature lists,
split ratios, single sequential-split validation metrics, final sequential-holdout
metrics, the selected sequence configuration, and the saved Keras model metadata.
`LSTM_hybrid` runs additionally record the tabular baseline model, the selected
`lstm_window_ratio`, and the baseline holdout metrics.

Start the local UI from the repository root with:

```powershell
python -m mlflow ui --backend-store-uri Scripts/Results/mlruns
```

## `model_lr_sw.py`

Runs Linear Regression for the selected Grand Prix using:

- median imputation for numerical predictors
- standard scaling
- one-hot encoding with `drop_first=True`
- categorical encoding fitted only on each training split/window
- sliding-window validation over the modeling block
- final sequential holdout evaluation
- saved final model artifact and metadata for downstream interpretability

## `model_xgb_sw.py`

Runs XGBoost for the selected Grand Prix using:

- one-hot encoding with all categories retained
- median imputation for numerical predictors
- categorical encoding fitted only on each training split/window
- per-window Optuna tuning when no saved parameters are available for the current search-space version, tuning strategy, YAML bounds, and sampler
- independent Optuna studies inside each sliding window, with `optuna_trials` interpreted as trials per window
- Optuna selection inside each window based exclusively on validation RMSE
- final holdout hyperparameters aggregated as the median of the best Optuna parameters from all sliding windows
- final `n_estimators` calibrated as the median early-stopping iteration across all sliding windows
- saved per-window parameter summaries and per-trial CSV files for search-space auditing
- final sequential holdout evaluation
- saved final booster artifact and metadata for downstream interpretability

## `model_lr_ew.py` and `model_xgb_ew.py`

Expanding-window (EW) variants of the Linear Regression and XGBoost scripts. The
training set grows cumulatively across folds and each fold validates on the next
fixed-size chunk (same chunk size as the SW validation portion). EW is the final
reported validation protocol for every circuit; the per-circuit EW window ratios
are `lr_ew_window_ratio` and `xgb_ew_window_ratio` in the YAML files. XGBoost-EW
runs an independent Optuna study per fold and aggregates the best hyperparameters
by median across all folds (`n_estimators` as the median early-stopping iteration).
Preprocessing, COS metrics, bootstrap holdout CIs, and the untouched sequential
holdout follow the same rules as the SW scripts.

## `model_lstm.py`

Runs the pure Keras LSTM for the selected Grand Prix using:

- a **single sequential split** instead of sliding/expanding windows: the first
  `window_train_ratio` of modeling laps train and the remaining laps validate
- sequences grouped by `lstm_group_cols` (`[Year, Driver]`), sorted by `LapNumber`;
  for each target lap the `sequence_length` preceding laps of the same group form
  the input, with `sequence_length = ceil(n_race_laps * lstm_window_ratio)`
- median imputation → `StandardScaler` on features, full-rank one-hot encoding, and
  a separate `StandardScaler` on the target, all fitted on the training portion only
- Optuna tuning on the single split (search-space version v8, tuning strategy
  `single_sequential_split_v1`), with `EarlyStopping` on `val_loss` calibrating the
  epoch count
- final retraining on the full 80% modeling block for
  `max(median_optuna_epochs, lstm_min_final_epochs)` epochs
- final sequential holdout evaluation with bootstrap CIs and COS metrics
- saved final Keras model artifact, metadata, Optuna params JSON, and per-trial CSV

The final sequential holdout is never used for training, hyperparameter selection,
or early stopping. Saved params are reused only when their `search_space_version`,
`tuning_strategy`, and `n_trials` match the current code.

## `model_lstm_hybrid.py`

Runs the selected `LSTM_hybrid` model: the best tabular expanding-window baseline
(LR-EW or XGBoost-EW, set per circuit via `hybrid_baseline_model` from validation
metrics, never the holdout) plus an LSTM that predicts the residual. The final
prediction is `baseline_prediction + lstm_residual_prediction`.

- Reuses the `model_lstm.py` core unchanged (network, sequences, Optuna, epoch
  calibration) by forcing `lstm_target_mode = 'residual_from_tabular'`.
- The baseline uses an out-of-fold (expanding-window) prediction series over the
  modeling block for both the validation split and the final residual targets, so
  every baseline value is produced by a model trained only on earlier laps; the
  holdout is never used to train or select the baseline. Standalone LR-EW/XGB-EW
  per-row prediction exports are reused when present (via `baseline_utils.py`).
- Sweeps `lstm_window_ratio_sweep` and keeps the configuration with the best
  validation RMSE (never the holdout).
- Artifacts are saved under `Scripts/Results/lstm_hybrid/` (models, params, and an
  audit copy of the baseline predictions).

## `model_interpretability.py`

Loads the saved final Linear Regression and XGBoost models for the selected
Grand Prix. Linear Regression is interpreted through standardized encoded
coefficients. XGBoost is interpreted through native feature importance and SHAP
values, following the analysis pattern used in the notebooks.

Run the model scripts first so the final model artifacts exist:

```powershell
$env:TARGET_GP_NAME = "Bahrain Grand Prix"
.\.venv\Scripts\python.exe Scripts/Source/model_lr_sw.py
.\.venv\Scripts\python.exe Scripts/Source/model_xgb_sw.py
.\.venv\Scripts\python.exe Scripts/Source/model_interpretability.py
.\.venv\Scripts\python.exe Scripts/Source/model_interpretability.py --force-index 518
.\.venv\Scripts\python.exe Scripts/Source/model_interpretability.py --all
```

When `--force-index` is omitted, the force plot explains the modeling-block row
whose XGBoost prediction is closest to the median modeling-block prediction.

Outputs are written under `Scripts/Results/model_interpretability/` by default:

- Linear Regression coefficient CSV and PNG
- XGBoost feature-importance CSV and gain plot
- XGBoost row-level SHAP values as CSV
- XGBoost SHAP summary CSV, beeswarm PNG, bar PNG, force-plot PNG/HTML, and force-plot contribution CSV
- manifest JSON linking the outputs to the saved source models

## `backward_elimination.py`

Runs p-value based backward elimination for the selected Grand Prix using the
same configured target, feature lists, and final 20% sequential holdout split.
The elimination procedure fits one-hot encoding, median imputation, scaling,
and OLS only on the first modeling block; the final holdout is reported as
untouched and is not used for feature selection. OLS is solved directly with
NumPy to avoid fragile `statsmodels`/`scipy` imports on local Python builds; the
reported p-values use a large-sample normal approximation.

```powershell
$env:TARGET_GP_NAME = "Bahrain Grand Prix"
.\.venv\Scripts\python.exe Scripts/Source/backward_elimination.py
.\.venv\Scripts\python.exe Scripts/Source/backward_elimination.py --all
```

Outputs are written under `Scripts/Results/backward_elimination/` by default:

- elimination history as CSV
- strong encoded-feature correlations (`r > 0.80` or `r < -0.80`) as CSV,
  including whether each correlated feature was removed
- selected and removed features as JSON
- full-model versus reduced-model `R2`, RMSE, and MAE comparison in JSON
- final OLS summary as text

## `correlation_ablation_lr.py`

Detects encoded-feature pairs with absolute correlation above the selected
threshold inside the first sequential modeling block only. For each correlated
pair, the script reruns the Linear Regression sliding-window and final
sequential-holdout protocol twice: once removing the first feature and once
removing the second feature. The holdout remains untouched when selecting the
correlated pairs.

```powershell
$env:TARGET_GP_NAME = "Bahrain Grand Prix"
.\.venv\Scripts\python.exe Scripts/Source/correlation_ablation_lr.py
.\.venv\Scripts\python.exe Scripts/Source/correlation_ablation_lr.py --threshold 0.80 --all
```

Outputs are written under `Scripts/Results/correlation_ablation_lr/` by default:

- correlated encoded-feature pairs as CSV
- one-row-per-ablation results as CSV
- baseline metrics, split metadata, and detected pairs as JSON

## `regression_diagnostics_lr.py`

Generates Linear Regression residual diagnostics after fitting preprocessing
and the model on the first sequential modeling block. Residual plots,
prediction tables, a standard statsmodels OLS summary, coefficient tables, and
coefficient plots are produced for that retrained 80% modeling block; the final
sequential holdout is not used by this diagnostic script and remains reserved
for final evaluation.

```powershell
$env:TARGET_GP_NAME = "Bahrain Grand Prix"
.\.venv\Scripts\python.exe Scripts/Source/regression_diagnostics_lr.py
.\.venv\Scripts\python.exe Scripts/Source/regression_diagnostics_lr.py --all
```

Outputs are written under `Scripts/Results/regression_diagnostics/` by default:

- modeling-block diagnostics as CSV
- modeling-block coefficient estimates and p-values as CSV
- standard statsmodels OLS summary as TXT
- modeling-block summary metrics as JSON
- regression-diagnostics panel as PNG
- residual-distribution histogram as PNG
- all encoded model coefficients as PNG

## `extract_pca_loading_cells.py`

Generates static 2D PCA loading PNGs directly from the configured cleaned
modeling datasets. The script follows the PCA logic previously used in the
notebooks, but it does not read or extract notebook cells.

```powershell
.\.venv\Scripts\python.exe Scripts/Source/extract_pca_loading_cells.py
```

Outputs are written under `Scripts/Results/pca_loading_cells/` by default:

- `pca_loading_manifest.json`
- `pca_loading_images_manifest.json`
- `images/<safe_gp_name>/pca_loadings_*.png`
- `images/<safe_gp_name>/pca_explained_variance_*.png`
- `top5_pc1_pc2_loadings_by_track.csv`
- `top5_pc1_pc2_loadings_by_track.png`
- `pc1_pc2_loadings_by_track.csv`
- `requested_pc1_pc2_loadings_by_track.csv`
- `pca_explained_variance_by_track.csv`

## `run_all_models.py`

Runs the configured Grand Prix YAML files in sequence using the same Python
interpreter that launched the batch script:

```powershell
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py
```

Useful options:

```powershell
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py --models lr
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py --models xgb
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py --models lstm
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py --continue-on-error
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py --configs bahrain.yaml usa.yaml
```

```bash
.venv/bin/python Scripts/Source/run_all_models.py --models lr
.venv/bin/python Scripts/Source/run_all_models.py --models xgb
.venv/bin/python Scripts/Source/run_all_models.py --models lstm
.venv/bin/python Scripts/Source/run_all_models.py --continue-on-error
.venv/bin/python Scripts/Source/run_all_models.py --configs bahrain.yaml usa.yaml
```

The default run order is Bahrain, Saudi Arabia, United States, Italy, and
Hungary. Each subprocess receives `CONFIG_PATH` and `TARGET_GP_NAME`, so the
individual model scripts keep their normal configuration flow and MLflow logging.

## `window_size_sweep.py`, `search_space_sweep.py`, `search_space_sweep_ew.py`

- `window_size_sweep.py`: evaluates all four SW/EW × LR/XGBoost combinations across
  the YAML-configured window-ratio range (default 5%–50% in 5% steps) and writes a
  per-circuit CSV. Used to select the final per-circuit method and window size.
- `search_space_sweep.py`: runs baseline XGBoost configurations on the generic
  window size to derive an initial directional prior for each circuit's search space.
- `search_space_sweep_ew.py`: runs the baseline XGBoost configurations on the final
  selected EW window size per circuit (`xgb_ew_window_ratio`) to derive the
  definitive circuit-specific search-space bounds stored in the YAML files.

## `run_experiment.py`

Runs the final reported experiment (LR-EW + XGBoost-EW) for all circuits, or a
subset via `--circuit`, using the window ratios encoded in each YAML:

```bash
python Scripts/Source/run_experiment.py
python Scripts/Source/run_experiment.py --circuit bahrain italy
python Scripts/Source/run_experiment.py --with-hybrid
python Scripts/Source/run_experiment.py --continue-on-error
```

Add `--with-hybrid` to also run `LSTM_hybrid` after the tabular models for each
circuit, so the reused EW baseline predictions are already available.

## `modeling_utils.py` and `baseline_utils.py`

`modeling_utils.py` contains shared configuration loading, temporal splitting,
one-hot alignment, metric calculation, bootstrap confidence intervals, and COS
reporting helpers. `baseline_utils.py` provides the hybrid tabular-baseline helpers
(out-of-fold expanding-window predictions, block predictions, saved-prediction
reuse, and XGBoost-EW hyperparameter resolution) with leakage control.

## COS Metrics

The Linear Regression and XGBoost scripts report (with `SW_or_EW` denoting the
validation block and `final` the sequential holdout):

```text
COS_MAE  = 0.5 * (MAE_SW_or_EW / MAE_final)  + 0.5 * (STD_SW_or_EW / STD_final)
COS_RMSE = 0.5 * (RMSE_SW_or_EW / RMSE_final) + 0.5 * (STD_SW_or_EW / STD_final)
```

For the LSTM and `LSTM_hybrid` models the single validation split plays the SW/EW
role in these formulas. The confidence intervals for COS are indicative because the
sliding windows overlap (SW) or the expanding training sets grow across folds (EW).
