# Modeling Scripts

Only the current sliding-window modeling scripts and shared modeling utilities
are kept here.

## YAML Configuration

The scripts can read experiment settings from `configs/*.yaml` through the
`CONFIG_PATH` environment variable:

```powershell
$env:CONFIG_PATH = "configs/bahrain.yaml"
python Scripts/Source/model_lr_sw.py
python Scripts/Source/model_xgb_sw.py
```

When `CONFIG_PATH` is not defined, the scripts fall back to the corresponding
YAML file for the selected `TARGET_GP_NAME`, then to the built-in defaults.

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

Both model scripts log experiment metadata to MLflow when `mlflow_enabled` is
true. The default tracking directory is `Scripts/Results/mlruns`, which is
generated output and should not be committed.

Each run records:

- selected Grand Prix, feature lists, split ratios, window sizes, and random seed
- sliding-window RMSE, MAE, R2, residual standard deviation, and per-window values
- final sequential-holdout RMSE, MAE, R2, residual standard deviation, and confidence intervals
- `COS_MAE`, `COS_RMSE`, and their descriptive confidence intervals
- JSON artifacts for the resolved configuration, per-window results, and summary metrics

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

## `model_xgb_sw.py`

Runs XGBoost for the selected Grand Prix using:

- one-hot encoding with all categories retained
- median imputation for numerical predictors
- categorical encoding fitted only on each training split/window
- Optuna tuning when no saved parameters are available
- calibrated `n_estimators` from sliding-window early stopping
- final sequential holdout evaluation

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
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py --continue-on-error
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py --configs bahrain.yaml usa.yaml
```

The default run order is Bahrain, Saudi Arabia, United States, Italy, and
Hungary. Each subprocess receives `CONFIG_PATH` and `TARGET_GP_NAME`, so the
individual model scripts keep their normal configuration flow and MLflow logging.

## `modeling_utils.py`

Contains shared configuration loading, temporal splitting, one-hot alignment,
metric calculation, bootstrap confidence intervals, and COS reporting helpers.

## COS Metrics

Both scripts report:

```text
COS_MAE  = 0.5 * (MAE_SW / MAE_final)  + 0.5 * (STD_SW / STD_final)
COS_RMSE = 0.5 * (RMSE_SW / RMSE_final) + 0.5 * (STD_SW / STD_final)
```

The confidence intervals for COS are indicative because the sliding windows overlap.
