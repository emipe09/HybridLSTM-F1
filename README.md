# Formula 1 Race-Pace Prediction

This repository contains the current research code and notebooks for multi-circuit Formula 1 lap-time prediction. The project uses public FastF1-derived race data to model `LapTime_seconds` with a temporal protocol that mirrors a real race: sliding-window validation inside the modeling segment and a final sequential holdout on the last laps.

## Scope

The current version focuses on five Grand Prix events from the 2022-2025 technical-regulation period:

| Grand Prix | Circuit | Location |
|---|---|---|
| Bahrain Grand Prix | Bahrain International Circuit | Sakhir, Bahrain |
| Saudi Arabian Grand Prix | Jeddah Corniche Circuit | Jeddah, Saudi Arabia |
| United States Grand Prix | Circuit of the Americas | Austin, United States |
| Italian Grand Prix | Autodromo Nazionale Monza | Monza, Italy |
| Hungarian Grand Prix | Hungaroring | Mogyorod, Hungary |

## Repository Layout

```text
TCC/
|- Data/
|  |- Bahrain/
|  |- Hungary/
|  |- Italy/
|  |- Saudi Arabia/
|  |- United States/
|- Scripts/
|  |- ModelData/
|  |- Notebooks/
|  |- Source/
|     |- backward_elimination.py
|     |- correlation_ablation_lr.py
|     |- model_lr_sw.py
|     |- model_xgb_sw.py
|     |- modeling_utils.py
|- configs/
|  |- bahrain.yaml
|  |- saudi.yaml
|  |- usa.yaml
|  |- italy.yaml
|  |- hungary.yaml
|- Utils/
|  |- compounds.json
|  |- requirements.txt
|- README.md
```

Generated outputs, FastF1 caches, local PDFs, notebook plot folders, XGBoost parameter dumps, and historical run logs are intentionally ignored by Git.

## Data

`Data/` stores raw race-session CSV files by circuit:

- race laps
- race weather
- race results

The modeling scripts run from cleaned datasets in `Scripts/ModelData/`. Those files contain the article-facing engineered data used by the notebooks and by the reproducible scripts in `Scripts/Source/`.

## Notebooks

The notebooks in `Scripts/Notebooks/` are the full circuit-specific analyses:

| Notebook | Circuit |
|---|---|
| `Notebook_Bahrain.ipynb` | Bahrain Grand Prix |
| `Notebook_Saudi.ipynb` | Saudi Arabian Grand Prix |
| `Notebook_USA.ipynb` | United States Grand Prix |
| `Notebook_Italia.ipynb` | Italian Grand Prix |
| `Notebook_Hungary.ipynb` | Hungarian Grand Prix |

Each notebook is written in English and follows the same structure: data preparation, exploratory analysis, feature engineering, Linear Regression, XGBoost, sliding-window validation, sequential holdout, and COS metrics.

## Modeling Scripts

The reproducible modeling and feature-selection scripts are kept in `Scripts/Source/`:

- `model_lr_sw.py`: Linear Regression with median imputation, standard scaling, sliding-window validation, and sequential holdout.
- `model_xgb_sw.py`: XGBoost with regularized Optuna hyperparameter tuning, sliding-window validation, and sequential holdout.
- `model_interpretability.py`: unified interpretability runner that loads the saved Linear Regression and XGBoost models, then exports LR coefficients, XGBoost feature importance, XGBoost SHAP values, and a local SHAP force plot.
- `backward_elimination.py`: p-value based backward elimination for the Linear Regression design matrix, fitted only on the first sequential modeling block.
- `correlation_ablation_lr.py`: Linear Regression ablation runner that detects encoded-feature pairs with `|r| >= 0.80` inside the modeling block, then reruns the full sliding-window and sequential-holdout protocol after removing one feature from each pair.
- `regression_diagnostics_lr.py`: Linear Regression residual-diagnostics runner that fits the model on the sequential modeling block and generates in-sample diagnostics for that retrained 80% block while keeping the final holdout unused.
- `run_all_models.py`: batch runner for executing both model scripts across all configured Grand Prix events.
- `modeling_utils.py`: shared configuration, temporal split, encoding, metric, confidence interval, COS, and MLflow tracking helpers.

Categorical encoders, imputers and scalers are fitted only on the training
portion of each split/window. Validation and final holdout records are
transformed using the training columns only, avoiding categorical leakage from
future laps.

Both scripts report:

- sliding-window RMSE, MAE, R2, and residual standard deviation
- sequential-holdout RMSE, MAE, and R2 with bootstrap confidence intervals
- `COS_MAE` and `COS_RMSE` with indicative 95% confidence intervals

The COS metrics are computed as:

```text
COS_MAE  = 0.5 * (MAE_final / MAE_SW)  + 0.5 * (STD_final / STD_SW)
COS_RMSE = 0.5 * (RMSE_final / RMSE_SW) + 0.5 * (STD_final / STD_SW)
```

The COS confidence intervals are descriptive because the sliding windows overlap.

When `mlflow_enabled` is true, both model scripts also log each run to MLflow
with the selected Grand Prix, feature lists, split/window settings,
sliding-window metrics, holdout metrics, COS metrics, and JSON artifacts for the
configuration and per-window results. XGBoost runs also log the generated
parameter JSON when it exists. The default local tracking directory is
`Scripts/Results/mlruns`, which is treated as generated output.

Detailed generated-result documentation is maintained in
`Scripts/Results/README.md`, including the latest saved XGBoost final
hyperparameters, the best validation window for each Grand Prix, and the
per-window Optuna-selected hyperparameters with seed, sampler and trial count.

## Installation

Create and activate a virtual environment before installing dependencies.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r Utils/requirements.txt
```

Windows/PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r Utils/requirements.txt
```

If you do not use a virtual environment, install dependencies with the same
Python interpreter that will run the scripts:

```bash
python -m pip install -r Utils/requirements.txt
```

## Running a Model

Select a configuration file directly.

Linux/macOS:

```bash
export CONFIG_PATH="configs/bahrain.yaml"
python Scripts/Source/model_lr_sw.py
python Scripts/Source/model_xgb_sw.py
python Scripts/Source/model_interpretability.py
python Scripts/Source/backward_elimination.py
python Scripts/Source/correlation_ablation_lr.py
python Scripts/Source/regression_diagnostics_lr.py
```

Windows/PowerShell:

```powershell
$env:CONFIG_PATH = "configs/bahrain.yaml"
.\.venv\Scripts\python.exe Scripts/Source/model_lr_sw.py
.\.venv\Scripts\python.exe Scripts/Source/model_xgb_sw.py
.\.venv\Scripts\python.exe Scripts/Source/model_interpretability.py
.\.venv\Scripts\python.exe Scripts/Source/backward_elimination.py
.\.venv\Scripts\python.exe Scripts/Source/correlation_ablation_lr.py
.\.venv\Scripts\python.exe Scripts/Source/regression_diagnostics_lr.py
```

The YAML files control the Grand Prix name, target column, lap column, feature
lists, validation ratios, COS coefficients, random seed, Optuna settings,
Grand Prix-specific XGBoost search-space bounds, and MLflow tracking settings,
and directory/file paths such as `model_data_dir`,
`results_dir`, and the cleaned dataset filename template. Relative paths are
resolved from the repository root.

To inspect the experiment history locally, start the MLflow UI from the
repository root after running at least one model:

Linux/macOS:

```bash
python -m mlflow ui --backend-store-uri Scripts/Results/mlruns
```

Windows/PowerShell:

```powershell
python -m mlflow ui --backend-store-uri Scripts/Results/mlruns
```

Then open the URL printed by MLflow, usually `http://127.0.0.1:5000`.

Alternatively, select a Grand Prix directly:

Linux/macOS:

```bash
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_lr_sw.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_xgb_sw.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_interpretability.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/backward_elimination.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/correlation_ablation_lr.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/regression_diagnostics_lr.py
```

Windows/PowerShell:

```powershell
$env:TARGET_GP_NAME = "Bahrain Grand Prix"
.\.venv\Scripts\python.exe Scripts/Source/model_lr_sw.py
.\.venv\Scripts\python.exe Scripts/Source/model_xgb_sw.py
.\.venv\Scripts\python.exe Scripts/Source/model_interpretability.py
.\.venv\Scripts\python.exe Scripts/Source/backward_elimination.py
.\.venv\Scripts\python.exe Scripts/Source/correlation_ablation_lr.py
.\.venv\Scripts\python.exe Scripts/Source/regression_diagnostics_lr.py
```

On Linux/macOS, paths are case-sensitive. Run commands from the repository root
and keep directory names exactly as shown, for example `Scripts/Source/` rather
than `scripts/source/`.

To run all configured Grand Prix events and both model families in sequence:

Linux/macOS:

```bash
python Scripts/Source/run_all_models.py
```

Windows/PowerShell:

```powershell
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py
```

You can limit the batch run to one model family:

Linux/macOS:

```bash
python Scripts/Source/run_all_models.py --models lr
python Scripts/Source/run_all_models.py --models xgb
```

Windows/PowerShell:

```powershell
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py --models lr
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py --models xgb
```

If an XGBoost parameter file is not available for a circuit, or if the saved
parameters do not match the current search-space version, tuning strategy, YAML
bounds, or sampler, the XGBoost script will run Optuna before
training, so the full batch may take substantially longer. The current XGBoost
tuning protocol runs an independent Optuna study for each sliding window inside
the first 80% modeling block; `optuna_trials` is therefore interpreted as trials
per window. Each study minimizes validation RMSE. Final hyperparameters are the
median of the best Optuna parameters selected in all sliding windows, with
integer parameters rounded to the nearest integer. Final `n_estimators` is the
median early-stopping iteration across those same windows. The untouched
sequential holdout is evaluated only after this selection is complete.
XGBoost searches also export a
per-trial CSV with the source window for auditability.
Backward-elimination outputs are generated under
`Scripts/Results/backward_elimination/` and are ignored by Git.
To run backward elimination for every configured Grand Prix:

Linux/macOS:

```bash
python Scripts/Source/backward_elimination.py --all
```

Windows/PowerShell:

```powershell
.\.venv\Scripts\python.exe Scripts/Source/backward_elimination.py --all
```

Correlation-ablation outputs are generated under
`Scripts/Results/correlation_ablation_lr/` and are ignored by Git. To run the
correlated-feature ablation for every configured Grand Prix:

Linux/macOS:

```bash
python Scripts/Source/correlation_ablation_lr.py --all
```

Windows/PowerShell:

```powershell
.\.venv\Scripts\python.exe Scripts/Source/correlation_ablation_lr.py --all
```

Regression-diagnostics outputs are generated under
`Scripts/Results/regression_diagnostics/` and are ignored by Git. The diagnostic
script preserves the final sequential holdout: preprocessing and the Linear
Regression model are fitted only on the first modeling block, then residual
plots, prediction tables, a standard statsmodels OLS summary, and coefficient
plots are produced for that retrained 80% modeling block. The final 20% holdout
remains unused by the diagnostic script and is reserved for final model
evaluation. To run the diagnostics for every configured Grand Prix:

Linux/macOS:

```bash
python Scripts/Source/regression_diagnostics_lr.py --all
```

Windows/PowerShell:

```powershell
.\.venv\Scripts\python.exe Scripts/Source/regression_diagnostics_lr.py --all
```

Supported `TARGET_GP_NAME` values are:

- `Bahrain Grand Prix`
- `Saudi Arabian Grand Prix`
- `United States Grand Prix`
- `Italian Grand Prix`
- `Hungarian Grand Prix`

## Key Features

Numerical predictors:

- `TyreLife`
- `LapNumber`
- `Humidity_RBF_Median`
- `Pressure_RBF_Median`
- `TrackTemp_RBF_Median`
- `WindSpeed_RBF_Median`
- `TempDelta_RBF_Median`
- `Year`
- `LapTime_prev`

Categorical predictors:

- `Driver`
- `Team`
- `pirelliCompound`

Target:

- `LapTime_seconds`

Feature lists are configured per Grand Prix in `configs/*.yaml`. After the
correlation and PCA-loading analysis, the following circuit-specific exclusions
were applied to reduce strong redundancy while preserving the most physically
interpretable variable in each correlated group:

| Grand Prix | Removed variable(s) | Retained correlated variable | Rationale |
|---|---|---|---|
| United States Grand Prix | `TempDelta_RBF_Median`, `Year` | `TrackTemp_RBF_Median` | The COTA layout combines high-speed direction changes, heavy braking, and traction-demanding exits, making track temperature a direct proxy for tire grip, thermal degradation, and operating-window effects. PCA loadings and correlation patterns supported keeping the real-time thermal condition instead of the derived temperature delta and year trend. |
| Saudi Arabian Grand Prix | `TrackTemp_RBF_Median` | `Pressure_RBF_Median` | At Jeddah, the selected correlated structure favored pressure as the broader atmospheric-state proxy. Pressure can reflect weather-density conditions that affect aerodynamic behavior and engine response, while avoiding redundant thermal information already captured through the retained predictors. |
| Hungarian Grand Prix | `Humidity_RBF_Median` | `TrackTemp_RBF_Median` | The Hungaroring is traction-limited and tire-energy sensitive, so track temperature has a clearer physical link to grip, overheating risk, and degradation than humidity in the retained correlated group. PCA loadings supported prioritizing the track-surface thermal condition. |

These exclusions do not change the temporal validation protocol. They only
adjust the configured feature set used by the affected circuit models.

## Reproducibility Notes

- The final 20% of race laps is reserved as a sequential holdout.
- Sliding-window validation is performed only inside the first 80% modeling block.
- XGBoost parameter files are generated under `Scripts/Results/` when needed and are ignored by Git.
- MLflow run metadata is generated under `Scripts/Results/mlruns/` by default and is ignored by Git.
- The notebooks remain the narrative, circuit-specific record of the analysis; the scripts are the lean reproducible runners for GitHub.

## Authors

- Marcos Paulo de Oliveira Pereira
- Carlos Henrique Gomes Ferreira
- Alexandre Magno de Sousa

Universidade Federal de Ouro Preto (UFOP)
