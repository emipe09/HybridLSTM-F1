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

The modeling scripts run from cleaned datasets in `Scripts/ModelData/`. Those files contain the article-facing engineered data used by the notebooks and by the two scripts in `Scripts/Source/`.

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

Only the current sliding-window scripts are kept in `Scripts/Source/`:

- `model_lr_sw.py`: Linear Regression with median imputation, standard scaling, sliding-window validation, and sequential holdout.
- `model_xgb_sw.py`: XGBoost with Optuna hyperparameter tuning, sliding-window validation, and sequential holdout.
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
COS_MAE  = 0.5 * (MAE_SW / MAE_final)  + 0.5 * (STD_SW / STD_final)
COS_RMSE = 0.5 * (RMSE_SW / RMSE_final) + 0.5 * (STD_SW / STD_final)
```

The COS confidence intervals are descriptive because the sliding windows overlap.

When `mlflow_enabled` is true, both model scripts also log each run to MLflow
with the selected Grand Prix, feature lists, split/window settings,
sliding-window metrics, holdout metrics, COS metrics, and JSON artifacts for the
configuration and per-window results. XGBoost runs also log the generated
parameter JSON when it exists. The default local tracking directory is
`Scripts/Results/mlruns`, which is treated as generated output.

## Installation

```bash
pip install -r Utils/requirements.txt
```

On Windows/PowerShell, prefer installing with the same Python interpreter that
will run the scripts:

```powershell
python -m pip install -r Utils/requirements.txt
```

## Running a Model

PowerShell:

```powershell
$env:CONFIG_PATH = "configs/bahrain.yaml"
python Scripts/Source/model_lr_sw.py
python Scripts/Source/model_xgb_sw.py
```

The YAML files control the Grand Prix name, target column, lap column, feature
lists, validation ratios, COS coefficients, random seed, Optuna settings, and
MLflow tracking settings, and directory/file paths such as `model_data_dir`,
`results_dir`, and the cleaned dataset filename template. Relative paths are
resolved from the repository root.

To inspect the experiment history locally, start the MLflow UI from the
repository root after running at least one model:

```powershell
python -m mlflow ui --backend-store-uri Scripts/Results/mlruns
```

Then open the URL printed by MLflow, usually `http://127.0.0.1:5000`.

Alternatively, select a Grand Prix directly:

```powershell
$env:TARGET_GP_NAME = "Bahrain Grand Prix"
python Scripts/Source/model_lr_sw.py
python Scripts/Source/model_xgb_sw.py
```

Bash:

```bash
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_lr_sw.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_xgb_sw.py
```

To run all configured Grand Prix events and both model families in sequence:

```powershell
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py
```

You can limit the batch run to one model family:

```powershell
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py --models lr
.\.venv\Scripts\python.exe Scripts/Source/run_all_models.py --models xgb
```

If an XGBoost parameter file is not available for a circuit, the XGBoost script
will run Optuna before training, so the full batch may take substantially longer.

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
- `LapTime_prev`

Categorical predictors:

- `Driver`
- `Team`
- `pirelliCompound`
- `Year`

Target:

- `LapTime_seconds`

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
