# Formula 1 Race-Pace Prediction

Research code and supplementary material for the paper *"Multi-Circuit Formula 1 Lap Time
Prediction: A Hybrid Deep Learning Approach for Race Pace Analysis"* (KDMiLe 2026). The
project models `LapTime_seconds` from public FastF1 race data with a temporal protocol that
mirrors a real race: expanding-window validation inside the modeling block and a final
sequential holdout on the last laps.

Three model families are compared:

- **Linear Regression** (expanding-window) — baseline;
- **XGBoost** (expanding-window, circuit-specific Optuna search space) — baseline;
- **`LSTM_hybrid`** — the selected model: an LR-EW baseline plus an LSTM that predicts the
  residual `LapTime_seconds − baseline_prediction` (final = `baseline + lstm_residual`),
  keeping a strong linear component while the LSTM captures the remaining complex structure.
  The LSTM uses a single sequential validation split rather than windows.

## Scope

Five Grand Prix events from the 2022–2025 technical-regulation period:

| Grand Prix | Circuit | Location |
|---|---|---|
| Bahrain Grand Prix | Bahrain International Circuit | Sakhir, Bahrain |
| Saudi Arabian Grand Prix | Jeddah Corniche Circuit | Jeddah, Saudi Arabia |
| United States Grand Prix | Circuit of the Americas | Austin, United States |
| Italian Grand Prix | Autodromo Nazionale Monza | Monza, Italy |
| Hungarian Grand Prix | Hungaroring | Mogyorod, Hungary |

## Documentation

| Document | Contents |
|---|---|
| [docs/data_pipeline.md](docs/data_pipeline.md) | Data provenance and the `prepare_data.py` cleaning step (`Data/` → `cleaned_data.csv`). |
| [docs/hyperparameters.md](docs/hyperparameters.md) | Full XGBoost & LSTM search spaces and final per-circuit hyperparameters (omitted from the paper for space). |
| [docs/paper_code_map.md](docs/paper_code_map.md) | Audit mapping each paper step to the code, plus the paper↔code divergences to resolve. |
| [docs/supplementary_analysis.md](docs/supplementary_analysis.md) | The complementary analysis/plot/table scripts (not part of the paper). |

## Repository Layout

```text
F1-MultiCircuit-LapTimeModel/
|- README.md
|- docs/                       # supplementary technical documentation
|- configs/                    # one YAML per circuit
|- Data/                       # raw FastF1 race CSVs (laps, weather, results)
|- Utils/                      # compounds.json, requirements.txt
|- Scripts/
|  |- ModelData/               # cleaned_data.csv per circuit (model input)
|  |- Source/
|  |  |- prepare_data.py        # Data/ -> cleaned_data.csv (reproducible cleaning)
|  |  |- model_lr_ew.py         # core: LR expanding-window
|  |  |- model_xgb_ew.py        # core: XGBoost expanding-window
|  |  |- model_lstm_hybrid.py   # core: LR-EW baseline + LSTM residual
|  |  |- model_lr_sw.py / model_xgb_sw.py / model_lstm_baseline.py   # extra (tested)
|  |  |- model_lr_ew_driver.py / model_xgb_ew_driver.py / model_lstm_driver.py  # driver-filtered baselines
|  |  |- modeling_utils.py / baseline_utils.py / xgb_utils.py        # shared helpers
|  |  |- run_experiment.py       # runner for all circuits
|  |  |- model_interpretability.py, plot_*.py, eda_*.py, ...  # analysis layer (see docs)
|  |- Results/                  # generated output (git-ignored)
```

Generated outputs, FastF1 caches, local PDFs and run logs are intentionally git-ignored.

## Data

`Data/<circuit>/Race/{Laps,Weather}/` holds the raw per-year race CSVs. The raw laps files
already carry `LapTime_seconds`, `pirelliCompound` and `IsAccurate` (the FastF1 collection and
Pirelli C1–C5 scraping happened upstream — see [docs/data_pipeline.md](docs/data_pipeline.md)).
`prepare_data.py` turns these into the cleaned, feature-engineered datasets in
`Scripts/ModelData/`, which every modeling script consumes.

## Models

**Core experiment** — `model_lr_ew.py`, `model_xgb_ew.py`, `model_lstm_hybrid.py`. XGBoost runs
an independent Optuna study per fold and aggregates hyperparameters by median across folds. The
hybrid uses an out-of-fold expanding-window LR-EW baseline (no leakage; see `baseline_utils.py`)
and sweeps `lstm_window_ratio`, keeping the best by validation RMSE.

**Extra models (tested, kept for comparison)** — `model_lr_sw.py`, `model_xgb_sw.py` (sliding
window), and `model_lstm_baseline.py` (baseline = previous lap time; LSTM learns
`LapTime_seconds − LapTime_prev`).

**Driver-filtered baselines (methodological sensitivity)** — `model_lr_ew_driver.py`,
`model_xgb_ew_driver.py`, `model_lstm_driver.py` filter to a single driver before the temporal
split. They write to dedicated `*_driver` result subdirectories and never overwrite or change
the core models' reported results.

Encoders, imputers and scalers are fit only on each split's training portion, transforming
validation/holdout with the training columns to avoid categorical leakage.

## Validation & Metrics

- **Sequential holdout:** the last 20% of laps (ordered `Year → LapNumber`) are reserved and
  never used for tuning/selection.
- **Expanding window (EW):** inside the first 80% modeling block, training grows cumulatively;
  each fold validates the next fixed-size chunk. EW was selected for every circuit and model.
- **Metrics:** per-fold RMSE/MAE/R²/residual STD; sequential-holdout RMSE/MAE/R² with bootstrap
  95% CIs (`calc_holdout_ci`); and the **COS** stability indicator (`α=β=0.5`):

  ```text
  COS_MAE  = 0.5 * (MAE_holdout  / MAE_windows)  + 0.5 * (STD_holdout / STD_windows)
  COS_RMSE = 0.5 * (RMSE_holdout / RMSE_windows) + 0.5 * (STD_holdout / STD_windows)
  ```

  COS ≈ 1 means in-window and holdout behave alike. (Note: the paper's Eq. 1 writes this ratio
  inverted; the code/Table II use the `holdout/windows` orientation above — see
  [docs/paper_code_map.md](docs/paper_code_map.md), D2.)

Selected per-circuit window ratios from the 5%–50% sweep:

| Grand Prix | LR-EW window | XGB-EW window |
|---|---|---|
| Bahrain | 5% | 30% |
| Saudi Arabian | 10% | 50% |
| United States | 45% | 5% |
| Italian | 5% | 50% |
| Hungarian | 45% | 40% |

When `mlflow_enabled` is true, runs are logged to `Scripts/Results/mlruns` (git-ignored).

## Installation

Reference environment: **Python 3.12** (tested on 3.12.3, Linux). The versions in
`Utils/requirements.txt` are pinned for reproducibility — using a different Python
version may fail to install the pinned `numpy`/`tensorflow` wheels, so prefer 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r Utils/requirements.txt
```

### Reproducibility

With the pinned versions above and the seeds in each `configs/*.yaml`
(`random_seed: 42`), the LR-EW and XGBoost-EW results reproduce bit-for-bit
(verified: identical Optuna/TPE search and XGBoost training). The pipeline reuses
cached hyperparameters (`Scripts/Results/.../*_params_ew.json`) when present;
delete them to re-run the Optuna tuning from scratch. The LSTM/hybrid models
(TensorFlow) may vary slightly on GPU, since the pipeline does not enable
deterministic GPU ops.

## Running

Pick a circuit with `CONFIG_PATH` (a YAML) or `TARGET_GP_NAME`, then run from the repository
root (paths are case-sensitive on Linux/macOS). Only the way you set the environment variable
differs between shells; the `python ...` commands below are identical on both.

Linux/macOS:

```bash
export CONFIG_PATH="configs/bahrain.yaml"
```

Windows/PowerShell:

```powershell
$env:CONFIG_PATH = "configs/bahrain.yaml"
```

Then (same on both):

```bash
# (re)build the cleaned dataset for this circuit
python Scripts/Source/prepare_data.py

# core experiment
python Scripts/Source/model_lr_ew.py
python Scripts/Source/model_xgb_ew.py
python Scripts/Source/model_lstm_hybrid.py
```

Reproduce the full reported experiment (LR-EW + XGBoost-EW for every circuit; add
`--with-hybrid` to also run `LSTM_hybrid`):

```bash
python Scripts/Source/run_experiment.py
python Scripts/Source/run_experiment.py --circuit bahrain italy
python Scripts/Source/run_experiment.py --with-hybrid
```

Driver-filtered baselines and analysis scripts take a `--driver` code, e.g.:

```bash
python Scripts/Source/model_lr_ew_driver.py --driver VER
python Scripts/Source/plot_driver_holdout_timeseries.py --driver VER
```

Inspect runs with the MLflow UI:

```bash
python -m mlflow ui --backend-store-uri Scripts/Results/mlruns
```

Supported `TARGET_GP_NAME`: `Bahrain Grand Prix`, `Saudi Arabian Grand Prix`,
`United States Grand Prix`, `Italian Grand Prix`, `Hungarian Grand Prix`.

## Features

- **Numerical:** `TyreLife`, `LapNumber`, `Humidity_RBF_Median`, `Pressure_RBF_Median`,
  `TrackTemp_RBF_Median`, `WindSpeed_RBF_Median`, `TempDelta_RBF_Median`, `Year`, `LapTime_prev`.
- **Categorical:** `Driver`, `Team`, `pirelliCompound`. **Target:** `LapTime_seconds`.

Per-circuit feature exclusions (from correlation + PCA-loading analysis) are configured in
`configs/*.yaml`: USA drops `Year` + `TempDelta_RBF_Median` (keeping `TrackTemp_RBF_Median`);
Saudi drops `TrackTemp_RBF_Median`; Hungary drops `Humidity_RBF_Median`; Bahrain and Italy keep
all. The paper §3.1 text mis-states the USA case (it says TrackTemp instead of TempDelta) — a
wording fix tracked as D1 in [docs/paper_code_map.md](docs/paper_code_map.md).

## Authors

Marcos Paulo de Oliveira Pereira · Carlos Henrique Gomes Ferreira · Alexandre Magno de Sousa
— Universidade Federal de Ouro Preto (UFOP)
