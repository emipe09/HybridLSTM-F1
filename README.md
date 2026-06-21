# Formula 1 Race-Pace Prediction

This repository contains the current research code and notebooks for multi-circuit Formula 1 lap-time prediction. The project uses public FastF1-derived race data to model `LapTime_seconds` with a temporal protocol that mirrors a real race: window-based validation (sliding-window and expanding-window) inside the modeling segment and a final sequential holdout on the last laps.

The final model comparison covers three model families:

- **Linear Regression** (expanding-window validation);
- **XGBoost** (expanding-window validation, circuit-specific Optuna search space);
- **`LSTM_hybrid`** — the selected third model: Linear Regression (LR-EW) as the baseline plus an LSTM that predicts the residual. The methodology deliberately keeps a strong linear component (LR-EW) and lets the LSTM capture the remaining complex relationships through the baseline residual. The LSTM component uses a single sequential validation split rather than sliding/expanding windows.

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
|     |- model_lr_ew.py            # core: LR expanding-window
|     |- model_xgb_ew.py           # core: XGBoost expanding-window
|     |- model_lstm_hybrid.py      # core: LR-EW baseline + LSTM residual
|     |- model_lr_sw.py            # extra: LR sliding-window
|     |- model_xgb_sw.py           # extra: XGBoost sliding-window
|     |- model_lstm_baseline.py    # extra: previous-lap baseline + LSTM residual
|     |- model_interpretability.py
|     |- plot_driver_holdout_timeseries.py
|     |- run_experiment.py
|     |- modeling_utils.py
|     |- baseline_utils.py
|     |- xgb_utils.py
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

Each notebook is written in English and follows the same structure: data preparation, exploratory analysis, feature engineering, Linear Regression, XGBoost, sliding-window validation, sequential holdout, COS metrics, and an expanding-window validation section that compares EW and SW fold-level metrics side by side.

## Modeling Scripts

The reproducible modeling scripts are kept in `Scripts/Source/`. The experiment is
built around three core models, plus two extra models retained because they were
tested during the research.

**Core experiment**

- `model_lr_ew.py`: Linear Regression with expanding-window validation and sequential holdout. The training set grows cumulatively across folds; the validation chunk is the same fixed size as the SW validation portion.
- `model_xgb_ew.py`: XGBoost with expanding-window validation and sequential holdout. Runs an independent Optuna study per fold and aggregates hyperparameters by median across all folds.
- `model_lstm_hybrid.py`: the selected `LSTM_hybrid` model. By design it uses Linear Regression (LR-EW) as the tabular expanding-window baseline so the model keeps a strong linear component, and the LSTM is trained to predict the residual `LapTime_seconds - baseline_prediction` to capture the remaining complex relationships; the final prediction is `baseline_prediction + lstm_residual_prediction`. The baseline predictions come from an out-of-fold expanding-window series over the modeling block (never the holdout). It reuses the LSTM core from `model_lstm_baseline.py` unchanged and sweeps `lstm_window_ratio` values, keeping the best by validation RMSE.

**Extra models (tested, kept for comparison)**

- `model_lr_sw.py`: Linear Regression with median imputation, standard scaling, sliding-window validation, and sequential holdout.
- `model_xgb_sw.py`: XGBoost with regularized Optuna hyperparameter tuning, sliding-window validation, and sequential holdout.
- `model_lstm_baseline.py`: the **baseline-LapTime_prev LSTM**. Its baseline is the driver's **previous lap time**: the network learns the residual `LapTime_seconds - LapTime_prev` (`lstm_target_mode = residual_from_laptime_prev`) and the final prediction is `LapTime_prev + lstm_residual_prediction`. **`LapTime_prev` is not a network input feature** — the configured `lstm_feature_mode` is `auxiliary_embedding`, which drops `LapTime_prev` while keeping Driver/Team embeddings; the previous-lap signal enters only through the residual target. Uses a single sequential split, Optuna tuning with `EarlyStopping`, retraining on the full modeling block, and the untouched holdout for final evaluation.

**Analysis, figures, and runner**

- `model_interpretability.py`: unified interpretability runner that loads the saved Linear Regression and XGBoost models, then exports LR coefficients, XGBoost feature importance, XGBoost SHAP values, and a local SHAP force plot.
- `plot_driver_holdout_timeseries.py`: plots a single driver's actual vs predicted lap-time series over the sequential holdout (LR or XGBoost final model) with an approximate 95% prediction band.
- `run_experiment.py`: runs the final per-circuit experiment (LR-EW + XGBoost-EW, plus `LSTM_hybrid` with `--with-hybrid`) for all circuits using the window ratios encoded in each YAML.

**Shared utilities**

- `modeling_utils.py`: shared configuration, temporal split, encoding, metric, confidence interval, COS, and MLflow tracking helpers. Includes `build_expanding_windows()` and path builders for EW artifacts.
- `baseline_utils.py`: shared helpers for the hybrid tabular baseline (out-of-fold expanding-window predictions, block predictions, saved-prediction reuse, and XGBoost-EW hyperparameter resolution) with leakage control.
- `xgb_utils.py`: shared XGBoost utilities (search-space definitions, Optuna integration, DMatrix construction, parameter aggregation) used by both SW and EW XGBoost scripts.

Categorical encoders, imputers and scalers are fitted only on the training
portion of each temporal split. Validation and final holdout records are
transformed using the training columns only, avoiding categorical leakage from
future laps.

### Validation Protocols

**Sliding-window (SW)**: a fixed-length window slides across the modeling block in steps equal to the validation portion. Training and validation subsets do not grow across windows.

**Expanding-window (EW)**: the training set grows cumulatively. Fold `k` trains on all laps from the start through the end of fold `k-1`'s validation chunk and validates on the next fixed-size chunk. The initial training size and validation chunk size match the first SW window. Confidence intervals under the EW protocol are descriptive because the training sets are not independent across folds.

The Linear Regression and XGBoost scripts (SW and EW variants) report:

- per-window/fold RMSE, MAE, R2, and residual standard deviation
- mean and indicative 95% confidence intervals across windows/folds
- sequential-holdout RMSE, MAE, and R2 with bootstrap confidence intervals
- `COS_MAE` and `COS_RMSE` with indicative 95% confidence intervals

The COS metrics are computed as:

```text
COS_MAE  = 0.5 * (MAE_SW_or_EW / MAE_final)  + 0.5 * (STD_SW_or_EW / STD_final)
COS_RMSE = 0.5 * (RMSE_SW_or_EW / RMSE_final) + 0.5 * (STD_SW_or_EW / STD_final)
```

The COS confidence intervals under SW are descriptive because the sliding windows overlap; under EW they are descriptive because the expanding training sets have growing sizes and are correlated across folds.
When `mlflow_enabled` is true, the model scripts also log each run to MLflow
with the selected Grand Prix, feature lists, split settings, validation metrics,
holdout metrics, and JSON artifacts for the resolved configuration and summary
results. XGBoost runs also log the generated parameter JSON when it exists. The default local tracking directory is
`Scripts/Results/mlruns`, which is treated as generated output.

### LSTM and LSTM_hybrid

The third model family uses an LSTM. Unlike LR and XGBoost, the LSTM does **not**
use sliding/expanding windows; it uses a **single sequential split** because each
`(Year, Driver)` group contributes only a small pool of sequences after windowing,
and multiple folds would fragment that pool and multiply training cost.

`model_lstm_hybrid.py` is the selected `LSTM_hybrid` model (LR-EW baseline + LSTM
residual). `model_lstm_baseline.py` is the extra **baseline-LapTime_prev LSTM**: its
baseline is the previous lap time, it learns the residual
`LapTime_seconds - LapTime_prev`, and **it does not use `LapTime_prev` as a network
feature** (`lstm_feature_mode = auxiliary_embedding`). Run them from the repository root:

```bash
# Linux / macOS
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_lstm_hybrid.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_lstm_baseline.py
```

```powershell
# Windows / PowerShell
$env:TARGET_GP_NAME = "Bahrain Grand Prix"
python Scripts/Source/model_lstm_hybrid.py
python Scripts/Source/model_lstm_baseline.py
```

**How `LSTM_hybrid` fits the methodology**

- **Temporal ordering.** The dataset is ordered by `Year`, then by `LapNumber`
  within each year (all 2022 laps precede 2023, then 2024, then 2025).
- **80/20 sequential holdout.** The first 80% of the ordered rows form the
  modeling block; the last 20% are the sequential holdout, which is never touched
  until final evaluation.
- **Single sequential validation split.** Inside the modeling block, the first
  `window_train_ratio` (0.80) of laps trains and the remaining laps validate.
  Optuna tunes the LSTM on this single split and `EarlyStopping` on `val_loss`
  calibrates the epoch count; the final model is retrained on the full modeling
  block for `max(median_optuna_epochs, lstm_min_final_epochs)` epochs.
- **Sequence construction.** Sequences are grouped by `lstm_group_cols`
  (`[Year, Driver]`) and sorted by `LapNumber`. For each target lap, the
  `sequence_length` immediately preceding laps of the same group form the input,
  where `sequence_length = ceil(n_race_laps * lstm_window_ratio)`. The hybrid
  sweeps a list of `lstm_window_ratio` values and keeps the best by validation
  RMSE (never the holdout).
- **Hybrid target.** The methodology deliberately uses Linear Regression (LR-EW)
  as the tabular baseline to give the model a strong linear component. This
  baseline predicts the lap time, the LSTM learns the residual
  `LapTime_seconds - baseline_prediction` to capture the remaining complex
  relationships, and the final prediction is
  `baseline_prediction + lstm_residual_prediction`.
- **Preprocessing.** Median imputation → `StandardScaler` on features, one-hot
  encoding (full rank), and a separate `StandardScaler` on the target. All
  transformers are fit on the training portion only.
- **Leakage prevention.** The tabular baseline used as the residual target is an
  out-of-fold (expanding-window) series over the modeling block, so every baseline
  value is produced by a model trained only on earlier laps. The holdout is never
  used to train or select the baseline, and no transformer is fit on validation or
  holdout data.
- **Metrics reported.** Validation split: RMSE, MAE, R², residual STD. Sequential
  holdout: RMSE, MAE, R², each with a 95% bootstrap confidence interval. Plus
  `COS_MAE` and `COS_RMSE` (the validation split plays the SW/EW role in the COS
  formula). The standalone tabular baseline's own holdout metrics are also logged
  for comparison.

The selected per-circuit `LSTM_hybrid` hyperparameters are documented in
[LSTM_hybrid Selected Hyperparameters](#lstm_hybrid-selected-hyperparameters).

## Window-Size Sensitivity Analysis and Circuit-Specific Model Selection

After implementing both the sliding-window (SW) and expanding-window (EW) protocols for Linear Regression (LR) and XGBoost (XGB), a window-size sensitivity sweep was conducted across all five circuits. The sweep evaluated window ratios from 5% to 55% in 5% increments and produced per-window validation metrics together with sequential-holdout performance. Selection criteria focused primarily on **holdout performance** and the **COS metric** (coefficient of stability), which captures how well the in-window error generalizes to the final holdout block. A COS close to 1.0 indicates stable generalization; values much greater than 1.0 signal that the model degrades significantly on the unseen holdout laps.

### Key Findings by Model and Protocol

**Linear Regression – Sliding Window (LR-SW):**  
SW proved impractical for Linear Regression across all five circuits. The fixed-size training window provides insufficient context for a linear model, leading to consistently high COS values and poor holdout generalization. LR-SW results were dismissed as a viable final configuration.

**Linear Regression – Expanding Window (LR-EW):**  
EW dramatically improved LR stability. As the training set grows cumulatively, the linear model accumulates more context and generalizes better. Multiple window sizes were competitive, but the selection focused on the best combination of holdout error and COS. The optimal window size varied by circuit character.

**XGBoost – Sliding Window (XGB-SW):**  
SW with XGBoost yielded consistent in-window performance across most circuits, with low per-fold RMSE. However, COS values sometimes signalled that holdout error rose relative to in-window error. For circuits where SW was selected over EW, the chosen window size balanced per-fold accuracy with the best holdout generalization.

**XGBoost – Expanding Window (XGB-EW):**  
EW with XGBoost sometimes improved holdout stability relative to SW, especially at larger window sizes where the model is trained on a substantial portion of the race. However, at very small window ratios (e.g., 5%), EW folds are short, in-window error is low, but the COS often rose — indicating the model was fitting narrow segments without generalizing to the tail of the race. The selection therefore balanced per-fold error and COS rather than minimizing in-window RMSE alone.

### Selected Configurations per Circuit

The table below records the final per-circuit model and window-ratio choices.
**Expanding-window (EW) validation was selected for every circuit and every
model**; no circuit uses a sliding-window (SW) result as its final reported
configuration. These values are encoded as `lr_ew_window_ratio` and
`xgb_ew_window_ratio` in each circuit's YAML configuration file. The third model,
`LSTM_hybrid`, uses Linear Regression (LR-EW) as its tabular baseline plus an
LSTM residual; see its own hyperparameter section below.

| Grand Prix | Best LR approach | LR window | Best XGB approach | XGB window |
|---|---|---|---|---|
| Bahrain Grand Prix | LR-EW | 5% | XGB-EW | 30% |
| Hungarian Grand Prix | LR-EW | 45% | XGB-EW | 40% |
| Italian Grand Prix | LR-EW | 5% | XGB-EW | 50% |
| Saudi Arabian Grand Prix | LR-EW | 10% | XGB-EW | 50% |
| United States Grand Prix | LR-EW | 45% | XGB-EW | 5% |

**Bahrain:** LR-EW with 5% was the clear winner — in-window error was the lowest of all EW window sizes and COS was well-behaved. For XGB, EW at 30% offered a good balance: per-fold performance was strong and holdout generalization was better than the 45% window, which had the best in-window scores but generalized slightly worse.

**Hungary:** LR-EW required a large window (45%) to achieve stable holdout results; smaller windows produced poor COS values in this circuit. For XGB, EW at 40% was the most consistent option, with low per-fold error and a COS indicating it maintained performance on the holdout.

**Italy:** LR-EW performed well with a small 5% window, which stood out clearly among all EW options. For XGB, EW at 50% gave the best balance of per-fold error and holdout stability, training the model on a substantial portion of the race.

**Saudi Arabia:** LR-EW with 10% was the best performer and showed the lowest COS among EW sizes, meaning strong and stable generalization to the holdout. XGB-EW at 50% was selected because the EW COS values at large window sizes were the most favourable; the XGB-EW 50% window delivered good in-fold error and maintained holdout performance.

**USA:** LR-EW at 45% was the only window where LR produced acceptable holdout results and COS; smaller windows had poor generalization. For XGB, EW at 5% had notably lower in-window error and a better COS than other sizes, making it the clear choice despite the very small window size.

## Generated XGBoost Hyperparameter Tables

The tables below document the **final, reported XGBoost configuration**, which uses
expanding-window (EW) validation for every circuit. The selected hyperparameters
are the saved artifacts from `Scripts/Results/xgboost/ew/params/`. They are kept in
this root README because `Scripts/Results/` is treated as generated output and is
normally ignored by Git. These values are reproducibility detail, not source-code
defaults; if a Grand Prix configuration, search space, sampler, or saved parameter
JSON changes, regenerate or review these tables before using them.

XGBoost tuning is performed separately for each expanding-window validation fold
inside the first 80% modeling block. For each fold, Optuna uses the configured
TPE sampler and runs `optuna_trials` (100) trials to minimize validation RMSE. The
final sequential holdout is not used during this selection stage. After all fold
studies are complete, the final model uses the median of the best fold-specific
hyperparameters across all folds; integer-valued parameters are rounded to the
nearest integer, and `n_estimators` is taken as the median early-stopping iteration
observed across the tuned folds (`n_estimators_source = median_all_folds`). This
keeps hyperparameter selection tied to the same temporal validation protocol used
for model assessment.

### XGBoost Search Space by Grand Prix

All ranges are inclusive and are read from the circuit YAML configuration files.

| Grand Prix | learning_rate | max_depth | min_child_weight | subsample | colsample_bytree | gamma | reg_alpha | reg_lambda |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bahrain Grand Prix | 0.006667-0.300 | 3-7 | 4-6 | 0.656-0.844 | 0.744-0.956 | 0.0-0.700 | 0.000333-0.0300 | 0.167-3.000 |
| Saudi Arabian Grand Prix | 0.006667-0.300 | 1-7 | 4-6 | 0.678-0.872 | 0.722-0.928 | 0.0-0.700 | 0.000333-0.0300 | 0.167-3.000 |
| United States Grand Prix | 0.0167-0.500 | 3-9 | 1-6 | 0.719-1.000 | 0.809-1.000 | 0.0-0.700 | 0.00000333-0.0300 | 0.0333-3.000 |
| Italian Grand Prix | 0.0167-0.300 | 1-7 | 4-6 | 0.678-0.872 | 0.722-0.928 | 0.0-0.700 | 0.000333-0.0300 | 0.167-3.000 |
| Hungarian Grand Prix | 0.006667-0.300 | 1-5 | 4-6 | 0.678-0.872 | 0.722-0.928 | 0.0-0.700 | 0.000333-0.003000 | 0.167-3.000 |

### XGBoost Final Hyperparameters (Expanding Window)

These are the final reported XGBoost-EW hyperparameters per circuit, aggregated as
the median of the best Optuna parameters across all expanding-window folds (source:
`Scripts/Results/xgboost/ew/params/{safe_gp_name}_xgb_params_ew.json`). The EW window
ratio is the selected `xgb_ew_window_ratio` from the window-size sweep.

| Grand Prix | EW window | Seed | Sampler | Optuna trials/fold | Final n_estimators | Final learning_rate | Final max_depth | Final min_child_weight | Final subsample | Final colsample_bytree | Final gamma | Final reg_alpha | Final reg_lambda |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bahrain Grand Prix | 30% | 42 | tpe | 100 | 249 | 0.164969 | 3 | 5 | 0.684174 | 0.904215 | 0.203528 | 0.001370 | 0.510169 |
| Saudi Arabian Grand Prix | 50% | 42 | tpe | 100 | 152 | 0.199282 | 3 | 4 | 0.756074 | 0.833232 | 0.316344 | 0.003249 | 0.283445 |
| United States Grand Prix | 5% | 42 | tpe | 100 | 85 | 0.201475 | 6 | 4 | 0.784062 | 0.887642 | 0.213003 | 0.000624 | 0.425945 |
| Italian Grand Prix | 50% | 42 | tpe | 100 | 395 | 0.061874 | 1 | 5 | 0.784588 | 0.779387 | 0.148427 | 0.002628 | 0.265021 |
| Hungarian Grand Prix | 40% | 42 | tpe | 100 | 1050 | 0.042898 | 1 | 6 | 0.769281 | 0.909664 | 0.504674 | 0.001244 | 0.531199 |

### LSTM_hybrid Selected Hyperparameters

These are the selected per-circuit hyperparameters for the `LSTM_hybrid` model
(source: `Scripts/Results/lstm_hybrid/models/{safe_gp_name}_full_embedding_lr_ew_lstm_hybrid_model_metadata.json`).
They are reproducibility detail for the repository and are not intended for the
article. Global settings shared by all circuits: tabular baseline `lr_ew`, LSTM
feature mode `full_embedding`, target mode `residual_from_tabular` (the LSTM learns
the baseline residual), sequence groups `[Year, Driver]`, embedding columns
`[Driver, Team]`, single (non-stacked) LSTM, Adam optimizer, `shuffle=False`,
seed 42, search-space version v11, tuning strategy `single_sequential_split_v1`.
The `lstm_window_ratio` is the value selected by validation RMSE from each circuit's
`lstm_window_ratio_sweep` (never the holdout); `sequence_length = ceil(n_race_laps *
lstm_window_ratio)`. Bahrain uses fixed sweep-found hyperparameters
(`lstm_tuning_enabled: false`); Saudi Arabia, USA, Italy and Hungary were Optuna-tuned.
All circuits load their saved parameters via `use_saved_lstm_params: true`.

| Grand Prix | lstm_window_ratio | sequence_length | lstm_units | dense_units | dropout | recurrent_dropout | learning_rate | batch_size | l2_reg | final epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bahrain Grand Prix | 0.30 | 17 | 24 | 48 | 0.4132 | 0.0829 | 0.000718 | 16 | 0.000693 | 11 |
| Saudi Arabian Grand Prix | 0.03 | 2 | 16 | 24 | 0.3308 | 0.0900 | 0.001230 | 16 | 0.000423 | 2 |
| United States Grand Prix | 0.10 | 6 | 16 | 48 | 0.2409 | 0.1745 | 0.001672 | 16 | 0.001493 | 3 |
| Italian Grand Prix | 0.05 | 3 | 24 | 24 | 0.3779 | 0.1432 | 0.001414 | 32 | 0.000954 | 2 |
| Hungarian Grand Prix | 0.03 | 3 | 16 | 24 | 0.3497 | 0.0998 | 0.001498 | 32 | 0.000732 | 20 |

The LSTM search space (v11) tuned by Optuna for the tuned circuits spans
`lstm_units` ∈ {8, 16, 24, 32}, `lstm_dense_units` ∈ {24, 32, 48}, `lstm_dropout` ∈
[0.22, 0.42], `lstm_recurrent_dropout` ∈ [0.05, 0.18], `lstm_learning_rate` ∈
[7e-4, 1.8e-3] (log), `lstm_batch_size` ∈ {16, 32}, `lstm_l2_reg` ∈ [4e-4, 1.5e-3],
and `lstm_huber_delta` ∈ [0.5, 2.0]; the saved per-circuit values above are loaded at
run time via `use_saved_lstm_params: true`.

### Best Individual Validation Window by Grand Prix (SW sweep diagnostics)

The tables in this and the following section come from the **sliding-window (SW)
sweep** and are retained for diagnostic and comparison purposes only; they are
**not** the final reported configuration (see the XGBoost-EW table above). The best
window is the sliding-window validation fold with the lowest validation RMSE among
the per-window Optuna winners.

| Grand Prix | Best window | Train laps | Validation laps | RMSE | MAE | R2 | n_estimators | learning_rate | max_depth | min_child_weight | subsample | colsample_bytree | gamma | reg_alpha | reg_lambda | Seed | Sampler | Optuna trials/window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bahrain Grand Prix | 9 | 19-25 | 26-27 | 0.261949 | 0.201646 | 0.957173 | 293 | 0.052610 | 4 | 2 | 0.733434 | 0.959617 | 0.177285 | 0.000385 | 0.000002 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 4 | 9-14 | 15-16 | 0.264663 | 0.213567 | 0.912265 | 283 | 0.058773 | 9 | 1 | 0.564455 | 0.836681 | 0.309259 | 0.004672 | 0.009723 | 42 | tpe | 200 |
| United States Grand Prix | 3 | 7-13 | 14-15 | 0.303206 | 0.238058 | 0.951607 | 36 | 0.066580 | 7 | 12 | 0.677726 | 0.905769 | 1.461300 | 0.009293 | 1.239400 | 42 | tpe | 200 |
| Italian Grand Prix | 9 | 19-24 | 25-26 | 0.237769 | 0.186375 | 0.976351 | 257 | 0.016494 | 6 | 6 | 0.694105 | 0.862718 | 0.214785 | 0.000114 | 2.306300 | 42 | tpe | 200 |
| Hungarian Grand Prix | 6 | 18-25 | 26-28 | 0.309160 | 0.239457 | 0.949815 | 92 | 0.087707 | 3 | 4 | 0.722591 | 0.837273 | 0.249881 | 0.005509 | 0.096828 | 42 | tpe | 200 |

### Per-Window XGBoost Hyperparameters

<details>
<summary>Show per-window XGBoost hyperparameter table</summary>

Each row is the best Optuna trial for one sliding-window validation fold inside the first 80% modeling block. The final sequential holdout is not used to select these values.

| Grand Prix | Window | Train laps | Validation laps | RMSE | MAE | R2 | n_estimators | learning_rate | max_depth | min_child_weight | subsample | colsample_bytree | gamma | reg_alpha | reg_lambda | Seed | Sampler | Optuna trials/window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bahrain Grand Prix | 1 | 3-9 | 10-11 | 0.301369 | 0.234505 | 0.956457 | 151 | 0.062377 | 4 | 2 | 0.556793 | 0.820024 | 0.168908 | 0.000000 | 0.075906 | 42 | tpe | 200 |
| Bahrain Grand Prix | 2 | 5-11 | 12-13 | 0.274314 | 0.212241 | 0.968982 | 248 | 0.062945 | 5 | 2 | 0.653872 | 0.916945 | 0.281432 | 0.000909 | 0.000038 | 42 | tpe | 200 |
| Bahrain Grand Prix | 3 | 7-13 | 14-15 | 0.346156 | 0.278328 | 0.948846 | 94 | 0.068494 | 3 | 3 | 0.560847 | 0.979989 | 0.448150 | 0.000004 | 0.000001 | 42 | tpe | 200 |
| Bahrain Grand Prix | 4 | 9-15 | 16-17 | 0.366010 | 0.280413 | 0.940251 | 986 | 0.033610 | 4 | 4 | 0.612948 | 0.885767 | 0.050281 | 0.000006 | 0.000188 | 42 | tpe | 200 |
| Bahrain Grand Prix | 5 | 11-17 | 18-19 | 0.315067 | 0.245935 | 0.891586 | 501 | 0.033438 | 3 | 8 | 0.671866 | 0.965361 | 0.142845 | 0.000000 | 0.000001 | 42 | tpe | 200 |
| Bahrain Grand Prix | 6 | 13-19 | 20-21 | 0.284931 | 0.212332 | 0.922330 | 309 | 0.064166 | 4 | 4 | 0.646711 | 0.831637 | 0.051960 | 0.000000 | 0.000000 | 42 | tpe | 200 |
| Bahrain Grand Prix | 7 | 15-21 | 22-23 | 0.273778 | 0.202716 | 0.942695 | 248 | 0.020943 | 4 | 8 | 0.724609 | 0.867856 | 0.052496 | 0.000000 | 0.000000 | 42 | tpe | 200 |
| Bahrain Grand Prix | 8 | 17-23 | 24-25 | 0.271955 | 0.207659 | 0.945730 | 250 | 0.021696 | 4 | 3 | 0.637371 | 0.876446 | 0.084780 | 0.000032 | 0.000000 | 42 | tpe | 200 |
| Bahrain Grand Prix | 9 | 19-25 | 26-27 | 0.261949 | 0.201646 | 0.957173 | 293 | 0.052610 | 4 | 2 | 0.733434 | 0.959617 | 0.177285 | 0.000385 | 0.000002 | 42 | tpe | 200 |
| Bahrain Grand Prix | 10 | 21-27 | 28-29 | 0.303704 | 0.227541 | 0.945454 | 222 | 0.034447 | 3 | 3 | 0.659919 | 0.854468 | 0.735223 | 0.006141 | 0.000000 | 42 | tpe | 200 |
| Bahrain Grand Prix | 11 | 23-29 | 30-31 | 0.283420 | 0.227231 | 0.959696 | 447 | 0.049478 | 5 | 2 | 0.719943 | 0.970066 | 0.580460 | 0.000005 | 0.000000 | 42 | tpe | 200 |
| Bahrain Grand Prix | 12 | 25-31 | 32-33 | 0.277430 | 0.210829 | 0.961672 | 282 | 0.026165 | 5 | 5 | 0.697181 | 0.964477 | 0.074215 | 0.000009 | 0.000017 | 42 | tpe | 200 |
| Bahrain Grand Prix | 13 | 27-33 | 34-35 | 0.273350 | 0.211208 | 0.954880 | 451 | 0.024869 | 3 | 2 | 0.680935 | 0.833804 | 0.161331 | 0.000000 | 0.000000 | 42 | tpe | 200 |
| Bahrain Grand Prix | 14 | 29-35 | 36-37 | 0.301174 | 0.233115 | 0.945483 | 290 | 0.067391 | 5 | 4 | 0.689499 | 0.950783 | 0.103345 | 0.000599 | 0.000001 | 42 | tpe | 200 |
| Bahrain Grand Prix | 15 | 31-37 | 38-39 | 0.353028 | 0.284503 | 0.915623 | 840 | 0.047956 | 3 | 3 | 0.617249 | 0.854685 | 0.366916 | 0.000197 | 0.000000 | 42 | tpe | 200 |
| Bahrain Grand Prix | 16 | 33-39 | 40-41 | 0.277880 | 0.224748 | 0.954196 | 86 | 0.065383 | 3 | 2 | 0.616115 | 0.858744 | 0.345287 | 0.000011 | 0.000010 | 42 | tpe | 200 |
| Bahrain Grand Prix | 17 | 35-41 | 42-43 | 0.269471 | 0.204263 | 0.957360 | 271 | 0.028926 | 5 | 8 | 0.713367 | 0.842439 | 0.113045 | 0.000000 | 0.000001 | 42 | tpe | 200 |
| Bahrain Grand Prix | 18 | 37-43 | 44-45 | 0.266915 | 0.204948 | 0.945768 | 60 | 0.059882 | 3 | 5 | 0.568595 | 0.928217 | 0.387798 | 0.000107 | 0.000000 | 42 | tpe | 200 |
| Bahrain Grand Prix | 19 | 38-44 | 45-46 | 0.294109 | 0.228401 | 0.936918 | 172 | 0.045710 | 5 | 5 | 0.595806 | 0.833264 | 0.408292 | 0.000000 | 0.000000 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 1 | 3-8 | 9-10 | 0.354021 | 0.270660 | 0.890483 | 43 | 0.058248 | 9 | 4 | 0.558445 | 0.832811 | 0.294984 | 0.003741 | 0.000349 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 2 | 5-10 | 11-12 | 0.340471 | 0.259417 | 0.889112 | 236 | 0.032151 | 8 | 4 | 0.618482 | 0.886740 | 0.353361 | 0.000276 | 0.000603 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 3 | 7-12 | 13-14 | 0.336389 | 0.260820 | 0.865800 | 175 | 0.041983 | 9 | 1 | 0.560258 | 0.829851 | 0.066230 | 0.001081 | 0.000607 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 4 | 9-14 | 15-16 | 0.264663 | 0.213567 | 0.912265 | 283 | 0.058773 | 9 | 1 | 0.564455 | 0.836681 | 0.309259 | 0.004672 | 0.009723 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 5 | 11-16 | 17-18 | 0.350945 | 0.270965 | 0.852905 | 82 | 0.053693 | 8 | 2 | 0.558792 | 0.847405 | 0.156660 | 0.001514 | 0.012347 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 6 | 13-18 | 19-20 | 0.285275 | 0.233352 | 0.885393 | 265 | 0.040674 | 9 | 2 | 0.699956 | 0.870856 | 0.326053 | 0.000553 | 0.019451 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 7 | 15-20 | 21-22 | 0.466472 | 0.360305 | 0.866117 | 405 | 0.032888 | 9 | 1 | 0.631936 | 0.824308 | 0.146237 | 0.000200 | 0.000351 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 8 | 17-22 | 23-24 | 0.415239 | 0.319195 | 0.873786 | 790 | 0.058492 | 8 | 4 | 0.628200 | 0.871786 | 0.393478 | 0.032788 | 0.004013 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 9 | 19-24 | 25-26 | 0.359578 | 0.283343 | 0.915097 | 283 | 0.053362 | 8 | 2 | 0.561972 | 0.848622 | 0.316898 | 0.046711 | 0.005249 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 10 | 21-26 | 27-28 | 0.336068 | 0.267148 | 0.911267 | 119 | 0.025017 | 9 | 1 | 0.555004 | 0.828128 | 0.334567 | 0.072904 | 0.000286 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 11 | 23-28 | 29-30 | 0.305807 | 0.246278 | 0.912249 | 78 | 0.040974 | 8 | 5 | 0.693974 | 0.828214 | 0.396948 | 0.000278 | 0.002080 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 12 | 25-30 | 31-32 | 0.281575 | 0.219758 | 0.931037 | 215 | 0.045846 | 9 | 3 | 0.600665 | 0.825396 | 0.458401 | 0.000476 | 0.001631 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 13 | 27-32 | 33-34 | 0.294521 | 0.222294 | 0.918413 | 168 | 0.026048 | 9 | 5 | 0.592183 | 0.829884 | 0.158431 | 0.046711 | 0.001401 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 14 | 29-34 | 35-36 | 0.298000 | 0.227481 | 0.889138 | 353 | 0.038544 | 10 | 1 | 0.689011 | 0.826644 | 0.060653 | 0.011514 | 0.000370 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 15 | 31-36 | 37-38 | 0.296714 | 0.225857 | 0.887914 | 341 | 0.028475 | 9 | 3 | 0.563268 | 0.832217 | 0.476283 | 0.012306 | 0.002359 | 42 | tpe | 200 |
| Saudi Arabian Grand Prix | 16 | 33-38 | 39-40 | 0.293616 | 0.235103 | 0.888082 | 872 | 0.025017 | 9 | 5 | 0.558753 | 0.871312 | 0.288562 | 0.000340 | 0.001179 | 42 | tpe | 200 |
| United States Grand Prix | 1 | 3-9 | 10-11 | 0.395712 | 0.317731 | 0.935875 | 311 | 0.058736 | 7 | 8 | 0.608547 | 0.925172 | 0.431517 | 0.000192 | 1.256400 | 42 | tpe | 200 |
| United States Grand Prix | 2 | 5-11 | 12-13 | 0.331124 | 0.271747 | 0.954508 | 1210 | 0.062191 | 6 | 11 | 0.645482 | 0.919626 | 0.402780 | 0.000470 | 0.529002 | 42 | tpe | 200 |
| United States Grand Prix | 3 | 7-13 | 14-15 | 0.303206 | 0.238058 | 0.951607 | 36 | 0.066580 | 7 | 12 | 0.677726 | 0.905769 | 1.461300 | 0.009293 | 1.239400 | 42 | tpe | 200 |
| United States Grand Prix | 4 | 9-15 | 16-17 | 0.371402 | 0.284914 | 0.931909 | 339 | 0.081744 | 6 | 12 | 0.615184 | 0.934682 | 0.897551 | 0.003311 | 1.192100 | 42 | tpe | 200 |
| United States Grand Prix | 5 | 11-17 | 18-19 | 0.352295 | 0.268500 | 0.920872 | 335 | 0.046724 | 6 | 12 | 0.603575 | 0.867726 | 0.664974 | 0.000764 | 0.326938 | 42 | tpe | 200 |
| United States Grand Prix | 6 | 13-19 | 20-21 | 0.311893 | 0.248990 | 0.916547 | 740 | 0.047698 | 5 | 11 | 0.675231 | 0.948236 | 0.466424 | 0.000277 | 0.758085 | 42 | tpe | 200 |
| United States Grand Prix | 7 | 15-21 | 22-23 | 0.379923 | 0.291544 | 0.879063 | 449 | 0.080878 | 5 | 7 | 0.629892 | 0.869973 | 0.405688 | 0.003280 | 0.384957 | 42 | tpe | 200 |
| United States Grand Prix | 8 | 17-23 | 24-25 | 0.304411 | 0.228617 | 0.925957 | 509 | 0.058701 | 7 | 13 | 0.663318 | 0.959957 | 0.580505 | 0.004337 | 0.603875 | 42 | tpe | 200 |
| United States Grand Prix | 9 | 19-25 | 26-27 | 0.422324 | 0.329661 | 0.915672 | 466 | 0.065618 | 6 | 7 | 0.701383 | 0.894246 | 0.520406 | 0.004946 | 0.220291 | 42 | tpe | 200 |
| United States Grand Prix | 10 | 21-27 | 28-29 | 0.316245 | 0.250462 | 0.956628 | 120 | 0.052104 | 7 | 7 | 0.609734 | 0.948200 | 0.501791 | 0.029063 | 0.280783 | 42 | tpe | 200 |
| United States Grand Prix | 11 | 23-29 | 30-31 | 0.312090 | 0.253940 | 0.958116 | 263 | 0.080888 | 5 | 8 | 0.709929 | 0.912281 | 0.608305 | 0.001704 | 0.301464 | 42 | tpe | 200 |
| United States Grand Prix | 12 | 25-31 | 32-33 | 0.391483 | 0.314186 | 0.934945 | 175 | 0.064246 | 6 | 8 | 0.603479 | 0.866010 | 0.424591 | 0.016758 | 1.392500 | 42 | tpe | 200 |
| United States Grand Prix | 13 | 27-33 | 34-35 | 0.424784 | 0.313894 | 0.928975 | 386 | 0.061913 | 6 | 13 | 0.655056 | 0.901441 | 0.461182 | 0.000492 | 0.971432 | 42 | tpe | 200 |
| United States Grand Prix | 14 | 29-35 | 36-37 | 0.345862 | 0.254255 | 0.956433 | 397 | 0.045741 | 5 | 8 | 0.604369 | 0.884784 | 0.455491 | 0.020671 | 0.255650 | 42 | tpe | 200 |
| United States Grand Prix | 15 | 31-37 | 38-39 | 0.303970 | 0.246535 | 0.962337 | 128 | 0.077068 | 6 | 11 | 0.638465 | 0.922501 | 0.785040 | 0.001897 | 0.435452 | 42 | tpe | 200 |
| United States Grand Prix | 16 | 33-39 | 40-41 | 0.350431 | 0.278964 | 0.948796 | 43 | 0.075585 | 7 | 9 | 0.701830 | 0.881012 | 1.076300 | 0.039650 | 0.314635 | 42 | tpe | 200 |
| United States Grand Prix | 17 | 35-41 | 42-43 | 0.338708 | 0.262118 | 0.951461 | 172 | 0.080172 | 5 | 7 | 0.623421 | 0.883180 | 0.844037 | 0.046422 | 1.485800 | 42 | tpe | 200 |
| United States Grand Prix | 18 | 37-43 | 44-45 | 0.328100 | 0.272571 | 0.958296 | 173 | 0.080909 | 7 | 8 | 0.648148 | 0.871918 | 0.611257 | 0.000886 | 0.329171 | 42 | tpe | 200 |
| Italian Grand Prix | 1 | 3-8 | 9-10 | 0.301864 | 0.238088 | 0.955131 | 359 | 0.016880 | 5 | 8 | 0.683778 | 0.840265 | 0.220332 | 0.003439 | 1.817600 | 42 | tpe | 200 |
| Italian Grand Prix | 2 | 5-10 | 11-12 | 0.256384 | 0.196097 | 0.969484 | 1028 | 0.013991 | 5 | 6 | 0.600529 | 0.824211 | 0.227925 | 0.000118 | 1.923300 | 42 | tpe | 200 |
| Italian Grand Prix | 3 | 7-12 | 13-14 | 0.285717 | 0.226356 | 0.965378 | 618 | 0.013617 | 6 | 6 | 0.673394 | 0.907047 | 0.300389 | 0.000644 | 1.583400 | 42 | tpe | 200 |
| Italian Grand Prix | 4 | 9-14 | 15-16 | 0.279039 | 0.221131 | 0.969312 | 1104 | 0.015668 | 5 | 7 | 0.619247 | 0.822207 | 0.218150 | 0.001353 | 1.653200 | 42 | tpe | 200 |
| Italian Grand Prix | 5 | 11-16 | 17-18 | 0.264224 | 0.198937 | 0.971976 | 859 | 0.018790 | 6 | 9 | 0.633588 | 0.830143 | 0.212434 | 0.000368 | 3.748500 | 42 | tpe | 200 |
| Italian Grand Prix | 6 | 13-18 | 19-20 | 0.272926 | 0.214533 | 0.968241 | 633 | 0.017889 | 6 | 11 | 0.666789 | 0.828191 | 0.305256 | 0.000749 | 4.057900 | 42 | tpe | 200 |
| Italian Grand Prix | 7 | 15-20 | 21-22 | 0.282596 | 0.226699 | 0.970848 | 671 | 0.017332 | 5 | 7 | 0.679594 | 0.894031 | 0.201253 | 0.000469 | 1.671300 | 42 | tpe | 200 |
| Italian Grand Prix | 8 | 17-22 | 23-24 | 0.258221 | 0.194768 | 0.973966 | 840 | 0.011354 | 6 | 6 | 0.614690 | 0.864367 | 0.242864 | 0.001313 | 1.698600 | 42 | tpe | 200 |
| Italian Grand Prix | 9 | 19-24 | 25-26 | 0.237769 | 0.186375 | 0.976351 | 257 | 0.016494 | 6 | 6 | 0.694105 | 0.862718 | 0.214785 | 0.000114 | 2.306300 | 42 | tpe | 200 |
| Italian Grand Prix | 10 | 21-26 | 27-28 | 0.275236 | 0.219757 | 0.970289 | 631 | 0.010684 | 6 | 11 | 0.700898 | 0.856122 | 0.239455 | 0.002920 | 4.174500 | 42 | tpe | 200 |
| Italian Grand Prix | 11 | 23-28 | 29-30 | 0.261021 | 0.200619 | 0.975283 | 738 | 0.017226 | 6 | 6 | 0.685289 | 0.899382 | 0.216410 | 0.000504 | 1.549600 | 42 | tpe | 200 |
| Italian Grand Prix | 12 | 25-30 | 31-32 | 0.264863 | 0.210851 | 0.976304 | 840 | 0.018583 | 6 | 6 | 0.645018 | 0.919420 | 0.221667 | 0.002519 | 1.542400 | 42 | tpe | 200 |
| Italian Grand Prix | 13 | 27-32 | 33-34 | 0.285959 | 0.217376 | 0.972000 | 1051 | 0.018993 | 6 | 11 | 0.604994 | 0.875238 | 0.200197 | 0.001178 | 2.588400 | 42 | tpe | 200 |
| Italian Grand Prix | 14 | 29-34 | 35-36 | 0.277219 | 0.199682 | 0.975816 | 1259 | 0.016447 | 6 | 7 | 0.687176 | 0.886947 | 0.211488 | 0.004601 | 1.603900 | 42 | tpe | 200 |
| Italian Grand Prix | 15 | 31-36 | 37-38 | 0.277831 | 0.225404 | 0.973294 | 309 | 0.019997 | 6 | 6 | 0.717498 | 0.872907 | 0.290514 | 0.008142 | 1.557100 | 42 | tpe | 200 |
| Italian Grand Prix | 16 | 33-38 | 39-40 | 0.275564 | 0.209580 | 0.970124 | 1656 | 0.010011 | 6 | 6 | 0.667180 | 0.831827 | 0.200814 | 0.000573 | 1.580900 | 42 | tpe | 200 |
| Italian Grand Prix | 17 | 35-40 | 41-42 | 0.317410 | 0.240205 | 0.967157 | 1553 | 0.019379 | 5 | 6 | 0.664288 | 0.870452 | 0.208864 | 0.000505 | 2.852600 | 42 | tpe | 200 |
| Hungarian Grand Prix | 1 | 3-10 | 11-13 | 0.310716 | 0.242880 | 0.925063 | 77 | 0.092795 | 2 | 4 | 0.600223 | 0.820882 | 0.368678 | 0.000974 | 1.272500 | 42 | tpe | 200 |
| Hungarian Grand Prix | 2 | 6-13 | 14-16 | 0.338268 | 0.258543 | 0.921818 | 216 | 0.049561 | 2 | 12 | 0.843497 | 0.764091 | 0.450592 | 0.000125 | 0.021438 | 42 | tpe | 200 |
| Hungarian Grand Prix | 3 | 9-16 | 17-19 | 0.336870 | 0.254798 | 0.938440 | 104 | 0.090437 | 2 | 9 | 0.631563 | 0.818692 | 0.323488 | 0.009219 | 0.827807 | 42 | tpe | 200 |
| Hungarian Grand Prix | 4 | 12-19 | 20-22 | 0.358004 | 0.278721 | 0.922989 | 346 | 0.058045 | 2 | 5 | 0.636026 | 0.812135 | 0.295892 | 0.020641 | 0.164727 | 42 | tpe | 200 |
| Hungarian Grand Prix | 5 | 15-22 | 23-25 | 0.327026 | 0.243322 | 0.944831 | 401 | 0.035478 | 2 | 5 | 0.663575 | 0.829986 | 0.210799 | 0.000803 | 0.026365 | 42 | tpe | 200 |
| Hungarian Grand Prix | 6 | 18-25 | 26-28 | 0.309160 | 0.239457 | 0.949815 | 92 | 0.087707 | 3 | 4 | 0.722591 | 0.837273 | 0.249881 | 0.005509 | 0.096828 | 42 | tpe | 200 |
| Hungarian Grand Prix | 7 | 21-28 | 29-31 | 0.359039 | 0.277049 | 0.911991 | 74 | 0.042134 | 3 | 5 | 0.849405 | 0.853912 | 0.258624 | 0.001498 | 0.279159 | 42 | tpe | 200 |
| Hungarian Grand Prix | 8 | 24-31 | 32-34 | 0.342491 | 0.260211 | 0.920781 | 265 | 0.041836 | 2 | 4 | 0.636951 | 0.759127 | 0.495152 | 0.002964 | 0.087688 | 42 | tpe | 200 |
| Hungarian Grand Prix | 9 | 27-34 | 35-37 | 0.365899 | 0.269131 | 0.925437 | 73 | 0.086351 | 2 | 4 | 0.733009 | 0.784514 | 0.624615 | 0.000552 | 0.029181 | 42 | tpe | 200 |
| Hungarian Grand Prix | 10 | 30-37 | 38-40 | 0.347562 | 0.265236 | 0.925916 | 81 | 0.091259 | 2 | 4 | 0.844065 | 0.746299 | 0.216963 | 0.006087 | 0.039366 | 42 | tpe | 200 |
| Hungarian Grand Prix | 11 | 33-40 | 41-43 | 0.350979 | 0.259860 | 0.929395 | 122 | 0.099873 | 2 | 4 | 0.724519 | 0.804054 | 0.336522 | 0.012390 | 0.185608 | 42 | tpe | 200 |
| Hungarian Grand Prix | 12 | 36-43 | 44-46 | 0.334563 | 0.248998 | 0.937644 | 437 | 0.034773 | 2 | 9 | 0.632446 | 0.821255 | 0.569455 | 0.000102 | 1.533800 | 42 | tpe | 200 |
| Hungarian Grand Prix | 13 | 39-46 | 47-49 | 0.344274 | 0.255529 | 0.941495 | 127 | 0.057810 | 2 | 4 | 0.692259 | 0.763183 | 0.248484 | 0.066293 | 0.046166 | 42 | tpe | 200 |
| Hungarian Grand Prix | 14 | 42-49 | 50-52 | 0.381210 | 0.285391 | 0.927675 | 227 | 0.067958 | 2 | 4 | 0.600639 | 0.849005 | 0.240431 | 0.485330 | 0.566273 | 42 | tpe | 200 |
| Hungarian Grand Prix | 15 | 45-52 | 53-55 | 0.347160 | 0.265932 | 0.941226 | 158 | 0.067304 | 2 | 5 | 0.739736 | 0.820591 | 0.239606 | 0.098333 | 0.020753 | 42 | tpe | 200 |
| Hungarian Grand Prix | 16 | 46-53 | 54-56 | 0.457701 | 0.294943 | 0.912278 | 65 | 0.096738 | 2 | 4 | 0.697012 | 0.837758 | 0.332502 | 0.001239 | 0.012171 | 42 | tpe | 200 |

</details>

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

Pick a circuit with either `CONFIG_PATH` (a YAML file) or `TARGET_GP_NAME`, then run
any script. Examples use `CONFIG_PATH`; on Windows use `python` from the activated
venv. Run commands from the repository root (paths are case-sensitive on Linux/macOS).

Core experiment (expanding-window) for one circuit:

```bash
export CONFIG_PATH="configs/bahrain.yaml"
python Scripts/Source/model_lr_ew.py
python Scripts/Source/model_xgb_ew.py
python Scripts/Source/model_lstm_hybrid.py
```

Extra models (sliding-window LR/XGBoost and the baseline-LapTime_prev LSTM):

```bash
export CONFIG_PATH="configs/bahrain.yaml"
python Scripts/Source/model_lr_sw.py
python Scripts/Source/model_xgb_sw.py
python Scripts/Source/model_lstm_baseline.py
```

Analysis and figures (run a final model first so its artifact exists):

```bash
export CONFIG_PATH="configs/bahrain.yaml"
python Scripts/Source/model_interpretability.py
python Scripts/Source/plot_driver_holdout_timeseries.py --driver VER
```

Reproduce the full reported experiment (LR-EW + XGBoost-EW for every circuit, using
the window ratios encoded in each YAML); add `--with-hybrid` to also run `LSTM_hybrid`:

```bash
python Scripts/Source/run_experiment.py
python Scripts/Source/run_experiment.py --circuit bahrain italy
python Scripts/Source/run_experiment.py --with-hybrid
```

If a circuit's XGBoost parameter file is missing or no longer matches the current YAML
bounds, sampler, or tuning procedure, the XGBoost script re-runs Optuna before
training (an independent study per fold), so the run takes longer.

Inspect the experiment history locally by starting the MLflow UI from the repository
root after running at least one model, then open the printed URL (usually
`http://127.0.0.1:5000`):

```bash
python -m mlflow ui --backend-store-uri Scripts/Results/mlruns
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
- Sliding-window and expanding-window validation are performed only inside the first 80% modeling block. The `LSTM_hybrid` and baseline-LapTime_prev LSTM models instead use a single sequential split inside that same modeling block.
- XGBoost SW parameter files are generated under `Scripts/Results/xgboost/sw/params/` when needed. XGBoost EW parameter files (the final reported configuration) are generated under `Scripts/Results/xgboost/ew/params/`. Both are ignored by Git.
- LSTM artifacts are generated under `Scripts/Results/lstm/` and `LSTM_hybrid` artifacts (models, params, and audit baseline predictions) under `Scripts/Results/lstm_hybrid/`; both are ignored by Git.
- MLflow run metadata is generated under `Scripts/Results/mlruns/` by default and is ignored by Git.
- The notebooks remain the narrative, circuit-specific record of the analysis; the scripts are the lean reproducible runners for GitHub.

## Authors

- Marcos Paulo de Oliveira Pereira
- Carlos Henrique Gomes Ferreira
- Alexandre Magno de Sousa

Universidade Federal de Ouro Preto (UFOP)
