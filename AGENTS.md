# AGENTS.md

## Project context

This repository contains research code for multi-circuit Formula 1 lap-time prediction.

The main goal is to model `LapTime_seconds` using FastF1-derived race data, with a temporal validation protocol that simulates real race conditions:

- the full dataset is ordered first by year, then by lap number within each year (e.g. all 2022 laps come before all 2023 laps);
- sliding-window (SW) and expanding-window (EW) validation inside the first 80% of the ordered dataset;
- final sequential holdout on the last 20% of the ordered dataset;
- comparison between Linear Regression, XGBoost and `LSTM_hybrid` (the selected third model: best tabular expanding-window baseline plus an LSTM residual);
- reporting of RMSE, MAE, R2, residual standard deviation, bootstrap confidence intervals, COS_MAE and COS_RMSE.

This is an academic/research project. Prioritize reproducibility, methodological consistency, clean code and clear experiment reporting.

## Language standard

English is the default language for the entire project.

Use English for:

- README files;
- notebook markdown cells;
- code comments;
- variable names;
- function names;
- file documentation;
- experiment descriptions.

Do not add Portuguese text unless explicitly requested.

## Repository structure

Important folders:

- `Data/`: raw race-session data by Grand Prix/circuit.
- `Scripts/ModelData/`: cleaned and engineered datasets used by the models.
- `Scripts/Notebooks/`: circuit-specific notebooks with full analysis.
- `Scripts/Source/`: reproducible Python scripts.
- `Utils/`: auxiliary files such as `requirements.txt`, mappings and configuration files.

Main scripts:

- `Scripts/Source/model_lr_sw.py` — Linear Regression with sliding-window validation.
- `Scripts/Source/model_lr_ew.py` — Linear Regression with expanding-window validation.
- `Scripts/Source/model_xgb_sw.py` — XGBoost with sliding-window validation.
- `Scripts/Source/model_xgb_ew.py` — XGBoost with expanding-window validation.
- `Scripts/Source/model_lstm.py` — pure LSTM with single sequential split and sequential holdout.
- `Scripts/Source/model_lstm_hybrid.py` — the selected third model (`LSTM_hybrid`): best tabular expanding-window baseline (LR-EW or XGBoost-EW, per circuit) plus an LSTM trained on the residual. Reuses the `model_lstm.py` core (single sequential split, sequence construction, Optuna, epoch calibration).
- `Scripts/Source/window_size_sweep.py` — window size sensitivity analysis across all approaches.
- `Scripts/Source/search_space_sweep.py` — baseline XGBoost configurations evaluated on the generic window size; derives initial search space bounds.
- `Scripts/Source/search_space_sweep_ew.py` — baseline XGBoost configurations evaluated on the **final selected EW window size** for each circuit (`xgb_ew_window_ratio`); derives the definitive circuit-specific search space bounds used in the experiment.
- `Scripts/Source/run_experiment.py` — runs the final LR-EW + XGBoost-EW experiment for all circuits.

## Dataset ordering

The dataset must be sorted by `Year` first, then by `LapNumber` within each year.
This produces a single chronological sequence that respects multi-season temporal
ordering (e.g. all 2022 laps precede all 2023 laps, which precede all 2024 laps).

- Do not sort only by `LapNumber` without also sorting by `Year`.
- Do not shuffle the dataset.
- Preserve this ordering in every script, notebook and data-preparation step.

## Validation protocol

The full ordered dataset is split into two non-overlapping blocks:

- **Modeling block**: first 80% of rows (by temporal order).
- **Final sequential holdout**: last 20% of rows.

The holdout must remain completely untouched until final evaluation.
All window-based validation (SW or EW) happens exclusively inside the modeling block.

### Sliding-window validation (SW)

- The window size is defined as a **fraction of the total dataset** (not of a single race).
- Each window is divided internally into 80% train / 20% validation.
- Windows advance by steps equal to the validation portion size.
- Example: with a 20% window size, each window spans 20% of the total dataset;
  the train portion is the first 16% and the validation portion is the last 4%;
  the next window shifts forward by 4%.
- Windows must not overlap with the final holdout block.

### Expanding-window validation (EW)

- Validation operates on the same ordered dataset (Year → LapNumber).
- The initial training set starts at the first row of the modeling block.
- Each iteration expands the training set by adding the next validation chunk,
  then validates on the following chunk of the same fixed size.
- The validation chunk size equals the SW validation portion (fraction of total dataset).
- Windows must not overlap with the final holdout block.
- The expanding window never shrinks; training always includes all previously seen data.

## General rules

- Do not change the temporal validation protocol unless explicitly requested.
- Preserve the final 20% sequential holdout.
- SW and EW validation must happen only inside the first 80% modeling block.
- Do not introduce data leakage.
- Do not use future laps to predict previous laps.
- Do not shuffle race laps when temporal ordering matters.
- Keep `Year` → `LapNumber` ordering consistent across all scripts and notebooks.
- Preserve the target column: `LapTime_seconds`.
- Preserve article-facing metrics unless explicitly asked to change them.
- Do not remove COS metrics.
- Do not simplify the experiment in a way that weakens the paper methodology.

## Notebook and script consistency

- Keep notebook pipelines clean and consistent with the separated model scripts.
- Notebooks and model scripts must produce equivalent results when using the same data, configuration and random seed.
- Any methodological change made in the reproducible scripts must be reflected in the related notebooks before the task is considered complete.
- Notebook code, outputs and markdown must tell the same methodological story as the scripts, including feature removals, validation protocol, interpretability steps and reported metrics.
- Avoid duplicated logic between notebooks and scripts.
- When possible, move reusable logic to shared functions or modules.
- If a script is updated, verify whether related notebooks need the same methodological update.
- If a notebook is updated, verify whether related scripts need the same methodological update.
- Keep notebook outputs, markdown explanations and code cells aligned with the current experiment protocol.

## Data collection consistency

- Keep data collection scripts updated and consistent with the notebooks.
- Changes in data collection logic must be reflected in the notebooks and modeling scripts when relevant.
- Do not allow notebooks, collection scripts and model scripts to follow different assumptions about columns, file names, circuits, sessions or preprocessing.
- Preserve compatibility with FastF1-derived data unless a change is explicitly requested.

## Configuration files

Create and maintain YAML configuration files for values reused across the project.

Use YAML files for:

- directory paths;
- dataset paths;
- fixed coefficients;
- model parameters;
- feature lists;
- target column names;
- supported Grand Prix names;
- split ratios;
- window sizes;
- constants reused across notebooks and scripts;
- output directories;
- random seeds;
- Optuna trial counts and sampler settings;
- window size sweep ranges and step sizes.

Avoid hardcoded values spread across multiple files.

Whenever a value is used in more than one place, prefer moving it to a YAML configuration file.

## Code quality

- Prioritize clean, readable and maintainable code.
- Use functions to calculate metrics.
- Reuse metric functions as much as possible.
- Organize shared metric logic in reusable modules.
- Remove redundant code from notebooks and scripts.
- Avoid repeated blocks of code that can be replaced by a function.
- Prefer explicit variable names.
- Keep functions focused on one responsibility.
- Avoid unnecessary abstractions.
- Avoid large rewrites unless they improve reproducibility or remove major duplication.
- Preserve existing naming conventions unless improving consistency across the project.

## README maintenance

Update the README whenever a change affects:

- project structure;
- execution commands;
- dependencies;
- datasets;
- configuration files;
- methodology;
- validation protocol;
- metrics;
- supported circuits;
- expected outputs;
- model scripts;
- notebooks;
- data collection flow.

The README must always remain consistent with the current state of the project.
Whenever new command-line instructions are added to the README, include the
Linux/macOS command form as well as the Windows/PowerShell form when both are
relevant. Keep environment-variable syntax, virtual-environment activation, and
path separators appropriate for each platform.

## Notebook markdown maintenance

Keep notebook markdown cells updated, organized and written in English.

Notebook markdown must include:

- clear titles;
- numbered sections;
- explanations consistent with the executed code;
- descriptions of preprocessing steps;
- descriptions of validation strategy;
- explanations of metrics;
- concise interpretation of results.

Do not leave outdated markdown that describes a previous version of the experiment.

## Modeling rules

Numerical features currently used:

- `TyreLife`
- `LapNumber`
- `Humidity_RBF_Median`
- `Pressure_RBF_Median`
- `TrackTemp_RBF_Median`
- `WindSpeed_RBF_Median`
- `TempDelta_RBF_Median`
- `Year`
- `LapTime_prev`

Categorical features currently used:

- `Driver`
- `Team`
- `pirelliCompound`

Target:

- `LapTime_seconds`

`Year` is a numerical feature, not a categorical feature. Do not one-hot encode
`Year` unless a future methodological change explicitly justifies and documents
that decision.

Do not add or remove features without explaining the methodological impact.

### Circuit-specific feature removals

The current article-facing feature sets include circuit-specific removals based
on correlation and PCA-loading analysis. Preserve these removals unless the user
explicitly requests a new feature-selection experiment:

- United States Grand Prix:
  - remove `TempDelta_RBF_Median`;
  - remove `Year`;
  - keep `TrackTemp_RBF_Median` as the more physically interpretable thermal/grip proxy.
- Saudi Arabian Grand Prix:
  - remove `TrackTemp_RBF_Median`;
  - keep `Pressure_RBF_Median` as the retained atmospheric-state proxy.
- Hungarian Grand Prix:
  - remove `Humidity_RBF_Median`;
  - keep `TrackTemp_RBF_Median` as the retained track-surface thermal proxy.
- Bahrain Grand Prix and Italian Grand Prix:
  - no circuit-specific removals are currently applied beyond the configured feature lists.

These removals affect modeling scripts, interpretability outputs, PCA/correlation
analysis, README descriptions and notebook narrative. If any removal is changed
in YAML or code, update the corresponding notebook code, notebook markdown,
script documentation and result interpretation together.

## Linear Regression

For `model_lr_sw.py` and `model_lr_ew.py`:

- Preserve preprocessing with imputation, scaling and categorical encoding.
- Fit preprocessing only on the training portion of each split/window.
- Never fit scalers, imputers or encoders using validation or holdout data.   
- Keep results comparable with the XGBoost scripts.
- `model_lr_sw.py` implements sliding-window validation (see Validation Protocol).
- `model_lr_ew.py` implements expanding-window validation (see Validation Protocol).

## XGBoost

For `model_xgb_sw.py` and `model_xgb_ew.py`:

- Preserve Optuna tuning unless explicitly asked to disable it.
- Do not tune hyperparameters using the final holdout.
- The final holdout must remain untouched until final evaluation.
- Generated parameter dumps or run outputs should remain ignored by Git when appropriate.
- `model_xgb_sw.py` implements sliding-window validation (see Validation Protocol).
- `model_xgb_ew.py` implements expanding-window validation (see Validation Protocol).

### Optuna configuration

- Number of trials: **100** per optimization run.
- Sampler: always **TPE** (`optuna.samplers.TPESampler`).
- Each Grand Prix has its own **circuit-specific search space**, defined in YAML.
- Before defining each circuit's search space, run a small set of **baseline
  configurations** (e.g. XGBoost defaults and two or three hand-picked variants)
  to establish a directional prior for the hyperparameter ranges. Document the
  baseline results and use them to justify the bounds chosen for each circuit's
  search space.
- The search space bounds (e.g. `max_depth`, `learning_rate`, `n_estimators`,
  `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`) are set
  arbitrarily per circuit based on the baseline runs. Do not share a single
  global search space across all circuits.
- Store circuit-specific search spaces in the YAML configuration file so they
  can be updated without touching source code.

## LSTM

Script: `Scripts/Source/model_lstm.py`. Requires TensorFlow/Keras.

`model_lstm.py` is the pure LSTM. The selected third model for the final comparison
is `LSTM_hybrid` (`Scripts/Source/model_lstm_hybrid.py`): the best tabular
expanding-window baseline (LR-EW or XGBoost-EW, set per circuit via
`hybrid_baseline_model` from validation metrics, never the holdout) plus an LSTM
trained to predict the residual `LapTime_seconds - baseline_prediction`; the final
prediction is `baseline_prediction + lstm_residual_prediction`. The hybrid reuses
the LSTM core below unchanged (single sequential split, sequence construction,
Optuna, epoch calibration) and forces `lstm_target_mode = 'residual_from_tabular'`.
The baseline series is an out-of-fold expanding-window prediction over the modeling
block (leakage-free for both the validation split and the final residual targets);
the holdout is never used to train or select the baseline. The same validation
protocol, preprocessing, leakage rules and metric set described below apply to the
hybrid. All current circuits use `lr_ew` as the hybrid baseline and the
`full_embedding` feature mode.

### Validation protocol

LSTM uses a **single sequential split** instead of sliding/expanding windows:

- The ordered modeling block (first 80% of all laps) is divided into a train split
  (first `window_train_ratio` of modeling laps) and a validation split (remaining laps).
- Optuna tuning runs on this single split; `EarlyStopping` on `val_loss` calibrates
  the epoch count.
- The final model is retrained on the **full modeling block** for
  `max(median_optuna_epochs, lstm_min_final_epochs)` epochs.
- The sequential holdout (last 20%) is never touched until final evaluation.

Expanding/sliding window is not used for LSTM because each (Year, Driver) group
contributes approximately 50 sequences after windowing — multiple folds would
fragment this small pool and multiply training cost linearly.

### Sequence construction

Sequences are grouped by `lstm_group_cols` (default: `[Year, Driver]`).
Within each group, laps are sorted by `LapNumber`.
For each target lap, the `sequence_length` immediately preceding laps of the same
group form the input sequence.

Sequence length is derived from YAML:

```
sequence_length = ceil(n_race_laps * lstm_window_ratio)
```

`lstm_window_ratio` is the primary key. Falls back to `lstm_ew_window_ratio`, then
`window_ratio` if not set.

### Preprocessing

- Median imputation → StandardScaler on features (fit on training portion only).
- One-hot encoding (full rank, no drop-first) aligned between train and context splits.
- Separate StandardScaler on the target `LapTime_seconds` (fit on training sequences only).

Never fit any transformer using validation or holdout data.

### Optuna configuration

- Trials: **20** per run (controlled by `lstm_optuna_trials` in YAML).
- Sampler: **TPE** (`optuna.samplers.TPESampler`, `multivariate=True`).
- Search space version: **v8**. Tuning strategy: **single_sequential_split_v1**.
- Objective: validation RMSE on the single sequential val split.
- `lstm_stacked` is always `False` during tuning.
- Saved parameters are loaded only when `use_saved_lstm_params: true` in YAML **and**
  the saved file matches the current `search_space_version`, `tuning_strategy`, and
  `n_trials`. Any mismatch triggers a fresh Optuna run.

Current search space bounds (v8):

| Parameter              | Type        | Range / Choices          |
|------------------------|-------------|--------------------------|
| `lstm_units`           | categorical | [64, 128]                |
| `lstm_dense_units`     | categorical | [64, 128]                |
| `lstm_dropout`         | float       | [0.05, 0.50]             |
| `lstm_recurrent_dropout` | float     | [0.20, 0.45]             |
| `lstm_learning_rate`   | float (log) | [3e-4, 5e-3]             |
| `lstm_batch_size`      | categorical | [32, 64]                 |
| `lstm_l2_reg`          | float       | [1e-4, 3e-3]             |

### Model architecture

Input → LSTM(units, dropout, recurrent_dropout, [L2]) → [optional second LSTM if stacked] →
BatchNormalization → Dense(dense_units, relu, [L2]) → Dense(1).
Compiled with Adam + MSE loss. Training uses `shuffle=False`.

### Artifacts

Saved under `Scripts/Results/lstm/`:

- `lstm/models/{safe_gp_name}_lstm_model.keras` — final Keras model.
- `lstm/models/{safe_gp_name}_lstm_model_metadata.json` — full run metadata.
- `lstm/params/{safe_gp_name}_lstm_params.json` — Optuna best params + epoch count.
- `lstm/params/{safe_gp_name}_lstm_optuna_trials.csv` — per-trial results.

### Key YAML keys for LSTM

- `lstm_window_ratio` — controls sequence length (primary key).
- `lstm_tuning_enabled` — set to `false` to skip Optuna and use YAML defaults.
- `lstm_optuna_trials` — number of Optuna trials (default 20).
- `lstm_group_cols` — list of columns used to group sequences (default `[Year, Driver]`).
- `use_saved_lstm_params` — reuse saved params if search space version/strategy/n_trials match.
- `lstm_min_final_epochs` — floor for the final epoch count (default 10).
- `lstm_tuning_epochs` / `lstm_tuning_patience` — epochs and patience used during Optuna trials.
- `lstm_epochs` / `lstm_patience` — epochs and patience used when tuning is disabled.

### Metrics reported

Same metric set as LR and XGBoost:

- Validation split: RMSE, MAE, R2, residual STD.
- Sequential holdout: RMSE, MAE, R2, each with 95% bootstrap CI.
- COS_MAE and COS_RMSE (val split plays the role of SW/EW in the formula).

### Running

```bash
# Linux / macOS
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_lstm.py

# Windows / PowerShell
$env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/model_lstm.py
```

## Window size sweep

The script `Scripts/Source/window_size_sweep.py` evaluates both sliding-window
and expanding-window approaches across a range of window sizes.

Rules:

- Window sizes to test: **5% to 50%** of the total dataset, in configurable steps
  (default step: 5%). The range and step must be defined in the YAML configuration
  file, not hardcoded.
- For each window size, run both SW and EW validation for both LR and XGBoost models.
- Apply the same temporal ordering (Year → LapNumber) and the same 80/20
  modeling/holdout split used in the main scripts.
- Report all standard metrics for each combination: RMSE, MAE, R2, residual STD,
  sample STD, bootstrap confidence intervals, COS_MAE, COS_RMSE.
- Save results to a structured output file (e.g. CSV or JSON) so that results
  can be compared across circuits, models and window sizes without re-running.
- The output file path must be defined in the YAML configuration file.
- Do not overwrite previous sweep results without explicit confirmation.
- The sweep script must be runnable independently of the main model scripts.

## Final selected configuration per Grand Prix

After running the window size sweep (`window_size_sweep.py`) across both
validation approaches, the following method and window size were selected as
the article-facing configuration for each supported Grand Prix. Expanding-window
(EW) validation was selected for every circuit and every model; no circuit
currently uses a sliding-window (SW) result as its final reported configuration.

- Bahrain Grand Prix:
  - Linear Regression: EW, window size 5%.
  - XGBoost: EW, window size 30%.
- Saudi Arabian Grand Prix:
  - Linear Regression: EW, window size 10%.
  - XGBoost: EW, window size 50%.
- United States Grand Prix:
  - Linear Regression: EW, window size 45%.
  - XGBoost: EW, window size 5%.
- Italian Grand Prix:
  - Linear Regression: EW, window size 5%.
  - XGBoost: EW, window size 50%.
- Hungarian Grand Prix:
  - Linear Regression: EW, window size 45%.
  - XGBoost: EW, window size 40%.

These final selections must be reflected in the per-circuit YAML configuration
(the window size and method used when running `model_lr_ew.py` and
`model_xgb_ew.py` for each Grand Prix) and in the README and notebook
narrative whenever article-facing results are reported. The `model_lr_sw.py`
and `model_xgb_sw.py` scripts and their SW results remain part of the sweep
and comparison analysis but are not the final reported configuration for any
circuit. Do not change these selections without explicitly documenting the
methodological reason and updating the corresponding notebooks, README and
result interpretation together.

## Metrics and reporting

Every model (LR and XGBoost, SW and EW) must report:

- sliding-window or expanding-window RMSE (mean across folds);
- sliding-window or expanding-window MAE (mean across folds);
- sliding-window or expanding-window R2 (mean across folds);
- sliding-window or expanding-window residual standard deviation;
- sample standard deviation of window-level metric estimates;
- bootstrap confidence intervals for holdout RMSE, MAE and R2;
- final sequential-holdout RMSE;
- final sequential-holdout MAE;
- final sequential-holdout R2;
- COS_MAE;
- COS_RMSE.

COS metrics follow:

```text
COS_MAE  = 0.5 * (MAE_SW_or_EW / MAE_final)  + 0.5 * (STD_SW_or_EW / STD_final)
COS_RMSE = 0.5 * (RMSE_SW_or_EW / RMSE_final) + 0.5 * (STD_SW_or_EW / STD_final)
```

Where `STD` refers to the residual standard deviation of each respective block.

Remember that COS confidence intervals are descriptive because sliding windows overlap (SW) or have growing training sets (EW).

## Reproducibility

Before finishing changes, check whether the scripts still run with:

```bash
# Linux / macOS
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_lr_sw.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_lr_ew.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_xgb_sw.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_xgb_ew.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_lstm.py

# Window size sweep
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/window_size_sweep.py
```

```powershell
# Windows / PowerShell
$env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/model_lr_sw.py
$env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/model_lr_ew.py
$env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/model_xgb_sw.py
$env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/model_xgb_ew.py
$env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/model_lstm.py

# Window size sweep
$env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/window_size_sweep.py
```

Supported values for `TARGET_GP_NAME`:

- `Bahrain Grand Prix`
- `Saudi Arabian Grand Prix`
- `United States Grand Prix`
- `Italian Grand Prix`
- `Hungarian Grand Prix`

If changing logic, test at least one Grand Prix first. If possible, test all supported Grand Prix events.

## Data handling

- Do not overwrite raw files in `Data/`.
- Do not commit generated outputs, caches, logs, parameter dumps or temporary plots unless explicitly requested.
- Do not rename columns casually.
- Do not change cleaned datasets in `Scripts/ModelData/` unless the task is specifically about data preparation.
- Be careful with missing values and categorical levels not seen during training.

## Academic writing awareness

This repository supports a paper/TCC. When changing code:

- preserve methodological consistency;
- avoid changes that make previous results incomparable;
- explain changes that affect reported metrics;
- explicitly state when a result changed because of feature-set revisions, such as circuit-specific removals;
- prefer deterministic behavior where possible;
- set random seeds when using stochastic methods;
- avoid optimistic evaluation protocols.

## Forbidden changes unless explicitly requested

- Removing the sequential holdout.
- Using random train/test split.
- Shuffling race laps.
- Sorting only by `LapNumber` without also sorting by `Year`.
- Fitting preprocessing on the full dataset before splitting.
- Tuning XGBoost on the final holdout.
- Removing `LapTime_prev` without methodological justification.
- Changing the target variable.
- Removing COS metrics.
- Removing residual STD or sample STD from reporting.
- Removing bootstrap confidence intervals from holdout metrics.
- Replacing the experiment protocol with cross-validation that ignores race order.
- Using a shared Optuna search space across all circuits instead of circuit-specific spaces.
- Changing the Optuna sampler away from TPE without explicit justification.
- Running Optuna without prior baseline runs to inform the search space.
- Leaving notebooks inconsistent with scripts.
- Changing script methodology without updating the corresponding notebooks.
- Leaving README or notebook markdown outdated after code changes.
- Duplicating metric calculation logic across multiple files.
- Changing the LSTM validation protocol (single sequential split) to sliding/expanding window without explicit justification.
- Changing the LSTM search space version or tuning strategy without updating `LSTM_SEARCH_SPACE_VERSION` / `LSTM_TUNING_STRATEGY` constants and clearing saved params.
- Fitting LSTM transformers (imputer, feature scaler, target scaler) on validation or holdout data.
- Using saved LSTM params when `search_space_version`, `tuning_strategy`, or `n_trials` do not match the current code.
- Changing the final selected method/window size for a Grand Prix (see "Final
  selected configuration per Grand Prix") without explicitly documenting the
  methodological reason and updating the YAML, README and notebooks together.

## Expected behavior from Claude

When asked to modify the project:

1. Inspect the relevant script, notebook, README and configuration files first.
2. Identify the smallest safe change.
3. Preserve the temporal modeling protocol (Year → LapNumber ordering, 80/20 split, SW/EW inside the modeling block, single sequential split for LSTM).
4. Keep notebooks and scripts methodologically consistent.
5. Move repeated constants and paths to YAML configuration files when appropriate.
6. Reuse shared functions for metrics and repeated logic.
7. Update README and notebook markdown when the change affects them.
8. Explain any impact on metrics, leakage risk or reproducibility.
9. For methodological changes, update the relevant notebooks so the narrative, code and outputs match the scripts.
10. Avoid unnecessary formatting-only diffs.