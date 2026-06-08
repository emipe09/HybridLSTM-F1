# AGENTS.md

## Project context

This repository contains research code for multi-circuit Formula 1 lap-time prediction.

The main goal is to model `LapTime_seconds` using FastF1-derived race data, with a temporal validation protocol that simulates real race conditions:

- the full dataset is ordered first by year, then by lap number within each year (e.g. all 2022 laps come before all 2023 laps);
- sliding-window (SW) and expanding-window (EW) validation inside the first 80% of the ordered dataset;
- final sequential holdout on the last 20% of the ordered dataset;
- comparison between Linear Regression and XGBoost models;
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
- `Scripts/Source/window_size_sweep.py` — window size sensitivity analysis across both approaches.

LSTM scripts remain untouched until further notice.

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

- Number of trials: **20** per optimization run.
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

# Window size sweep
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/window_size_sweep.py
```

```powershell
# Windows / PowerShell
$env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/model_lr_sw.py
$env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/model_lr_ew.py
$env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/model_xgb_sw.py
$env:TARGET_GP_NAME="Bahrain Grand Prix"; python Scripts/Source/model_xgb_ew.py

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
- Touching LSTM scripts without explicit instruction.

## Expected behavior from Codex

When asked to modify the project:

1. Inspect the relevant script, notebook, README and configuration files first.
2. Identify the smallest safe change.
3. Preserve the temporal modeling protocol (Year → LapNumber ordering, 80/20 split, SW/EW inside the modeling block).
4. Keep notebooks and scripts methodologically consistent.
5. Move repeated constants and paths to YAML configuration files when appropriate.
6. Reuse shared functions for metrics and repeated logic.
7. Update README and notebook markdown when the change affects them.
8. Explain any impact on metrics, leakage risk or reproducibility.
9. For methodological changes, update the relevant notebooks so the narrative, code and outputs match the scripts.
10. Avoid unnecessary formatting-only diffs.
11. Do not modify LSTM scripts unless explicitly instructed.