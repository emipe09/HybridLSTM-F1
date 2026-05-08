# AGENTS.md

## Project context

This repository contains research code for multi-circuit Formula 1 lap-time prediction.

The main goal is to model `LapTime_seconds` using FastF1-derived race data, with a temporal validation protocol that simulates real race conditions:

- sliding-window validation inside the first 80% of ordered laps;
- final sequential holdout on the last 20% of ordered laps;
- comparison between Linear Regression and XGBoost models;
- reporting of RMSE, MAE, R², residual standard deviation, COS_MAE and COS_RMSE.

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

- `Scripts/Source/model_lr_sw.py`
- `Scripts/Source/model_xgb_sw.py`

## General rules

- Do not change the temporal validation protocol unless explicitly requested.
- Preserve the final 20% sequential holdout.
- Sliding-window validation must happen only inside the first 80% modeling block.
- Do not introduce data leakage.
- Do not use future laps to predict previous laps.
- Do not shuffle race laps when temporal ordering matters.
- Keep `LapNumber` ordering consistent.
- Preserve the target column: `LapTime_seconds`.
- Preserve article-facing metrics unless explicitly asked to change them.
- Do not remove COS metrics.
- Do not simplify the experiment in a way that weakens the paper methodology.

## Notebook and script consistency

- Keep notebook pipelines clean and consistent with the separated model scripts.
- Notebooks and model scripts must produce equivalent results when using the same data, configuration and random seed.
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
- random seeds.

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
- `LapTime_prev`

Categorical features currently used:

- `Driver`
- `Team`
- `pirelliCompound`
- `Year`

Target:

- `LapTime_seconds`

Do not add or remove features without explaining the methodological impact.

## Linear Regression

For `model_lr_sw.py`:

- Preserve preprocessing with imputation, scaling and categorical encoding.
- Fit preprocessing only on the training portion of each split/window.
- Never fit scalers, imputers or encoders using validation or holdout data.
- Keep results comparable with the XGBoost script.

## XGBoost

For `model_xgb_sw.py`:

- Preserve Optuna tuning unless explicitly asked to disable it.
- Do not tune hyperparameters using the final holdout.
- The final holdout must remain untouched until final evaluation.
- Generated parameter dumps or run outputs should remain ignored by Git when appropriate.

## Metrics and reporting

Keep reporting:

- sliding-window RMSE;
- sliding-window MAE;
- sliding-window R²;
- sliding-window residual standard deviation;
- final sequential-holdout RMSE;
- final sequential-holdout MAE;
- final sequential-holdout R²;
- bootstrap confidence intervals for holdout metrics when present;
- COS_MAE;
- COS_RMSE.

COS metrics follow:

```text
COS_MAE  = 0.5 * (MAE_SW / MAE_final)  + 0.5 * (STD_SW / STD_final)
COS_RMSE = 0.5 * (RMSE_SW / RMSE_final) + 0.5 * (STD_SW / STD_final)
```

Remember that COS confidence intervals are descriptive because sliding windows overlap.

## Reproducibility

Before finishing changes, check whether the scripts still run with:

```bash
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_lr_sw.py
TARGET_GP_NAME="Bahrain Grand Prix" python Scripts/Source/model_xgb_sw.py
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
- prefer deterministic behavior where possible;
- set random seeds when using stochastic methods;
- avoid optimistic evaluation protocols.

## Forbidden changes unless explicitly requested

- Removing the sequential holdout.
- Using random train/test split.
- Shuffling race laps.
- Fitting preprocessing on the full dataset before splitting.
- Tuning XGBoost on the final holdout.
- Removing `LapTime_prev` without methodological justification.
- Changing the target variable.
- Removing COS metrics.
- Replacing the experiment protocol with cross-validation that ignores race order.
- Leaving notebooks inconsistent with scripts.
- Leaving README or notebook markdown outdated after code changes.
- Duplicating metric calculation logic across multiple files.

## Expected behavior from Codex

When asked to modify the project:

1. Inspect the relevant script, notebook, README and configuration files first.
2. Identify the smallest safe change.
3. Preserve the temporal modeling protocol.
4. Keep notebooks and scripts methodologically consistent.
5. Move repeated constants and paths to YAML configuration files when appropriate.
6. Reuse shared functions for metrics and repeated logic.
7. Update README and notebook markdown when the change affects them.
8. Explain any impact on metrics, leakage risk or reproducibility.
9. Avoid unnecessary formatting-only diffs.
