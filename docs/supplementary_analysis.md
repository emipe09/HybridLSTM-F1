# Supplementary Analysis & Insight Layer

The scripts catalogued here are **complementary** to the paper. They consume the trained
models / saved predictions and produce extra figures, tables and diagnostics used for
insight extraction (paper §3.2 "Insight Extraction") and thesis reporting. **None of these
plots/tables appear in the KDMiLe paper itself** — the paper reports only Table II.

All outputs are written under `Scripts/Results/` (git-ignored). The core modeling scripts
(`model_lr_ew.py`, `model_xgb_ew.py`, `model_lstm_hybrid.py`, plus `run_experiment.py` and
the `*_utils.py` helpers) are documented in the [README](../README.md) and
[paper_code_map.md](paper_code_map.md), not here.

## Residual diagnostics
| Script | Produces |
|---|---|
| `plot_lr_residuals_qq.py` | Normal Q-Q plots of LR residuals (per circuit + grid). → `lr_residual_qq/` |
| `plot_lr_residuals_vs_fitted.py` | LR residuals vs fitted (homoscedasticity). → `lr_residual_vs_fitted/` |
| `plot_cdf_residuals.py` | Empirical CDF of holdout residuals, LR vs XGBoost vs Hybrid. → `cdf_residuals/` |
| `plot_hybrid_residuals_qq_top3.py` | Q-Q of hybrid residuals for Italy's top-3 drivers. → `hybrid_residual_qq/` |
| `plot_hybrid_residuals_vs_laptime.py` | Hybrid residuals vs lap time (+ Pearson/Spearman). → `hybrid_residual_vs_laptime/` |
| `analyze_residuals_by_finish_group.py` | Holdout residuals grouped by 2025 finish bands (P1-5…P16-20). → `residuals_by_finish_group/` |

## Driver-level analysis
| Script | Produces |
|---|---|
| `plot_cdf_driver_metrics.py` | CDFs of per-driver MAE/RMSE/R² across models. → `cdf_driver_metrics/` |
| `plot_boxplot_driver_metrics.py` | Boxplots of per-driver metrics per circuit. → `boxplot_driver_metrics/` |
| `plot_driver_holdout_timeseries.py` | Per-driver holdout actual vs predicted (+95% band). → `driver_holdout_timeseries/` |
| `extract_driver_holdout_metrics.py` | Per-driver holdout metrics (RMSE/MAE/R² + bootstrap CI). Console/CSV. |
| `extract_driver_hybrid_holdout.py` | Same, sliced from the hybrid holdout predictions. |

## Data characterization / EDA
| Script | Produces |
|---|---|
| `plot_hist_laptime_splits.py` | Lap-time histograms over train/val/test splits. → `hist_laptime_splits/` |
| `plot_density_laptime_splits.py` | KDE of lap times over splits. → `density_laptime_splits/` |
| `eda_val_holdout_hybrid.py` | Validation-vs-holdout EDA (histograms, CDFs, boxplots, compound usage). → `eda_val_holdout/` |
| `plot_boxplot_weather_year_italy.py` | Italy weather + lap-time boxplots by year (paper Fig. 2 style). → `boxplot_weather_year/` |
| `plot_bar_compound_year_italy.py` | Italy compound usage per season. → `bar_compound_year/` |

## Experiments & interpretability
| Script | Produces |
|---|---|
| `experiment_rq1_lr_comparison.py` | RQ1 motivation: LR per-driver vs full-circuit. → `rq1_lr_*.csv`, `tables_rq1_lr_comparison.tex` |
| `model_interpretability.py` | LR coefficients, XGBoost importance, SHAP summary/force plots. → `model_interpretability/<circuit>/` |

## Methodological sensitivity baselines (driver-filtered)
Complementary baselines that filter to a single driver before the temporal split. They do
**not** replace the main EW models or change any reported result.
- `model_lr_ew_driver.py` — driver-filtered LR-EW.
- `model_xgb_ew_driver.py` — driver-filtered XGBoost-EW.
- `model_lstm_driver.py` — driver-filtered standalone LSTM (absolute target).
