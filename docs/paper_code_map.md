# Paper ↔ Code Alignment Audit

Maps every methodology step of the paper *"Multi-Circuit Formula 1 Lap Time Prediction:
A Hybrid Deep Learning Approach for Race Pace Analysis"* (KDMiLe 2026) to the code that
implements it, and records the divergences found between the written paper and the code.
**The code is the source of truth unless noted otherwise.**

---

## 1. Step → code map (paper §3)

| Paper step | Implementation |
|---|---|
| FastF1 collection, 5 GPs, 2022–2025, main race only | Upstream collection produced `Data/<circuit>/Race/{Laps,Weather}/` (already carries `LapTime_seconds`, `pirelliCompound`, `IsAccurate`). Not re-run in repo — see [data_pipeline.md](data_pipeline.md). |
| Pirelli C1–C5 web scraping | Upstream; mapping stored in [`Utils/compounds.json`](../Utils/compounds.json). |
| Outlier removal (rain / SC / VSC / Pit In-Out) + 95th-percentile cutoff | `prepare_data.py` → lap-to-lap `laps_diff` filtered to its [5th, 95th]-percentile band. |
| Median imputation of numeric features | `xgb_utils.build_xgb_matrix` / `modeling_utils` (`fillna(median)`) at model time. |
| One-hot encoding (Driver, Team, pirelliCompound) | `modeling_utils.align_one_hot`. |
| `TempDelta = TrackTemp − AirTemp` | `prepare_data.py` (`build_cleaned_frame`). |
| RBF kernel on multimodal weather features | `prepare_data.py` (`RBF_GAMMA=0.1`, median-centred). |
| Normalization (mean 0, std 1; target unscaled) | `modeling_utils` (`StandardScaler`); `LapTime_seconds` never scaled. |
| Expanding-window validation, 80/20, window 20%, step 4% | `modeling_utils.build_expanding_windows`. |
| Window-size sweep 5%–50% | `*_window_ratio` keys in `configs/*.yaml`. |
| Linear Regression baseline | `model_lr_ew.py`. |
| XGBoost + Optuna (100 trials, early stopping, median per window) | `model_xgb_ew.py`, `xgb_utils.py` — see [hyperparameters.md](hyperparameters.md). |
| Hybrid LR-LSTM (residual learning) | `model_lstm_hybrid.py` + `model_lstm_baseline.py`, `baseline_utils.py`. |
| RMSE / MAE / R² with 95% CI | `modeling_utils.calc_holdout_ci` (bootstrap, 1000 resamples, percentile CI). |
| COS indicator (α=β=0.5) | `modeling_utils.summarize_cos` / `calc_cos_metric`. |

### Instance counts (verified)
`prepare_data.py` reproduces the dataset sizes reported in the paper **exactly**:

| | Bahrain | Saudi | USA | Italy | Hungary |
|---|---|---|---|---|---|
| Paper §3.1 | 3,161 | 2,660 | 3,063 | 2,994 | 4,213 |
| Code output | 3,161 | 2,660 | 3,063 | 2,994 | 4,213 |

---

## 2. Divergences to resolve

| ID | Where | Paper says | Code does | Decision |
|---|---|---|---|---|
| **D1** | USA feature selection (§3.1 vs `configs/usa.yaml`) | Text says remove **Year** and **TrackTemp** | Removes **Year** and **TempDelta**; keeps `TrackTemp_RBF_Median` | **Code is the truth** (confirmed by the authors: the variables actually removed for USA were `Year` and `TempDelta`, with `TrackTemp` retained). `usa.yaml` is correct as-is; the **paper §3.1 wording** mis-states the thermal variable and should be corrected to "Year and TempDelta were removed". |
| **D2** | COS equation (§4 Eq. 1) | `α·(Perf_train/Perf_test) + β·(σ_train/σ_test)` | Computes `final/sliding` = `test/train` (reciprocal). Table II values match the code, e.g. Bahrain LR `COS_RMSE=0.95 = 0.5·(0.31/0.36)+0.5·(0.314/0.306)` | **Code is the truth** (it produced the published numbers). Eq. (1) wording is inverted and should be corrected by the authors. COS ≈ 1 still means train/test parity either way. |
| **D3** | `window_step_ratio` | Step size of 4% | EW always advances by `val_size = window_ratio·(1−train_ratio) = 4%`; the YAML key `window_step_ratio: 0.20` is **unused/dead**. Effective behavior matches the paper (4%). | Documentation only; optionally remove the dead key from the YAMLs. |
| **D4** | `WindDirection` | Not in the data dictionary nor the RBF list | `WindDirection_RBF_Median` is created in `prepare_data.py` but is **not** used as a feature in any circuit's `numerical_features`. | Documentation only; harmless leftover column. |
| **D5** | Paper §3.3 placeholder `[Alex: Falta acrescentar...]` | Hybrid-residual description, embeddings, and LSTM hyperparameter tuning are *missing* | Fully implemented in code. | [hyperparameters.md](hyperparameters.md) supplies the technical content for the authors to complete the text. |
| **D6** | 95th-percentile filter (§3.1) | "the 95th percentile was used as the cutoff threshold" | Filter is a **two-sided [5th, 95th] band on the lap-to-lap delta** `laps_diff`, not a per-feature 95th-percentile cut. | Documentation only; wording could be made precise in the paper. |

Per-circuit feature selection in the code: USA (−Year, −TempDelta; keeps TrackTemp),
Saudi (−TrackTemp), Hungary (−Humidity), Bahrain & Italy (all retained). All match the
authors' intent; only the USA **wording** in the paper text needs fixing (D1).
