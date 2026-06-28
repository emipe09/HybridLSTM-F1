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
