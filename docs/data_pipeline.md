# Data Pipeline & Provenance

How the raw race files become the cleaned datasets the models consume, and where the
reproducible boundary of this repository lies.

```
FastF1 API + Pirelli C1–C5 scraping        (upstream, NOT in repo)
        │
        ▼
Data/<circuit>/Race/{Laps,Weather}/*.csv   (raw per-year files, tracked in git)
        │   prepare_data.py
        ▼
Scripts/ModelData/<GP>/<safe_gp_name>_cleaned_data.csv   (model input, tracked)
        │   model_lr_ew.py / model_xgb_ew.py / model_lstm_hybrid.py
        ▼
Scripts/Results/...                        (models, metrics, figures — git-ignored)
```

## Reproducible boundary
The repository is reproducible **from `Data/` onward**. The raw laps files already contain
`LapTime_seconds`, `pirelliCompound` and `IsAccurate`, i.e. the FastF1 collection and the
Pirelli C1–C5 web scraping (paper §3.1) were performed by an upstream step that is **not**
included here. The Pirelli season→compound mapping it used is preserved in
[`Utils/compounds.json`](../Utils/compounds.json).

## `prepare_data.py` — Data/ → cleaned_data.csv
[`Scripts/Source/prepare_data.py`](../Scripts/Source/prepare_data.py) is a faithful
extraction of the data-preparation cells that previously lived only in the per-circuit
notebooks (now removed). It rebuilds each `cleaned_data.csv`:

1. Load per-year laps + weather (2022–2025); keep `IsAccurate == True`; drop rows missing
   `LapTime_seconds` / `TyreLife` / `pirelliCompound`.
2. `LapTime_prev` = previous lap time within `[Year, Driver, Stint]`.
3. `merge_asof` each lap to the nearest preceding weather sample (≤ 60 s, per Year).
4. `TempDelta = TrackTemp − AirTemp`; normalize team names across rebrands.
5. **Outlier filter:** keep laps whose lap-to-lap delta `laps_diff` lies in the
   [5th, 95th]-percentile band (also drops the first lap of each stint, which has no
   `LapTime_prev`).
6. **RBF kernel** (γ = 0.1, median-centred) for `TrackTemp, Humidity, Pressure, WindSpeed,
   WindDirection, TempDelta` → `*_RBF_Median` columns.
7. Write the full frame to `Scripts/ModelData/<GP>/<safe_gp_name>_cleaned_data.csv`.

### Usage
Linux/macOS:
```bash
CONFIG_PATH=configs/usa.yaml        python Scripts/Source/prepare_data.py
TARGET_GP_NAME="Italian Grand Prix" python Scripts/Source/prepare_data.py
```

Windows/PowerShell:
```powershell
$env:CONFIG_PATH = "configs/usa.yaml";          python Scripts/Source/prepare_data.py
$env:TARGET_GP_NAME = "Italian Grand Prix";     python Scripts/Source/prepare_data.py
```

### Verification
Re-running for all five circuits reproduces the exact instance counts reported in the paper
(Bahrain 3,161 · Saudi 2,660 · USA 3,063 · Italy 2,994 · Hungary 4,213) and the same column
schema as the committed `cleaned_data.csv` files. See
[paper_code_map.md](paper_code_map.md#instance-counts-verified).

## Notes
- The cleaned CSV keeps raw weather columns, `laps_diff`, and the `*_RBF_Median` features;
  the YAML `numerical_features` list selects which columns each model actually uses.
- Median imputation and `StandardScaler` are applied **at model time** (per fold, on training
  data only), not baked into the cleaned CSV.
- `WindDirection_RBF_Median` is produced but unused by every circuit config (see D4 in
  [paper_code_map.md](paper_code_map.md)).
