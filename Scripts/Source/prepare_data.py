"""Reproducible data-preparation step: raw ``Data/`` CSVs -> ``ModelData/*_cleaned_data.csv``.

This script rebuilds the cleaned, feature-engineered dataset consumed by every
modeling script (``model_lr_ew.py``, ``model_xgb_ew.py``, ``model_lstm_hybrid.py``...).
It is a faithful extraction of the data-preparation cells that previously lived only
in the per-circuit notebooks, so the pipeline documented in the paper (Section 3.1) is
reproducible from the command line.

Reproducible boundary: the upstream FastF1 collection and the Pirelli C1-C5 web
scraping are NOT performed here. They already produced the per-year race files under
``Data/<circuit>/Race/{Laps,Weather}/`` (the laps files already carry
``LapTime_seconds``, ``pirelliCompound`` and ``IsAccurate``). This script starts from
those files and produces the cleaned dataset.

Steps (matching the notebooks):
  1. Load per-year laps + weather (2022-2025), keep ``IsAccurate == True`` and drop rows
     missing ``LapTime_seconds`` / ``TyreLife`` / ``pirelliCompound``.
  2. ``LapTime_prev`` = previous lap time within ``[Year, Driver, Stint]``.
  3. ``merge_asof`` each lap with the nearest preceding weather sample (<= 60 s, per Year).
  4. ``TempDelta = TrackTemp - AirTemp`` and team-name normalization.
  5. Outlier filter: keep laps whose lap-to-lap delta ``laps_diff`` falls inside the
     [5th, 95th] percentile band (this also drops the first lap of each stint, which has
     no ``LapTime_prev``).
  6. Gaussian RBF-kernel transform (gamma=0.1) of the multimodal weather features,
     centred on each feature's median.
  7. Write the full frame to ``ModelData/<GP>/<safe_gp_name>_cleaned_data.csv``.

Usage (same env conventions as the other scripts):
    CONFIG_PATH=configs/usa.yaml python Scripts/Source/prepare_data.py
    TARGET_GP_NAME="Italian Grand Prix" python Scripts/Source/prepare_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modeling_utils import (
    build_cleaned_data_path,
    load_config,
    resolve_repo_path,
    safe_gp_name,
)

# Year range covered by the paper (single technical-regulation era), end-exclusive.
START_YEAR = 2022
END_YEAR = 2026

# Weather merge tolerance: nearest preceding sample within this many seconds.
WEATHER_TOLERANCE_SECONDS = 60

# Lap-to-lap delta outlier band (percentiles of ``laps_diff``).
DIFF_LOWER_PERCENTILE = 5
DIFF_UPPER_PERCENTILE = 95

# Gaussian RBF-kernel parameters (similarity to each feature's median).
RBF_GAMMA = 0.1
RBF_WEATHER_COLS = ["TrackTemp", "Humidity", "Pressure", "WindSpeed", "WindDirection", "TempDelta"]

# Team-name normalization across seasons (rebrands / engine-supplier renames).
TEAM_MAPPING = {
    "Alfa Romeo Racing": "Kick Sauber",
    "Alfa Romeo": "Kick Sauber",
    "Racing Point": "Aston Martin",
    "Toro Rosso": "Racing Bulls",
    "AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
    "Renault": "Alpine",
}

# Maps the YAML ``target_gp_name`` to the ``Data/<folder>`` subdirectory. Kept here
# (not in the YAML) so the circuit configs stay untouched.
GP_TO_DATA_SUBDIR = {
    "Bahrain Grand Prix": "Bahrain",
    "Saudi Arabian Grand Prix": "Saudi Arabia",
    "United States Grand Prix": "United States",
    "Italian Grand Prix": "Italy",
    "Hungarian Grand Prix": "Hungary",
}


def load_year_frames(laps_dir: Path, weather_dir: Path, safe_name: str):
    laps_by_year, weather_by_year = {}, {}
    for year in range(START_YEAR, END_YEAR):
        laps_file = laps_dir / f"{safe_name}_laps_{year}.csv"
        weather_file = weather_dir / f"{safe_name}_weather_{year}.csv"
        if not (laps_file.exists() and weather_file.exists()):
            print(f"  [!] Missing files for {year}; skipping.")
            continue

        df_laps = pd.read_csv(laps_file)
        df_weather = pd.read_csv(weather_file)
        df_laps["Time"] = pd.to_timedelta(df_laps["Time"])
        df_weather["Time"] = pd.to_timedelta(df_weather["Time"])
        if "LapTime" in df_laps.columns:
            df_laps["LapTime"] = pd.to_timedelta(df_laps["LapTime"])
        df_laps["Year"] = year
        df_weather["Year"] = year
        laps_by_year[year] = df_laps.sort_values("Time")
        weather_by_year[year] = df_weather.sort_values("Time")
    return laps_by_year, weather_by_year


def build_cleaned_frame(laps_by_year: dict, weather_by_year: dict) -> pd.DataFrame:
    combined_laps = pd.concat(laps_by_year.values(), ignore_index=True)
    combined_weather = pd.concat(weather_by_year.values(), ignore_index=True)

    # 1. Keep accurate laps with the fields required downstream.
    clean = combined_laps[combined_laps["IsAccurate"] == True].copy()  # noqa: E712
    clean.dropna(subset=["LapTime_seconds", "TyreLife", "pirelliCompound"], inplace=True)
    clean["Year"] = clean["Year"].astype(int)

    # 2. Previous lap time within each stint.
    clean = clean.sort_values(["Year", "Driver", "Stint", "LapNumber"])
    clean["LapTime_prev"] = clean.groupby(["Year", "Driver", "Stint"])["LapTime_seconds"].shift(1)

    # 3. Attach the nearest preceding weather sample.
    laps_sorted = clean.sort_values("Time").reset_index(drop=True)
    weather_sorted = (
        combined_weather.sort_values("Time")
        .drop_duplicates(subset=["Time", "Year"])
        .reset_index(drop=True)
    )
    laps_sorted["Year"] = laps_sorted["Year"].astype(int)
    weather_sorted["Year"] = weather_sorted["Year"].astype(int)
    merged = pd.merge_asof(
        laps_sorted,
        weather_sorted,
        on="Time",
        by="Year",
        direction="backward",
        tolerance=pd.Timedelta(seconds=WEATHER_TOLERANCE_SECONDS),
    )

    # 4. Engineered TempDelta + team normalization.
    merged["TempDelta"] = merged["TrackTemp"] - merged["AirTemp"]
    merged["Team"] = merged["Team"].replace(TEAM_MAPPING)

    # 5. Outlier filter on the lap-to-lap delta (drops slow/pit/SC laps and first-of-stint).
    merged["laps_diff"] = merged["LapTime_seconds"] - merged["LapTime_prev"]
    diff_data = merged["laps_diff"].dropna()
    low = np.percentile(diff_data, DIFF_LOWER_PERCENTILE)
    high = np.percentile(diff_data, DIFF_UPPER_PERCENTILE)
    cleaned = merged[(merged["laps_diff"] >= low) & (merged["laps_diff"] <= high)].copy()

    # 6. Gaussian RBF-kernel transform of the multimodal weather features.
    for col in RBF_WEATHER_COLS:
        if col not in cleaned.columns:
            continue
        median_val = cleaned[col].median()
        squared_dist = (cleaned[col] - median_val) ** 2
        cleaned[f"{col}_RBF_Median"] = np.exp(-RBF_GAMMA * squared_dist).fillna(0)

    return cleaned


def main():
    repo_root = Path(__file__).resolve().parents[2]
    config, config_path = load_config(repo_root)
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)

    data_subdir = GP_TO_DATA_SUBDIR.get(target_gp_name)
    if data_subdir is None:
        raise KeyError(
            f"No Data/ subdirectory mapping for {target_gp_name!r}. "
            f"Add it to GP_TO_DATA_SUBDIR."
        )

    data_dir = resolve_repo_path(repo_root, str(config["data_dir"])) / data_subdir / "Race"
    laps_dir = data_dir / "Laps"
    weather_dir = data_dir / "Weather"
    output_path = build_cleaned_data_path(repo_root, config)

    print(f"Using config:\n{config_path}")
    print(f"--- DATA PREPARATION: {target_gp_name} ---")
    print(f"Reading raw race data from:\n{data_dir}")

    laps_by_year, weather_by_year = load_year_frames(laps_dir, weather_dir, safe_name)
    if not laps_by_year:
        raise FileNotFoundError(f"No laps/weather files found under {data_dir}.")

    cleaned = build_cleaned_frame(laps_by_year, weather_by_year)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"Wrote {len(cleaned)} cleaned laps to:\n{output_path}")


if __name__ == "__main__":
    main()
