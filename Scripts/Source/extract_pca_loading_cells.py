"""Generate PCA loading plots from the configured cleaned modeling datasets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from modeling_utils import load_simple_yaml, safe_gp_name


DEFAULT_GRAND_PRIX = [
    ("Bahrain Grand Prix", "bahrain.yaml", "Bahrain"),
    ("Saudi Arabian Grand Prix", "saudi.yaml", "Saudi Arabia"),
    ("United States Grand Prix", "usa.yaml", "USA"),
    ("Italian Grand Prix", "italy.yaml", "Italy"),
    ("Hungarian Grand Prix", "hungary.yaml", "Hungary"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PCA loading plots from cleaned model datasets for all configured Grand Prix."
    )
    parser.add_argument(
        "--output-dir",
        default="Scripts/Results/pca_loading_cells",
        help="Directory for generated PCA artifacts. Default: Scripts/Results/pca_loading_cells.",
    )
    return parser.parse_args()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def build_cleaned_data_path(repo_root: Path, config: dict) -> Path:
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    filename_template = str(config["cleaned_data_filename_template"])
    filename = filename_template.format(target_gp_name=target_gp_name, safe_gp_name=safe_name)
    return resolve_repo_path(repo_root, str(config["model_data_dir"])) / target_gp_name / filename


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def prepare_pca_inputs(laps_cleaned: pd.DataFrame, config: dict) -> tuple[PCA, pd.Index]:
    numerical_features = list(config["numerical_features"])
    categorical_features = list(config["categorical_features"])

    def existing_unique(columns: list[str]) -> list[str]:
        selected = []
        seen = set()
        for col in columns:
            if col in seen or col not in laps_cleaned.columns:
                continue
            selected.append(col)
            seen.add(col)
        return selected

    num_cols = existing_unique(numerical_features)
    cat_cols = existing_unique(categorical_features)

    cat_dummies = pd.get_dummies(laps_cleaned[cat_cols].astype(str), prefix=cat_cols)
    num_df = laps_cleaned[num_cols].copy()
    ml_df_full = pd.concat([num_df, cat_dummies], axis=1)

    imputer_num = SimpleImputer(strategy="mean")
    ml_df_imputed_num = imputer_num.fit_transform(ml_df_full[num_cols])
    num_df_imputed = pd.DataFrame(ml_df_imputed_num, columns=num_cols, index=ml_df_full.index)
    ml_df_imputed_analysis = pd.concat([num_df_imputed, cat_dummies], axis=1)

    scaler = StandardScaler()
    ml_df_scaled = scaler.fit_transform(ml_df_imputed_analysis)

    pca = PCA()
    pca.fit(ml_df_scaled)
    return pca, pd.Index(ml_df_imputed_analysis.columns)


def pca_loading_groups(feature_names: pd.Index) -> dict[str, list[str]]:
    return {
        "lap_time_and_tyres": [
            "LapTime_prev",
            "TyreLife",
            "LapNumber",
            "pirelliCompound_C1",
            "pirelliCompound_C2",
            "pirelliCompound_C3",
            "pirelliCompound_C4",
        ],
        "weather": [
            "Humidity_RBF_Median",
            "Pressure_RBF_Median",
            "WindSpeed_RBF_Median",
            "TrackTemp_RBF_Median",
            "TempDelta_RBF_Median",
            "WindDirection_RBF_Median",
            "Year",
        ],
        "teams": [col for col in feature_names if col.startswith("Team_")],
        "drivers": [col for col in feature_names if col.startswith("Driver_")],
    }


def group_title(group_name: str, circuit_label: str) -> str:
    title_map = {
        "lap_time_and_tyres": "PCA Loadings - Lap Time and Tyres",
        "weather": "PCA Loadings - Weather & Year",
        "teams": "PCA Loadings - Teams",
        "drivers": "PCA Loadings - Drivers",
    }
    return f"{title_map[group_name]} - {circuit_label}"


def plot_pca_loadings_subset_2d_static(pca, subset_features, full_feature_names, title, output_path):
    loadings = pca.components_.T
    pc1 = loadings[:, 0]
    pc2 = loadings[:, 1]

    if not isinstance(full_feature_names, pd.Index):
        full_feature_names = pd.Index(full_feature_names)

    subset_features = [feature for feature in subset_features if feature in full_feature_names]
    if len(subset_features) == 0:
        return []

    idx = [full_feature_names.get_loc(feature) for feature in subset_features]

    plt.figure(figsize=(8, 6))
    plt.scatter(pc1[idx], pc2[idx])

    for feature_index, feature in zip(idx, subset_features):
        plt.text(pc1[feature_index], pc2[feature_index], feature, fontsize=12, ha="center", va="bottom")

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return subset_features


def compact_feature_label(feature_name: str) -> str:
    replacements = {
        "Humidity_RBF_Median": "Humidity",
        "Pressure_RBF_Median": "Pressure",
        "TrackTemp_RBF_Median": "TrackTemp",
        "WindSpeed_RBF_Median": "WindSpeed",
        "TempDelta_RBF_Median": "TempDelta",
        "LapTime_prev": "LapTime_prev",
        "pirelliCompound_": "Compound_",
    }
    label = feature_name
    for old, new in replacements.items():
        label = label.replace(old, new)
    label = label.replace("Driver_", "Drv_").replace("Team_", "Team_")
    return label


def top_component_loadings(
    pca: PCA,
    feature_names: pd.Index,
    grand_prix: str,
    circuit_label: str,
    top_n: int = 5,
) -> list[dict]:
    loadings = pca.components_.T
    rows = []
    for component_index, component_name in enumerate(["PC1", "PC2"]):
        component_loadings = loadings[:, component_index]
        top_indices = np.argsort(np.abs(component_loadings))[::-1][:top_n]
        for rank, feature_index in enumerate(top_indices, start=1):
            loading = float(component_loadings[feature_index])
            rows.append(
                {
                    "grand_prix": grand_prix,
                    "circuit_label": circuit_label,
                    "component": component_name,
                    "rank": rank,
                    "feature": str(feature_names[feature_index]),
                    "feature_label": compact_feature_label(str(feature_names[feature_index])),
                    "loading": loading,
                    "abs_loading": abs(loading),
                }
            )
    return rows


def plot_grouped_top_component_loadings(loadings_df: pd.DataFrame, output_path: Path, top_n: int = 5) -> None:
    circuits = list(dict.fromkeys(loadings_df["circuit_label"]))
    colors = ["#2f6f9f", "#d55e00", "#009e73", "#cc79a7", "#f0b429"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    x = np.arange(len(circuits))
    width = 0.14
    offsets = (np.arange(top_n) - (top_n - 1) / 2) * width

    for ax, component in zip(axes, ["PC1", "PC2"]):
        component_df = loadings_df[loadings_df["component"] == component]
        for rank in range(1, top_n + 1):
            rank_df = (
                component_df[component_df["rank"] == rank]
                .set_index("circuit_label")
                .reindex(circuits)
                .reset_index()
            )
            bar_positions = x + offsets[rank - 1]
            bars = ax.bar(
                bar_positions,
                rank_df["abs_loading"],
                width=width,
                label=f"Top {rank}",
                color=colors[rank - 1],
            )
            for bar, feature_label in zip(bars, rank_df["feature_label"]):
                if pd.isna(feature_label):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    feature_label,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                )

        ax.set_title(f"Top {top_n} absolute PCA loadings by circuit - {component}")
        ax.set_ylabel("Absolute loading")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(0, component_df["abs_loading"].max() * 1.28)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(circuits)
    axes[0].legend(ncol=top_n, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pca_loading_images(repo_root: Path, output_dir: Path) -> list[dict]:
    image_results = []
    top_loading_rows = []
    images_root = output_dir / "images"

    for grand_prix, config_name, circuit_label in DEFAULT_GRAND_PRIX:
        config_path = repo_root / "configs" / config_name
        config = load_simple_yaml(config_path)
        cleaned_data_path = build_cleaned_data_path(repo_root, config)
        laps_cleaned = pd.read_csv(cleaned_data_path)

        pca, feature_names = prepare_pca_inputs(laps_cleaned, config)
        groups = pca_loading_groups(feature_names)
        top_loading_rows.extend(top_component_loadings(pca, feature_names, grand_prix, circuit_label))

        gp_output_dir = images_root / safe_gp_name(grand_prix)
        gp_output_dir.mkdir(parents=True, exist_ok=True)

        gp_result = {
            "grand_prix": grand_prix,
            "config": str(config_path),
            "cleaned_data": str(cleaned_data_path),
            "feature_count": len(feature_names),
            "images": [],
        }

        for group_name, subset_features in groups.items():
            title = group_title(group_name, circuit_label)
            output_path = gp_output_dir / f"{slugify(title)}.png"
            plotted_features = plot_pca_loadings_subset_2d_static(
                pca,
                subset_features,
                feature_names,
                title,
                output_path,
            )
            gp_result["images"].append(
                {
                    "group": group_name,
                    "title": title,
                    "path": str(output_path),
                    "plotted_features": plotted_features,
                    "plotted_feature_count": len(plotted_features),
                    "saved": bool(plotted_features),
                }
            )

        image_results.append(gp_result)

    top_loadings_df = pd.DataFrame(top_loading_rows)
    top_loadings_csv = output_dir / "top5_pc1_pc2_loadings_by_track.csv"
    top_loadings_png = output_dir / "top5_pc1_pc2_loadings_by_track.png"
    top_loadings_df.to_csv(top_loadings_csv, index=False)
    plot_grouped_top_component_loadings(top_loadings_df, top_loadings_png)
    image_results.append(
        {
            "grand_prix": "all",
            "images": [
                {
                    "group": "top_pc1_pc2_loadings_by_track",
                    "title": "Top 5 PC1 and PC2 Loadings by Track",
                    "path": str(top_loadings_png),
                    "data": str(top_loadings_csv),
                    "saved": True,
                }
            ],
        }
    )

    return image_results


def main():
    args = parse_args()
    repo_root = repo_root_from_script()
    output_dir = resolve_repo_path(repo_root, args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output_path = output_dir / "pca_loading_manifest.json"
    images_manifest_output_path = output_dir / "pca_loading_images_manifest.json"

    image_results = save_pca_loading_images(repo_root, output_dir)
    images_manifest_output_path.write_text(json.dumps(image_results, indent=2), encoding="utf-8")

    manifest = {
        "description": "PCA loading plots generated directly from cleaned modeling datasets.",
        "grand_prix": [
            {
                "grand_prix": grand_prix,
                "config": config_name,
                "circuit_label": circuit_label,
            }
            for grand_prix, config_name, circuit_label in DEFAULT_GRAND_PRIX
        ],
        "outputs": {
            "manifest": str(manifest_output_path),
            "images_manifest": str(images_manifest_output_path),
            "images": str(output_dir / "images"),
            "top5_pc1_pc2_loadings_csv": str(output_dir / "top5_pc1_pc2_loadings_by_track.csv"),
            "top5_pc1_pc2_loadings_png": str(output_dir / "top5_pc1_pc2_loadings_by_track.png"),
        },
    }
    manifest_output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("--- PCA LOADING GENERATION ---")
    print("\nSaved PCA loading images:")
    for result in image_results:
        saved_count = sum(1 for image in result["images"] if image["saved"])
        print(f"{result['grand_prix']}: {saved_count} PNG files")
    print("\nOutputs:")
    print(f"- {manifest_output_path}")
    print(f"- {images_manifest_output_path}")
    print(f"- {output_dir / 'images'}")
    print(f"- {output_dir / 'top5_pc1_pc2_loadings_by_track.csv'}")
    print(f"- {output_dir / 'top5_pc1_pc2_loadings_by_track.png'}")


if __name__ == "__main__":
    main()
