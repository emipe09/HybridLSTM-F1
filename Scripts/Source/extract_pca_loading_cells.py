"""Extract PCA loading cells from all circuit notebooks.

The script reads the configured notebook folder, selects only cells that build
or call PCA loading plots, and writes a compact notebook plus a plain Python
file under Scripts/Results/pca_loading_cells by default.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from modeling_utils import load_simple_yaml, safe_gp_name


DEFAULT_NOTEBOOKS = [
    ("Bahrain Grand Prix", "bahrain.yaml", "Notebook_Bahrain.ipynb", "Bahrain"),
    ("Saudi Arabian Grand Prix", "saudi.yaml", "Notebook_Saudi.ipynb", "Saudi Arabia"),
    ("United States Grand Prix", "usa.yaml", "Notebook_USA.ipynb", "USA"),
    ("Italian Grand Prix", "italy.yaml", "Notebook_Italia.ipynb", "Italy"),
    ("Hungarian Grand Prix", "hungary.yaml", "Notebook_Hungary.ipynb", "Hungary"),
]

DEFAULT_MATCH_TERMS = [
    "Loadings PCA",
    "plot_pca_loadings",
    "loadings = pca.components_",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract only PCA loading cells from all Grand Prix notebooks."
    )
    parser.add_argument(
        "--notebooks-dir",
        default="Scripts/Notebooks",
        help="Directory containing the circuit notebooks. Default: Scripts/Notebooks.",
    )
    parser.add_argument(
        "--output-dir",
        default="Scripts/Results/pca_loading_cells",
        help="Directory for extracted artifacts. Default: Scripts/Results/pca_loading_cells.",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep notebook cell outputs. By default outputs are stripped.",
    )
    parser.add_argument(
        "--skip-cell-export",
        action="store_true",
        help="Only save PCA loading images; skip the extracted notebook and Python cell exports.",
    )
    return parser.parse_args()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def is_pca_loading_cell(cell: dict, match_terms: list[str]) -> bool:
    text = source_text(cell)
    return any(term in text for term in match_terms)


def strip_outputs(cell: dict) -> dict:
    clean_cell = copy.deepcopy(cell)
    if clean_cell.get("cell_type") == "code":
        clean_cell["outputs"] = []
        clean_cell["execution_count"] = None
    return clean_cell


def make_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(text),
    }


def extract_cells_from_notebook(notebook_path: Path, keep_outputs: bool) -> list[dict]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    extracted = []

    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if not is_pca_loading_cell(cell, DEFAULT_MATCH_TERMS):
            continue

        extracted_cell = copy.deepcopy(cell) if keep_outputs else strip_outputs(cell)
        extracted_cell.setdefault("metadata", {})
        extracted_cell["metadata"] = {
            **extracted_cell["metadata"],
            "source_notebook": notebook_path.name,
            "source_cell_index": cell_index,
        }
        extracted.append(extracted_cell)

    return extracted


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
    target_col = str(config["target_col"])
    numerical_features = list(config["numerical_features"])
    categorical_features = list(config["categorical_features"])

    num_cols = [target_col] + [col for col in numerical_features if col != target_col]
    num_cols = [col for col in num_cols if col in laps_cleaned.columns]
    cat_cols = [col for col in categorical_features if col in laps_cleaned.columns]

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
        ],
        "teams": [col for col in feature_names if col.startswith("Team_")],
        "drivers": [col for col in feature_names if col.startswith("Driver_")],
        "years": [col for col in feature_names if col.startswith("Year_")],
    }


def group_title(group_name: str, circuit_label: str) -> str:
    title_map = {
        "lap_time_and_tyres": "PCA Loadings - Lap Time and Tyres",
        "weather": "PCA Loadings - Weather",
        "teams": "PCA Loadings - Teams",
        "drivers": "PCA Loadings - Drivers",
        "years": "PCA Loadings - Years",
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


def save_pca_loading_images(repo_root: Path, output_dir: Path) -> list[dict]:
    image_results = []
    images_root = output_dir / "images"

    for grand_prix, config_name, _notebook_name, circuit_label in DEFAULT_NOTEBOOKS:
        config_path = repo_root / "configs" / config_name
        config = load_simple_yaml(config_path)
        cleaned_data_path = build_cleaned_data_path(repo_root, config)
        laps_cleaned = pd.read_csv(cleaned_data_path)

        pca, feature_names = prepare_pca_inputs(laps_cleaned, config)
        groups = pca_loading_groups(feature_names)

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

    return image_results


def build_extracted_notebook(extracted_by_gp: list[tuple[str, str, str, str, list[dict]]]) -> dict:
    cells = [
        make_markdown_cell(
            "# PCA Loading Cells\n\n"
            "This notebook contains only the PCA loading cells extracted from the circuit notebooks.\n"
            "It assumes that the PCA preparation cells from the original notebooks have already run."
        )
    ]

    for grand_prix, _config_name, notebook_name, _circuit_label, cells_for_gp in extracted_by_gp:
        cells.append(make_markdown_cell(f"## {grand_prix}\n\nSource notebook: `{notebook_name}`"))
        if cells_for_gp:
            cells.extend(cells_for_gp)
        else:
            cells.append(make_markdown_cell("_No PCA loading cells found._"))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_python_export(extracted_by_gp: list[tuple[str, str, str, str, list[dict]]]) -> str:
    blocks = [
        '"""PCA loading cells extracted from all circuit notebooks.',
        "",
        "This file is generated by Scripts/Source/extract_pca_loading_cells.py.",
        '"""',
        "",
    ]

    for grand_prix, _config_name, notebook_name, _circuit_label, cells_for_gp in extracted_by_gp:
        blocks.append("# " + "=" * 78)
        blocks.append(f"# {grand_prix} | {notebook_name}")
        blocks.append("# " + "=" * 78)
        blocks.append("")

        for cell in cells_for_gp:
            cell_index = cell.get("metadata", {}).get("source_cell_index", "unknown")
            blocks.append(f"# Source cell: {cell_index}")
            blocks.append(source_text(cell).rstrip())
            blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def main():
    args = parse_args()
    repo_root = repo_root_from_script()
    notebooks_dir = resolve_repo_path(repo_root, args.notebooks_dir)
    output_dir = resolve_repo_path(repo_root, args.output_dir)

    if not notebooks_dir.exists():
        raise FileNotFoundError(f"Notebook directory not found: {notebooks_dir}")

    extracted_by_gp = []
    for grand_prix, config_name, notebook_name, circuit_label in DEFAULT_NOTEBOOKS:
        notebook_path = notebooks_dir / notebook_name
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
        cells = extract_cells_from_notebook(notebook_path, keep_outputs=args.keep_outputs)
        extracted_by_gp.append((grand_prix, config_name, notebook_name, circuit_label, cells))

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_output_path = output_dir / "pca_loading_cells_all_gps.ipynb"
    python_output_path = output_dir / "pca_loading_cells_all_gps.py"
    manifest_output_path = output_dir / "pca_loading_cells_manifest.json"
    images_manifest_output_path = output_dir / "pca_loading_images_manifest.json"

    if not args.skip_cell_export:
        extracted_notebook = build_extracted_notebook(extracted_by_gp)
        notebook_output_path.write_text(json.dumps(extracted_notebook, indent=1), encoding="utf-8")
        python_output_path.write_text(build_python_export(extracted_by_gp), encoding="utf-8")

    image_results = save_pca_loading_images(repo_root, output_dir)
    images_manifest_output_path.write_text(json.dumps(image_results, indent=2), encoding="utf-8")

    manifest = {
        "match_terms": DEFAULT_MATCH_TERMS,
        "keep_outputs": args.keep_outputs,
        "notebooks": [
            {
                "grand_prix": grand_prix,
                "config": config_name,
                "notebook": notebook_name,
                "extracted_cells": len(cells),
                "source_cell_indices": [
                    cell.get("metadata", {}).get("source_cell_index") for cell in cells
                ],
            }
            for grand_prix, config_name, notebook_name, _circuit_label, cells in extracted_by_gp
        ],
        "outputs": {
            "notebook": None if args.skip_cell_export else str(notebook_output_path),
            "python": None if args.skip_cell_export else str(python_output_path),
            "manifest": str(manifest_output_path),
            "images_manifest": str(images_manifest_output_path),
        },
    }
    manifest_output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("--- PCA LOADING CELL EXTRACTION AND IMAGE EXPORT ---")
    for grand_prix, _config_name, notebook_name, _circuit_label, cells in extracted_by_gp:
        indices = [cell.get("metadata", {}).get("source_cell_index") for cell in cells]
        print(f"{grand_prix}: {len(cells)} cells from {notebook_name} | indices={indices}")
    print("\nSaved PCA loading images:")
    for result in image_results:
        saved_count = sum(1 for image in result["images"] if image["saved"])
        print(f"{result['grand_prix']}: {saved_count} PNG files")
    print("\nOutputs:")
    if not args.skip_cell_export:
        print(f"- {notebook_output_path}")
        print(f"- {python_output_path}")
    print(f"- {manifest_output_path}")
    print(f"- {images_manifest_output_path}")
    print(f"- {output_dir / 'images'}")


if __name__ == "__main__":
    main()
