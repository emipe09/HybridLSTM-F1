"""Gera uma tabela LaTeX por corrida com a media do tempo de volta por numero
da volta, para cada ano (colunas = anos, linhas = numero da volta).

As celulas que correspondem a voltas de VALIDACAO do LSTM_hybrid sao destacadas
em azul e as de HOLDOUT em vermelho. A particao e definida pela mesma chave
temporal composta (Year*10000 + LapNumber) usada pelo modelo.

Saida: Scripts/Results/tables_laptime_by_lap.tex
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from modeling_utils import (
    build_sequential_split,
    load_cleaned_data,
    prepare_raw_features,
    select_modeling_columns,
)

SCRIPT_PATH = Path(__file__)
REPO_ROOT = SCRIPT_PATH.resolve().parents[2]
OUT_PATH = REPO_ROOT / "Scripts/Results/tables_laptime_by_lap.tex"

CIRCUITS = [
    ("configs/bahrain.yaml", "Bahrein", "bahrein"),
    ("configs/saudi.yaml", "Arabia Saudita", "arabia"),
    ("configs/hungary.yaml", "Hungria", "hungria"),
    ("configs/italy.yaml", "Italia", "italia"),
    ("configs/usa.yaml", "Estados Unidos", "eua"),
]


def circuit_data(config_rel: str):
    """Retorna (config, df_valid, val_steps, hold_steps) onde df_valid sao as
    linhas validas (target+features) e *_steps sao conjuntos de chaves compostas."""
    os.environ["CONFIG_PATH"] = config_rel
    _, config, _, df_base = load_cleaned_data(SCRIPT_PATH)
    target_col = str(config["target_col"])
    lap_col = str(config["lap_col"])

    num_cols, cat_cols = select_modeling_columns(df_base, config)
    _, _, valid_indices = prepare_raw_features(df_base, num_cols, cat_cols, target_col)

    step_series, _, _, model_idx, holdout_idx, *_ = build_sequential_split(
        df_base, valid_indices, float(config["holdout_ratio"]), lap_col
    )
    model_step = step_series.loc[model_idx]
    model_steps = np.sort(model_step.unique())
    n_train = max(2, int(np.floor(len(model_steps) * float(config["window_train_ratio"]))))
    val_steps = set(model_steps[n_train:])
    hold_steps = set(np.sort(step_series.loc[holdout_idx].unique()))

    df_valid = df_base.loc[valid_indices, ["Year", "LapNumber", target_col]].copy()
    df_valid["LapNumber"] = df_valid["LapNumber"].astype(int)
    df_valid["Year"] = df_valid["Year"].astype(int)
    return config, target_col, df_valid, val_steps, hold_steps


def table(config_rel, label, slug) -> str:
    config, target_col, df, val_steps, hold_steps = circuit_data(config_rel)
    years = sorted(df["Year"].unique())
    pivot = df.pivot_table(index="LapNumber", columns="Year",
                           values=target_col, aggfunc="mean")
    pivot = pivot.reindex(columns=years).sort_index()

    L = []
    L.append(r"\begin{table}[ttt]")
    L.append(r"    \centering")
    L.append(
        f"    \\caption{{Media do tempo de volta (s) por numero da volta e por ano "
        f"-- {label}. Celulas em \\textcolor{{blue}}{{azul}} = voltas de validacao "
        f"do LSTM\\_hybrid; em \\textcolor{{red}}{{vermelho}} = voltas de holdout.}}"
    )
    L.append(f"    \\label{{tab:laptime_por_volta_{slug}}}")
    L.append(r"    \scriptsize")
    L.append(r"    \begin{tabular}{l" + "r" * len(years) + r"}")
    L.append(r"        \hline")
    L.append(r"        \rowcolor[HTML]{D9D9D9}")
    head = r"        \textbf{Volta}"
    for y in years:
        head += f" & \\multicolumn{{1}}{{c}}{{\\textbf{{{y}}}}}"
    head += r" \\ \hline"
    L.append(head)

    for lap, row in pivot.iterrows():
        cells = []
        for y in years:
            v = row[y]
            if pd.isna(v):
                cells.append("--")
                continue
            txt = f"{v:.2f}"
            key = y * 10000 + int(lap)
            if key in val_steps:
                txt = f"\\textcolor{{blue}}{{{txt}}}"
            elif key in hold_steps:
                txt = f"\\textcolor{{red}}{{{txt}}}"
            cells.append(txt)
        L.append(f"        \\texttt{{{int(lap)}}} & " + " & ".join(cells) + r" \\ \hline")

    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    blocks = [table(cfg, label, slug) for cfg, label, slug in CIRCUITS]
    out = "\n\n".join(blocks) + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(out, encoding="utf-8")
    print(f"Salvo em: {OUT_PATH}")
    print(f"Tabelas geradas: {len(blocks)}")


if __name__ == "__main__":
    main()
