"""Gera tabelas LaTeX com o resultado final (posicao de chegada) de cada piloto
por ano, uma tabela para cada um dos 5 circuitos.

Fonte: Data/<Circuito>/Results/*_results_<ano>.csv (coluna ClassifiedPosition).
Legenda: R = abandonou (retired), D = desclassificado, W = nao largou.
"""
import glob
import os
import re

import pandas as pd

OUT_PATH = "Scripts/Results/tables_driver_results.tex"

# (pasta em Data/, label no TCC, slug)
CIRCUITS = [
    ("Bahrain", "Bahrein", "bahrein"),
    ("Saudi Arabia", "Arabia Saudita", "arabia"),
    ("Hungary", "Hungria", "hungria"),
    ("Italy", "Italia", "italia"),
    ("United States", "Estados Unidos", "eua"),
]


def load_circuit(folder: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(f"Data/{folder}/Results/*_results_*.csv"):
        year = int(re.search(r"_(\d{4})\.csv$", f).group(1))
        d = pd.read_csv(f, usecols=["Abbreviation", "ClassifiedPosition"])
        d["Year"] = year
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def sort_key(pos: str) -> float:
    # ordena por posicao numerica; R/D/W vao para o fim
    try:
        return float(pos)
    except (TypeError, ValueError):
        return 99.0


def table(folder, label, slug) -> str:
    df = load_circuit(folder)
    years = sorted(df["Year"].unique())
    pivot = df.pivot_table(
        index="Abbreviation", columns="Year", values="ClassifiedPosition",
        aggfunc="first",
    ).reindex(columns=years)
    # ordena pelo desempenho medio (melhores primeiro)
    mean_pos = pivot.map(sort_key).mean(axis=1)
    pivot = pivot.loc[mean_pos.sort_values().index]

    col_spec = "l" + "c" * len(years)
    L = []
    L.append(r"\begin{table}[ttt]")
    L.append(r"    \centering")
    L.append(
        f"    \\caption{{Resultado final (posicao de chegada) de cada piloto "
        f"por ano -- {label}.}}"
    )
    L.append(f"    \\label{{tab:resultados_{slug}}}")
    L.append(r"    \scriptsize")
    L.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    L.append(r"        \hline")
    L.append(r"        \rowcolor[HTML]{D9D9D9}")
    head = r"        \textbf{Piloto}"
    for y in years:
        head += f" & \\multicolumn{{1}}{{c}}{{\\textbf{{{y}}}}}"
    head += r" \\ \hline"
    L.append(head)
    for drv, r in pivot.iterrows():
        cells = []
        for y in years:
            v = r[y]
            cells.append("--" if pd.isna(v) else str(v))
        L.append(f"        \\texttt{{{drv}}} & " + " & ".join(cells) + r" \\ \hline")
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    blocks = [table(folder, label, slug) for folder, label, slug in CIRCUITS]
    out = "\n\n".join(blocks) + "\n"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        fh.write(out)
    print(out)
    print(f"% Salvo em: {OUT_PATH}")


if __name__ == "__main__":
    main()
