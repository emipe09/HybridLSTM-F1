"""Gera tabelas LaTeX do numero de instancias (voltas) por abordagem da RQ1.

Motivacao: evidenciar que o volume de dados cai drasticamente quando se filtra
por piloto (Abordagem 1) em relacao a usar todas as voltas do circuito juntas
(Abordagem 2). Uma instancia = uma volta = uma linha do dataset limpo.

Saidas:
  1. Tabela-resumo comparativa (uma linha por circuito): instancias por circuito
     (Abordagem 2) vs. estatisticas das instancias por piloto (Abordagem 1).
  2. Tabela detalhada por circuito: instancias por piloto (Abordagem 1), com a
     linha de total do circuito (Abordagem 2).
"""
import glob
import os

import pandas as pd

MODEL_DATA_GLOB = "Scripts/ModelData/*/*_cleaned_data.csv"
OUT_PATH = "Scripts/Results/tables_instancias_rq1.tex"

# Nomes de circuito mais curtos para legendas/labels.
CIRCUIT_SHORT = {
    "Bahrain Grand Prix": ("Bahrein", "bahrein"),
    "Saudi Arabian Grand Prix": ("Arabia Saudita", "arabia"),
    "Hungarian Grand Prix": ("Hungria", "hungria"),
    "Italian Grand Prix": ("Italia", "italia"),
    "United States Grand Prix": ("Estados Unidos", "eua"),
}


def load() -> pd.DataFrame:
    dfs = []
    for f in glob.glob(MODEL_DATA_GLOB):
        d = pd.read_csv(f, usecols=["Driver", "Year", "LapNumber"])
        d["Circuit"] = os.path.basename(os.path.dirname(f))
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def fmt(x, dec=1):
    if pd.isna(x):
        return "--"
    return f"{x:.{dec}f}"


# --------------------------------------------------------------------------
# Tabela 1 (resumo): comparacao direta das duas abordagens, por circuito.
# --------------------------------------------------------------------------
def table_summary(df: pd.DataFrame) -> str:
    L = []
    L.append(r"\begin{table}[ttt]")
    L.append(r"    \centering")
    L.append(
        r"    \caption{Numero de instancias (voltas) por abordagem da RQ1. "
        r"Abordagem 2 usa todas as voltas do circuito; a Abordagem 1 filtra por "
        r"piloto, reduzindo o volume em uma ordem de grandeza.}"
    )
    L.append(r"    \label{tab:instancias_rq1_resumo}")
    L.append(r"    \scriptsize")
    L.append(r"    \begin{tabular}{lrrrrr}")
    L.append(r"        \hline")
    L.append(r"        \rowcolor[HTML]{D9D9D9}")
    L.append(
        r"         & \textbf{N\textsuperscript{o}} & "
        r"\textbf{Abord. 2} & \multicolumn{3}{c}{\textbf{Abordagem 1 (por piloto)}} \\"
    )
    L.append(r"        \rowcolor[HTML]{D9D9D9}")
    L.append(
        r"        \textbf{Circuito} & \textbf{pilotos} & "
        r"\textbf{Total} & \textbf{Media} & \textbf{Min} & \textbf{Max} \\ \hline"
    )
    for circuit, (short, _slug) in CIRCUIT_SHORT.items():
        sub = df[df["Circuit"] == circuit]
        per_driver = sub.groupby("Driver").size()
        total = int(len(sub))
        n_drv = int(sub["Driver"].nunique())
        L.append(
            f"        {short} & {n_drv} & {total} & "
            f"{fmt(per_driver.mean())} & {int(per_driver.min())} & "
            f"{int(per_driver.max())} \\\\ \\hline"
        )
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Tabela 2 (detalhe por circuito): instancias por piloto + total do circuito.
# --------------------------------------------------------------------------
def table_per_circuit(df: pd.DataFrame, circuit: str) -> str:
    short, slug = CIRCUIT_SHORT[circuit]
    sub = df[df["Circuit"] == circuit]
    per_driver = sub.groupby("Driver").size().sort_values(ascending=False)
    total = int(len(sub))

    L = []
    L.append(r"\begin{table}[ttt]")
    L.append(r"    \centering")
    L.append(
        f"    \\caption{{Numero de instancias (voltas) por piloto -- {short}. "
        f"A Abordagem 1 treina um modelo por piloto; a Abordagem 2 usa o total "
        f"do circuito ({total} instancias).}}"
    )
    L.append(f"    \\label{{tab:instancias_{slug}}}")
    L.append(r"    \scriptsize")
    L.append(r"    \begin{tabular}{lr}")
    L.append(r"        \hline")
    L.append(r"        \rowcolor[HTML]{D9D9D9}")
    L.append(
        r"        \textbf{Piloto} & "
        r"\multicolumn{1}{c}{\textbf{Instancias (Abord. 1)}} \\ \hline"
    )
    for name, n in per_driver.items():
        safe = str(name).replace("_", r"\_").replace("&", r"\&")
        L.append(f"        \\texttt{{{safe}}} & {int(n)} \\\\ \\hline")
    L.append(
        f"        \\textbf{{Total (Abord. 2)}} & \\textbf{{{total}}} \\\\ \\hline"
    )
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    df = load()
    blocks = [table_summary(df)]
    for circuit in CIRCUIT_SHORT:
        blocks.append(table_per_circuit(df, circuit))
    out = "\n\n".join(blocks) + "\n"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        fh.write(out)
    print(out)
    print(f"\n% Salvo em: {OUT_PATH}")


if __name__ == "__main__":
    main()
