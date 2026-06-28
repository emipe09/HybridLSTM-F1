"""Gera tabelas LaTeX de caracterizacao do tempo de volta.

Saidas:
  1. Tabela com a media do numero de voltas por piloto e por circuito,
     para cada ano (agregando todos os circuitos).
  2. Para cada circuito: media do tempo de volta por piloto por ano e
     media do tempo de volta por equipe por ano.
"""
import glob
import os

import pandas as pd

MODEL_DATA_GLOB = "Scripts/ModelData/*/*_cleaned_data.csv"
OUT_PATH = "Scripts/Results/tables_laptime_characterization.tex"

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
        d = pd.read_csv(
            f, usecols=["Driver", "Team", "LapNumber", "LapTime_seconds", "Year"]
        )
        d["Circuit"] = os.path.basename(os.path.dirname(f))
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def fmt(x, dec=2):
    if pd.isna(x):
        return "--"
    return f"{x:.{dec}f}"


# --------------------------------------------------------------------------
# Tabela 1: media de voltas por piloto e total de voltas no circuito, por ano
# (uma tabela por circuito)
# --------------------------------------------------------------------------
def table_laps(df: pd.DataFrame, circuit: str) -> str:
    short, slug = CIRCUIT_SHORT[circuit]
    sub_c = df[df["Circuit"] == circuit]
    years = sorted(sub_c["Year"].unique())
    rows = []
    for y in years:
        sub = sub_c[sub_c["Year"] == y]
        n_drv = sub["Driver"].nunique()
        # media do numero de voltas por piloto = voltas que cada piloto
        # completou no circuito/ano, media entre pilotos
        laps_per_driver = sub.groupby("Driver").size().mean()
        # numero de voltas do circuito no ano = distancia da prova
        # (maior LapNumber registrado)
        circuit_laps = int(sub["LapNumber"].max())
        rows.append((y, n_drv, laps_per_driver, circuit_laps))

    L = []
    L.append(r"\begin{table}[ttt]")
    L.append(r"    \centering")
    L.append(
        f"    \\caption{{Media do numero de voltas por piloto e numero de "
        f"voltas do circuito, por ano -- {short}.}}"
    )
    L.append(f"    \\label{{tab:voltas_{slug}}}")
    L.append(r"    \scriptsize")
    L.append(r"    \begin{tabular}{lrrr}")
    L.append(r"        \hline")
    L.append(r"        \rowcolor[HTML]{D9D9D9}")
    L.append(
        r"        \textbf{Ano} & "
        r"\multicolumn{1}{c}{\textbf{N\textsuperscript{o} pilotos}} & "
        r"\multicolumn{1}{c}{\textbf{Voltas/piloto}} & "
        r"\multicolumn{1}{c}{\textbf{Voltas do circuito}} \\ \hline"
    )
    for y, n_drv, lpd, circuit_laps in rows:
        L.append(
            f"        \\texttt{{{y}}} & {n_drv} & {fmt(lpd)} & "
            f"{circuit_laps} \\\\ \\hline"
        )
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Tabela 2: media do tempo de volta por <grupo> por ano, para cada circuito
# --------------------------------------------------------------------------
def table_laptime_by_group(df: pd.DataFrame, circuit: str, group: str) -> str:
    short, slug = CIRCUIT_SHORT[circuit]
    sub = df[df["Circuit"] == circuit]
    years = sorted(sub["Year"].unique())
    pivot = (
        sub.groupby([group, "Year"])["LapTime_seconds"]
        .mean()
        .unstack("Year")
        .reindex(columns=years)
    )
    # ordena pelo tempo medio geral (mais rapido primeiro)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    grp_label = "piloto" if group == "Driver" else "equipe"
    grp_head = "Piloto" if group == "Driver" else "Equipe"
    col_spec = "l" + "r" * len(years)

    L = []
    L.append(r"\begin{table}[ttt]")
    L.append(r"    \centering")
    L.append(
        f"    \\caption{{Media do tempo de volta (s) por {grp_label} e por ano "
        f"-- {short}.}}"
    )
    L.append(f"    \\label{{tab:tempo_{grp_label}_{slug}}}")
    L.append(r"    \scriptsize")
    L.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    L.append(r"        \hline")
    L.append(r"        \rowcolor[HTML]{D9D9D9}")
    head = f"        \\textbf{{{grp_head}}}"
    for y in years:
        head += f" & \\multicolumn{{1}}{{c}}{{\\textbf{{{y}}}}}"
    head += r" \\ \hline"
    L.append(head)
    for name, r in pivot.iterrows():
        safe = str(name).replace("_", r"\_").replace("&", r"\&")
        cells = " & ".join(fmt(r[y]) for y in years)
        L.append(f"        \\texttt{{{safe}}} & {cells} \\\\ \\hline")
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    df = load()
    blocks = []
    for circuit in CIRCUIT_SHORT:
        blocks.append(table_laps(df, circuit))
        blocks.append(table_laptime_by_group(df, circuit, "Driver"))
        blocks.append(table_laptime_by_group(df, circuit, "Team"))
    out = "\n\n".join(blocks) + "\n"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        fh.write(out)
    print(out)
    print(f"\n% Salvo em: {OUT_PATH}")


if __name__ == "__main__":
    main()
