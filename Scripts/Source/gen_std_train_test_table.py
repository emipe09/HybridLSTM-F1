"""Gera tabelas LaTeX com o desvio padrao dos residuos no treino (validacao)
e no teste (holdout sequencial), por circuito, uma tabela para cada modelo
escolhido: LR-EW, XGB-EW e LSTM_hybrid.

Fonte: runs do MLflow (Scripts/Results/mlruns), run mais recente por modelo.
  treino -> ew_residual_std_mean (LR/XGB) ou val_residual_std (LSTM_hybrid)
  teste  -> holdout_residual_std
"""
import os

import mlflow
import pandas as pd

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

OUT_PATH = "Scripts/Results/tables_std_train_test.tex"
TRACKING_URI = "file:Scripts/Results/mlruns"

# circuitos na ordem desejada (label, slug do runName)
CIRCUITS = [
    ("Bahrein", "bahrain_grand_prix"),
    ("Arabia Saudita", "saudi_arabian_grand_prix"),
    ("Hungria", "hungarian_grand_prix"),
    ("Italia", "italian_grand_prix"),
    ("Estados Unidos", "united_states_grand_prix"),
]

# (titulo, sufixo do runName, metrica de treino)
MODELS = [
    ("Regressao Linear (janela expansiva)", "linear_regression-ew", "ew_residual_std_mean"),
    ("XGBoost (janela expansiva)", "xgboost-ew", "ew_residual_std_mean"),
    ("LSTM\\_hybrid", "lstm_hybrid-single_split", "val_residual_std"),
]


def load_runs() -> pd.DataFrame:
    mlflow.set_tracking_uri(TRACKING_URI)
    exps = [e for e in mlflow.search_experiments() if e.name.startswith("f1-lap-time-")]
    runs = mlflow.search_runs(experiment_ids=[e.experiment_id for e in exps])
    # mantem o run mais recente por nome (descarta reexecucoes antigas)
    runs = runs.sort_values("start_time").groupby(
        "tags.mlflow.runName", as_index=False
    ).tail(1)
    return runs.set_index("tags.mlflow.runName")


def fmt(x):
    return "--" if pd.isna(x) else f"{x:.4f}"


def table(runs, title, suffix, train_metric, idx):
    L = []
    L.append(r"\begin{table}[ttt]")
    L.append(r"    \centering")
    L.append(
        f"    \\caption{{Desvio padrao dos residuos (s) no treino (validacao) e "
        f"no teste (holdout sequencial) por circuito -- {title}.}}"
    )
    L.append(f"    \\label{{tab:std_treino_teste_{idx}}}")
    L.append(r"    \scriptsize")
    L.append(r"    \begin{tabular}{lrr}")
    L.append(r"        \hline")
    L.append(r"        \rowcolor[HTML]{D9D9D9}")
    L.append(
        r"        \textbf{Circuito} & "
        r"\multicolumn{1}{c}{ {\small$\boldsymbol{s_{\text{treino}}}$} } & "
        r"\multicolumn{1}{c}{ {\small$\boldsymbol{s_{\text{teste}}}$} } \\ \hline"
    )
    for label, slug in CIRCUITS:
        row = runs.loc[f"{slug}-{suffix}"]
        s_tr = row.get("metrics." + train_metric)
        s_te = row.get("metrics.holdout_residual_std")
        L.append(f"        \\texttt{{{label}}} & {fmt(s_tr)} & {fmt(s_te)} \\\\ \\hline")
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    runs = load_runs()
    blocks = [
        table(runs, title, suffix, tm, i)
        for i, (title, suffix, tm) in enumerate(MODELS)
    ]
    out = "\n\n".join(blocks) + "\n"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        fh.write(out)
    print(out)
    print(f"% Salvo em: {OUT_PATH}")


if __name__ == "__main__":
    main()
