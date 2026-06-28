"""Experimento de motivacao da RQ1: Regressao Linear em dois regimes de dados.

RQ1: How can a Formula 1 driver's lap time be predicted in scenarios with
limited data?

Objetivo: evidenciar que o desempenho de uma Regressao Linear simples PIORA
quando filtramos por piloto (pouco dado), em relacao a treinar com todas as
voltas do circuito juntas (muito dado). Isso motiva o uso do LSTM Hybrid.

Duas abordagens, mesmas features e mesmo pre-processamento, split temporal 80/20
(o bloco de holdout final = ultimos 20% das voltas, sem vazamento temporal):

  Abordagem 1 (por piloto):   para cada (circuito, piloto) treina-se uma LR.
                              O desempenho reportado e a MEDIA das LRs dos pilotos.
  Abordagem 2 (por circuito): para cada circuito treina-se uma unica LR com todos
                              os pilotos juntos.

Saidas:
  - Console: resumo por circuito e geral.
  - CSV:     Scripts/Results/rq1_lr_comparison.csv (linhas por circuito + media).
  - CSV:     Scripts/Results/rq1_lr_per_driver.csv (metricas de cada piloto).
  - LaTeX:   Scripts/Results/tables_rq1_lr_comparison.tex
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modeling_utils import (
    CONFIG_ALIASES,
    build_cleaned_data_path,
    build_sequential_split,
    calc_stats,
    fit_predict_linear_regression,
    load_simple_yaml,
    prepare_raw_features,
    select_modeling_columns,
)

# (slug do circuito -> (nome curto PT, slug curto)) para a tabela final.
CIRCUIT_SHORT = {
    "Bahrain Grand Prix": ("Bahrein", "bahrein"),
    "Saudi Arabian Grand Prix": ("Arabia Saudita", "arabia"),
    "Hungarian Grand Prix": ("Hungria", "hungria"),
    "Italian Grand Prix": ("Italia", "italia"),
    "United States Grand Prix": ("Estados Unidos", "eua"),
}

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "Scripts" / "Results"
CSV_SUMMARY = OUT_DIR / "rq1_lr_comparison.csv"
CSV_PER_DRIVER = OUT_DIR / "rq1_lr_per_driver.csv"
TEX_PATH = OUT_DIR / "tables_rq1_lr_comparison.tex"


def metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_single_lr(df, num_cols, cat_cols, target_col, lap_col, holdout_ratio):
    """Treina uma LR no bloco de treino (primeiros 80% temporais) e reporta o
    desempenho no conjunto de TESTE (holdout dos ultimos 20% temporais). Retorna
    None se nao for possivel formar o split (poucas voltas)."""
    X_raw, y_raw, valid_idx = prepare_raw_features(df, num_cols, cat_cols, target_col)
    if len(valid_idx) < 5:
        return None
    try:
        (step_series, _mn, _mx, model_idx, holdout_idx, *_rest) = build_sequential_split(
            df, valid_idx, holdout_ratio, lap_col
        )
    except (ValueError, KeyError):
        return None

    X_model, y_model = X_raw.loc[model_idx], y_raw.loc[model_idx]
    X_hold, y_hold = X_raw.loc[holdout_idx], y_raw.loc[holdout_idx]
    if len(X_model) == 0 or len(X_hold) == 0:
        return None

    preds, *_ = fit_predict_linear_regression(X_model, y_model, X_hold, cat_cols)
    out = metrics(y_hold, preds)
    out["n_train"] = int(len(X_model))
    out["n_test"] = int(len(X_hold))
    out["n_total"] = int(len(valid_idx))
    return out


def load_configs() -> list[dict]:
    configs = []
    for yaml_name in CONFIG_ALIASES.values():
        cfg = load_simple_yaml(REPO_ROOT / "configs" / yaml_name)
        configs.append(cfg)
    return configs


def main():
    configs = load_configs()
    summary_rows = []
    per_driver_rows = []

    for config in configs:
        gp = str(config["target_gp_name"])
        short = CIRCUIT_SHORT.get(gp, (gp, gp))[0]
        target_col = str(config["target_col"])
        lap_col = str(config["lap_col"])
        holdout_ratio = float(config["holdout_ratio"])

        csv_path = build_cleaned_data_path(REPO_ROOT, config)
        df = pd.read_csv(csv_path)
        num_cols, cat_cols = select_modeling_columns(df, config)

        # ---- Abordagem 2: por circuito (todos os pilotos juntos) ----
        circ_m = run_single_lr(df, num_cols, cat_cols, target_col, lap_col, holdout_ratio)

        # ---- Abordagem 1: por piloto (uma LR por piloto, depois media) ----
        drivers = sorted(df["Driver"].dropna().astype(str).str.upper().unique())
        cat_cols_drv = [c for c in cat_cols if c != "Driver"]  # constante apos o filtro
        drv_metrics = []
        n_skipped = 0
        for drv in drivers:
            sub = df[df["Driver"].astype(str).str.upper() == drv].reset_index(drop=True)
            m = run_single_lr(sub, num_cols, cat_cols_drv, target_col, lap_col, holdout_ratio)
            if m is None:
                n_skipped += 1
                continue
            drv_metrics.append(m)
            per_driver_rows.append({"circuit": short, "driver": drv, **m})

        n_used = len(drv_metrics)
        dm = pd.DataFrame(drv_metrics)

        def agg(col):
            # media, desvio-padrao e IC 95% (t-Student) da media entre pilotos.
            # Descarta NaN (ex.: R2 indefinido quando o piloto tem 1 volta no teste).
            vals = dm[col].dropna().to_numpy()
            mean_v, lo, hi = calc_stats(vals)
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            return mean_v, sd, lo, hi

        a1_mae, a1_mae_sd, a1_mae_lo, a1_mae_hi = agg("mae")
        a1_rmse, a1_rmse_sd, a1_rmse_lo, a1_rmse_hi = agg("rmse")
        a1_r2, a1_r2_sd, a1_r2_lo, a1_r2_hi = agg("r2")

        summary_rows.append(
            {
                "circuit": short,
                "gp": gp,
                # Abordagem 2 (por circuito)
                "a2_mae": circ_m["mae"],
                "a2_rmse": circ_m["rmse"],
                "a2_r2": circ_m["r2"],
                "a2_n_total": circ_m["n_total"],
                # Abordagem 1 (por piloto, media +- IC95%)
                "a1_mae": a1_mae,
                "a1_mae_sd": a1_mae_sd,
                "a1_mae_lo": a1_mae_lo,
                "a1_mae_hi": a1_mae_hi,
                "a1_rmse": a1_rmse,
                "a1_rmse_sd": a1_rmse_sd,
                "a1_rmse_lo": a1_rmse_lo,
                "a1_rmse_hi": a1_rmse_hi,
                "a1_r2": a1_r2,
                "a1_r2_sd": a1_r2_sd,
                "a1_r2_lo": a1_r2_lo,
                "a1_r2_hi": a1_r2_hi,
                "a1_n_drivers": n_used,
                "a1_n_skipped": n_skipped,
                "a1_mean_inst": float(dm["n_total"].mean()),
            }
        )

        print(f"\n=== {short} ({gp}) ===")
        print(
            f"  Abordagem 2 (circuito, n={circ_m['n_total']}): "
            f"MAE={circ_m['mae']:.3f} | RMSE={circ_m['rmse']:.3f} | R2={circ_m['r2']:.3f}"
        )
        print(
            f"  Abordagem 1 (media de {n_used} pilotos, {n_skipped} ignorados, "
            f"inst.media={dm['n_total'].mean():.0f}):"
        )
        print(f"      MAE  = {a1_mae:.3f}  IC95% [{a1_mae_lo:.3f}, {a1_mae_hi:.3f}]")
        print(f"      RMSE = {a1_rmse:.3f}  IC95% [{a1_rmse_lo:.3f}, {a1_rmse_hi:.3f}]")
        print(f"      R2   = {a1_r2:.3f}  IC95% [{a1_r2_lo:.3f}, {a1_r2_hi:.3f}]")

    summary = pd.DataFrame(summary_rows)

    # ---- Linha "Media" (media simples entre circuitos) ----
    overall = {
        "circuit": "Media",
        "gp": "",
        "a2_mae": summary["a2_mae"].mean(),
        "a2_rmse": summary["a2_rmse"].mean(),
        "a2_r2": summary["a2_r2"].mean(),
        "a2_n_total": summary["a2_n_total"].mean(),
        "a1_mae": summary["a1_mae"].mean(),
        "a1_mae_sd": np.nan,
        "a1_rmse": summary["a1_rmse"].mean(),
        "a1_rmse_sd": np.nan,
        "a1_r2": summary["a1_r2"].mean(),
        "a1_r2_sd": np.nan,
        "a1_n_drivers": summary["a1_n_drivers"].mean(),
        "a1_n_skipped": summary["a1_n_skipped"].sum(),
        "a1_mean_inst": summary["a1_mean_inst"].mean(),
    }
    summary_out = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_out.to_csv(CSV_SUMMARY, index=False)
    pd.DataFrame(per_driver_rows).to_csv(CSV_PER_DRIVER, index=False)

    write_latex(summary_out)

    print("\n--- RESUMO GERAL (media entre circuitos) ---")
    print(
        f"  Abordagem 2 (por circuito): MAE={overall['a2_mae']:.3f} | "
        f"RMSE={overall['a2_rmse']:.3f} | R2={overall['a2_r2']:.3f}"
    )
    print(
        f"  Abordagem 1 (por piloto):   MAE={overall['a1_mae']:.3f} | "
        f"RMSE={overall['a1_rmse']:.3f} | R2={overall['a1_r2']:.3f}"
    )
    print(f"\nSalvo: {CSV_SUMMARY}\nSalvo: {CSV_PER_DRIVER}\nSalvo: {TEX_PATH}")


def fmt(x, dec=3):
    if pd.isna(x):
        return "--"
    return f"{x:.{dec}f}"


def write_latex(summary: pd.DataFrame) -> None:
    L = []
    L.append(r"\begin{table}[ttt]")
    L.append(r"    \centering")
    L.append(
        r"    \caption{Desempenho no conjunto de teste da Regressao Linear (divisao "
        r"temporal 80/20, avaliacao no holdout dos ultimos 20\%) em dois regimes de "
        r"dados. Abordagem 1: media das LR por piloto; Abordagem 2: uma LR por circuito "
        r"(todos os pilotos juntos). O erro cresce ao filtrar por piloto, evidenciando "
        r"o regime de dados limitados da RQ1.}"
    )
    L.append(r"    \label{tab:rq1_lr_comparacao}")
    L.append(r"    \scriptsize")
    L.append(r"    \begin{tabular}{lrrrrrr}")
    L.append(r"        \hline")
    L.append(r"        \rowcolor[HTML]{D9D9D9}")
    L.append(
        r"         & \multicolumn{3}{c}{\textbf{Abord. 1 (por piloto)}} & "
        r"\multicolumn{3}{c}{\textbf{Abord. 2 (por circuito)}} \\"
    )
    L.append(r"        \rowcolor[HTML]{D9D9D9}")
    L.append(
        r"        \textbf{Circuito} & \textbf{MAE} & \textbf{RMSE} & \textbf{R\textsuperscript{2}} "
        r"& \textbf{MAE} & \textbf{RMSE} & \textbf{R\textsuperscript{2}} \\ \hline"
    )
    for _, r in summary.iterrows():
        bold = r["circuit"] == "Media"
        name = f"\\textbf{{{r['circuit']}}}" if bold else r["circuit"]

        def cell(v, dec=3):
            s = fmt(v, dec)
            return f"\\textbf{{{s}}}" if bold else s

        L.append(
            f"        {name} & {cell(r['a1_mae'])} & {cell(r['a1_rmse'])} & {cell(r['a1_r2'])} & "
            f"{cell(r['a2_mae'])} & {cell(r['a2_rmse'])} & {cell(r['a2_r2'])} \\\\ \\hline"
        )
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    TEX_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
