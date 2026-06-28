"""Analise exploratoria das voltas de VALIDACAO e de HOLDOUT do LSTM_hybrid.

Para cada pista, reconstroi exatamente o split do hibrido (mesma chave temporal
composta Year*10000+LapNumber e mesmos ratios da config) e gera, com as features
AGRUPADAS por tema (dinamica de volta e climaticas) em paineis (subplots):
  - histogramas SEPARADOS (um arquivo da validacao, outro do holdout) por grupo
  - CDF empirica (validacao vs holdout) por grupo
  - boxplot (validacao vs holdout) por grupo

Alem disso gera um grafico de barras com os COMPOSTOS de pneu (pirelliCompound)
usados em cada particao (proporcao validacao vs holdout), por pista.

Tambem salva uma tabela de estatisticas descritivas por pista/feature/particao.

As voltas batem com o que o modelo realmente usou (reusa as funcoes oficiais de
split). A particao de treino fica de fora (so validacao e holdout, como pedido).

Uso:
    python Scripts/Source/eda_val_holdout_hybrid.py
    python Scripts/Source/eda_val_holdout_hybrid.py --show
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from modeling_utils import (
    build_sequential_split,
    load_cleaned_data,
    prepare_raw_features,
    select_modeling_columns,
)

SCRIPT_PATH = Path(__file__)
REPO_ROOT = SCRIPT_PATH.resolve().parents[2]

CIRCUITS = [
    ("configs/bahrain.yaml", "Bahrein", "bahrein"),
    ("configs/saudi.yaml", "Arabia Saudita", "arabia"),
    ("configs/hungary.yaml", "Hungria", "hungria"),
    ("configs/italy.yaml", "Italia", "italia"),
    ("configs/usa.yaml", "Estados Unidos", "eua"),
]

VAL_COLOR = "#1f77b4"
HOLD_COLOR = "#d62728"

FEATURE_LABELS = {
    "LapTime_seconds": "Tempo de volta (s)",
    "LapTime_prev": "Tempo da volta anterior (s)",
    "TyreLife": "Vida do pneu (voltas)",
    "LapNumber": "Numero da volta",
    "Humidity_RBF_Median": "Umidade (%)",
    "Pressure_RBF_Median": "Pressao (hPa)",
    "TrackTemp_RBF_Median": "Temp. da pista (C)",
    "WindSpeed_RBF_Median": "Velocidade do vento (m/s)",
    "TempDelta_RBF_Median": "Delta de temperatura (C)",
    "Year": "Ano",
}

# Grupos tematicos -> (slug do arquivo, titulo). As features de cada grupo sao
# resolvidas contra as colunas disponiveis na pista.
GROUPS = [
    ("dinamica", "Dinamica da volta",
     ["LapTime_seconds", "LapTime_prev", "TyreLife", "LapNumber"]),
    ("climaticas", "Variaveis climaticas",
     ["TrackTemp_RBF_Median", "Humidity_RBF_Median", "Pressure_RBF_Median",
      "WindSpeed_RBF_Median", "TempDelta_RBF_Median"]),
]


def label_of(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def col(df, feature):
    return pd.to_numeric(df[feature], errors="coerce").dropna()


def grid_shape(k):
    ncols = min(3, k)
    nrows = math.ceil(k / ncols)
    return nrows, ncols


def split_val_holdout(config_rel: str):
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
    val_idx = model_step.index[model_step.isin(val_steps)]

    df_val = df_base.loc[val_idx].copy()
    df_hold = df_base.loc[holdout_idx].copy()
    return config, df_val, df_hold


def describe(pista, feature, label, df):
    x = col(df, feature).to_numpy()
    return {
        "pista": pista, "feature": feature, "particao": label, "n": len(x),
        "media": np.mean(x), "desvio": np.std(x, ddof=1),
        "min": np.min(x), "Q1": np.percentile(x, 25), "mediana": np.median(x),
        "Q3": np.percentile(x, 75), "max": np.max(x),
        "assimetria": stats.skew(x), "curtose": stats.kurtosis(x),
    }


def _finish(fig, plt, path, show):
    fig.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        print(f"Salvo: {path}")
    plt.close(fig)


def _grid_axes(plt, k, suptitle):
    nrows, ncols = grid_shape(k)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[k:]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=12)
    return fig, axes


def plot_hist_group(plt, features, df_val, df_hold, bins_map, partition, suptitle, path, show):
    """Histogramas de um grupo (uma particao por arquivo)."""
    df = df_val if partition == "validacao" else df_hold
    color = VAL_COLOR if partition == "validacao" else HOLD_COLOR
    fig, axes = _grid_axes(plt, len(features), suptitle)
    for ax, f in zip(axes, features):
        ax.hist(col(df, f), bins=bins_map[f], color=color, alpha=0.78, density=True)
        ax.set_xlabel(label_of(f))
        ax.set_ylabel("Densidade")
        ax.grid(True, alpha=0.3)
    _finish(fig, plt, path, show)


def plot_cdf_group(plt, features, df_val, df_hold, suptitle, path, show):
    fig, axes = _grid_axes(plt, len(features), suptitle)
    for ax, f in zip(axes, features):
        for df, c, name in [(df_val, VAL_COLOR, "Validacao"), (df_hold, HOLD_COLOR, "Holdout")]:
            x = np.sort(col(df, f).to_numpy())
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, color=c, label=name)
        ax.set_xlabel(label_of(f))
        ax.set_ylabel("F(x)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    _finish(fig, plt, path, show)


def plot_box_group(plt, features, df_val, df_hold, suptitle, path, show):
    fig, axes = _grid_axes(plt, len(features), suptitle)
    for ax, f in zip(axes, features):
        bp = ax.boxplot([col(df_val, f), col(df_hold, f)],
                        tick_labels=["Val.", "Hold."],
                        patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], [VAL_COLOR, HOLD_COLOR]):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_ylabel(label_of(f))
        ax.grid(True, axis="y", alpha=0.3)
    _finish(fig, plt, path, show)


def plot_compounds(plt, df_val, df_hold, title, path, show):
    c = "pirelliCompound"
    cats = sorted(set(df_val[c].dropna()).union(df_hold[c].dropna()))
    pv = df_val[c].value_counts(normalize=True).reindex(cats, fill_value=0)
    ph = df_hold[c].value_counts(normalize=True).reindex(cats, fill_value=0)
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, pv.values * 100, w, color=VAL_COLOR, alpha=0.8,
           label=f"Validacao (n={len(df_val)})")
    ax.bar(x + w / 2, ph.values * 100, w, color=HOLD_COLOR, alpha=0.8,
           label=f"Holdout (n={len(df_hold)})")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_xlabel("Composto de pneu (Pirelli)")
    ax.set_ylabel("Proporcao das voltas (%)")
    ax.set_title(title)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    _finish(fig, plt, path, show)


def shared_bins(df_val, df_hold, features):
    bins = {}
    for f in features:
        allv = pd.concat([col(df_val, f), col(df_hold, f)])
        lo, hi = float(allv.min()), float(allv.max())
        nb = min(30, max(8, int(np.sqrt(len(allv)))))
        bins[f] = np.linspace(lo, hi, nb) if hi > lo else 10
    return bins


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="Exibir em vez de salvar.")
    parser.add_argument("--outdir", default="Scripts/Results/eda_val_holdout")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = REPO_ROOT / args.outdir
    if not args.show:
        out_dir.mkdir(parents=True, exist_ok=True)
    stats_rows = []

    for config_rel, label, slug in CIRCUITS:
        config, df_val, df_hold = split_val_holdout(config_rel)
        print(f"\n=== {label} ===")
        print(f"  Validacao: anos {sorted(df_val.Year.unique())} | "
              f"voltas {int(df_val.LapNumber.min())}-{int(df_val.LapNumber.max())} | n={len(df_val)}")
        print(f"  Holdout:   anos {sorted(df_hold.Year.unique())} | "
              f"voltas {int(df_hold.LapNumber.min())}-{int(df_hold.LapNumber.max())} | n={len(df_hold)}")

        for gslug, gtitle, gfeats in GROUPS:
            feats = [f for f in gfeats if f in df_val.columns]
            if not feats:
                continue
            bins_map = shared_bins(df_val, df_hold, feats)
            base = f"{label} — {gtitle}"
            plot_hist_group(plt, feats, df_val, df_hold, bins_map, "validacao",
                            f"{base} (validacao)", out_dir / f"hist_{gslug}_{slug}_validacao.pdf", args.show)
            plot_hist_group(plt, feats, df_val, df_hold, bins_map, "holdout",
                            f"{base} (holdout)", out_dir / f"hist_{gslug}_{slug}_holdout.pdf", args.show)
            plot_cdf_group(plt, feats, df_val, df_hold, f"{base} — CDF",
                           out_dir / f"cdf_{gslug}_{slug}.pdf", args.show)
            plot_box_group(plt, feats, df_val, df_hold, f"{base} — boxplot",
                           out_dir / f"box_{gslug}_{slug}.pdf", args.show)
            for f in feats:
                for pl, df in [("validacao", df_val), ("holdout", df_hold)]:
                    stats_rows.append(describe(label, f, pl, df))

        if "pirelliCompound" in df_val.columns:
            plot_compounds(plt, df_val, df_hold,
                           f"{label} — compostos usados (validacao vs holdout)",
                           out_dir / f"compostos_{slug}.pdf", args.show)

    stats_df = pd.DataFrame(stats_rows)
    print("\n=== Estatisticas descritivas (resumo) ===")
    print(stats_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not args.show:
        csv_path = out_dir / "stats_val_holdout.csv"
        stats_df.to_csv(csv_path, index=False)
        print(f"\nSalvo: {csv_path}")


if __name__ == "__main__":
    main()
