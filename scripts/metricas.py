"""
Funcoes compartilhadas pelos analisadores de CPU e GPU.

Metricas (comentarios do professor):

  FLOPs por celula atualizada = 4   (3 somas + 1 multiplicacao)
  FLOPs totais                = 4 * (N-2)^2 * ITERACOES
  MFLOPS                      = FLOPs totais / 1e6 / tempo_s
  MFLOPS/W                    = MFLOPS / potencia_media_W
                              = (FLOPs totais / 1e6) / energia_J

MFLOPS e MFLOPS/W sao TAXAS: nao dependem do numero de iteracoes, entao
permitem comparar tamanhos e dispositivos (CPU x GPU) mesmo quando o numero
de iteracoes usado em cada experimento e diferente.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLOPS_POR_CELULA = 4
NOMES_TAMANHO = ["Pequena", "Média", "Grande"]


def flops_totais(n, iteracoes):
    return FLOPS_POR_CELULA * (n - 2) ** 2 * iteracoes


def num(serie):
    """Converte para numero; 'NA' e vazios viram NaN."""
    return pd.to_numeric(serie, errors="coerce")


def rotulos_tamanho(ns):
    """{n: 'Pequena (N=512)'} quando ha 3 tamanhos; senao 'N=...'."""
    ns = sorted(set(int(n) for n in ns))
    if len(ns) == 3:
        return {n: f"{nome}\n(N={n})" for nome, n in zip(NOMES_TAMANHO, ns)}
    return {n: f"N={n}" for n in ns}


def _taxas(df, col_tempo, prefixo):
    flops = flops_totais(df["n"], df["iteracoes"])
    df[f"mflops_{prefixo}"] = flops / 1e6 / df[col_tempo]


def _mflops_por_watt(df, col_mflops):
    pot = df["potencia_media_w"].where(df["potencia_media_w"] > 0)
    df["mflops_por_watt"] = df[col_mflops] / pot


def carrega_cpu(caminho):
    df = pd.read_csv(caminho)
    if "precisao" not in df.columns:
        df["precisao"] = "double"          # CSVs antigos: tudo era double
    df["schedule"] = df["schedule"].astype(str)
    _taxas(df, "tempo_s", "e2e")
    df["mflops"] = df["mflops_e2e"]
    return df


def carrega_gpu(caminho):
    df = pd.read_csv(caminho)
    if "precisao" not in df.columns:
        df["precisao"] = "double"
    for c in ["potencia_media_w", "energia_j"]:
        df[c] = num(df[c]) if c in df.columns else np.nan
    _taxas(df, "tempo_kernel_s", "kernel")
    _taxas(df, "tempo_e2e_s", "e2e")
    _mflops_por_watt(df, "mflops_e2e")
    return df


def barras_agrupadas(ax, categorias, series, ylabel=None, titulo=None,
                     log=False, valores=True, formato=None):
    """
    categorias: lista de rotulos do eixo x
    series: dict {nome_da_serie: [valor_por_categoria]}
    """
    nser = max(len(series), 1)
    largura = 0.8 / nser
    x = np.arange(len(categorias))
    for k, (nome, vals) in enumerate(series.items()):
        vals = [np.nan if v is None else v for v in vals]
        pos = x - 0.4 + largura * (k + 0.5)
        barras = ax.bar(pos, vals, largura, label=nome)
        if valores:
            for b, v in zip(barras, vals):
                if np.isfinite(v):
                    fmt = formato or ("{:.0f}" if abs(v) >= 100 else "{:.2f}")
                    ax.annotate(fmt.format(v),
                                (b.get_x() + b.get_width() / 2, b.get_height()),
                                ha="center", va="bottom", fontsize=7,
                                xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(categorias)
    if ylabel:
        ax.set_ylabel(ylabel)
    if titulo:
        ax.set_title(titulo)
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    if nser > 1:
        ax.legend(fontsize=8)


def salva(fig, pasta, nome):
    caminho = os.path.join(pasta, nome)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def tem_energia(serie):
    return bool(serie.notna().any())
