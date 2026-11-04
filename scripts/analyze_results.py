#!/usr/bin/env python3
"""
Analisa os experimentos de CPU (sequencial, Pthreads, OpenMP) do stencil 2D.

Para cada precisao (double/float) e tamanho de matriz calcula:
  - tempo medio e desvio padrao
  - speedup = t_seq / t_paralelo   e   eficiencia = speedup / threads
  - MFLOPS  (4 FLOPs por celula: 3 somas + 1 multiplicacao)
  (energia so e medida na GPU; ver analyze_gpu_results.py)

Gera graficos em results/graficos/ (nao so tabelas).

Uso:
    python3 scripts/analyze_results.py results/resultados_cpu_XXXX.csv
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metricas import carrega_cpu, rotulos_tamanho, barras_agrupadas, salva


def melhor(resumo, versao, col):
    """Linha de maior `col` entre as configuracoes da versao (ou None)."""
    sub = resumo[resumo["versao"] == versao]
    if sub.empty or not sub[col].notna().any():
        return None
    return sub.loc[sub[col].idxmax()]


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 analyze_results.py <csv_cpu>")
        sys.exit(1)

    caminho = sys.argv[1]
    df = carrega_cpu(caminho)
    saida = os.path.join(os.path.dirname(caminho) or ".", "graficos")
    os.makedirs(saida, exist_ok=True)

    resumo = (
        df.groupby(["precisao", "n", "iteracoes", "versao", "threads", "schedule"])
        .agg(tempo_medio_s=("tempo_s", "mean"),
             desvio_padrao_s=("tempo_s", "std"),
             mflops_medio=("mflops", "mean"),
             n_execucoes=("tempo_s", "count"))
        .reset_index()
    )
    base = (resumo[resumo["versao"] == "seq"]
            .set_index(["precisao", "n"])["tempo_medio_s"].rename("t_seq"))
    resumo = resumo.join(base, on=["precisao", "n"])
    resumo["speedup"] = resumo["t_seq"] / resumo["tempo_medio_s"]
    resumo["eficiencia"] = resumo["speedup"] / resumo["threads"]

    resumo.to_csv(os.path.join(saida, "resumo.csv"), index=False)

    cols = ["versao", "threads", "schedule", "tempo_medio_s", "desvio_padrao_s",
            "mflops_medio", "speedup", "eficiencia"]
    for (prec, n), g in resumo.groupby(["precisao", "n"]):
        print(f"\n=== {prec} | N={n} | iteracoes={int(g['iteracoes'].iloc[0])} ===")
        print(g[cols].round(4).to_string(index=False))
    print(f"\nResumo completo salvo em: {os.path.join(saida, 'resumo.csv')}")

    ns = sorted(resumo["n"].unique())
    rot = rotulos_tamanho(ns)
    cats = [rot[n] for n in ns]

    for prec in sorted(resumo["precisao"].unique()):
        rp = resumo[resumo["precisao"] == prec]

        # ---- speedup e eficiencia vs threads (uma curva por tamanho) ----
        for versao in ["pthreads", "openmp"]:
            sub = rp[(rp["versao"] == versao) & (rp["schedule"].isin(["-", "static"]))]
            if sub.empty:
                continue
            for metrica, titulo, nome in [("speedup", "Speedup", "speedup"),
                                          ("eficiencia", "Eficiência", "eficiencia")]:
                fig, ax = plt.subplots()
                for n in ns:
                    s = sub[sub["n"] == n].sort_values("threads")
                    ax.plot(s["threads"], s[metrica], marker="o", label=rot[n].replace("\n", " "))
                tmax = sub["threads"].max()
                if metrica == "speedup":
                    ax.plot([1, tmax], [1, tmax], "--", color="gray", label="Ideal (linear)")
                else:
                    ax.axhline(1.0, linestyle="--", color="gray", label="Ideal")
                ax.set_xlabel("Número de threads")
                ax.set_ylabel(titulo)
                ax.set_title(f"{titulo} vs. threads — {versao} ({prec})")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                salva(fig, saida, f"{nome}_{versao}_{prec}.png")

        # ---- MFLOPS vs threads (pthreads | openmp static) ----
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, versao in zip(axs, ["pthreads", "openmp"]):
            sub = rp[(rp["versao"] == versao) & (rp["schedule"].isin(["-", "static"]))]
            for n in ns:
                s = sub[sub["n"] == n].sort_values("threads")
                ax.plot(s["threads"], s["mflops_medio"], marker="o", label=rot[n].replace("\n", " "))
            ax.set_title(f"{versao} ({prec})")
            ax.set_xlabel("Número de threads")
            ax.grid(True, alpha=0.3)
        axs[0].set_ylabel("MFLOPS")
        axs[0].legend(fontsize=8)
        salva(fig, saida, f"mflops_vs_threads_{prec}.png")

        # ---- desempenho por tamanho: seq x melhor pthreads x melhor openmp ----
        series = {}
        for rotulo, versao in [("Sequencial", "seq"), ("Pthreads (melhor)", "pthreads"),
                               ("OpenMP (melhor)", "openmp")]:
            vals = []
            for n in ns:
                m = melhor(rp[rp["n"] == n], versao, "mflops_medio")
                vals.append(np.nan if m is None else m["mflops_medio"])
            series[rotulo] = vals
        fig, ax = plt.subplots(figsize=(8, 4.5))
        barras_agrupadas(ax, cats, series, ylabel="MFLOPS",
                         titulo=f"Desempenho por tamanho de matriz — CPU ({prec})")
        salva(fig, saida, f"desempenho_por_tamanho_{prec}.png")

        # ---- escalonamentos OpenMP (maior numero de threads) ----
        so = rp[rp["versao"] == "openmp"]
        if not so.empty:
            tmax = so["threads"].max()
            so = so[so["threads"] == tmax]
            scheds = sorted(so["schedule"].unique())
            ser = {}
            for s in scheds:
                ser[s] = [
                    (so[(so["n"] == n) & (so["schedule"] == s)]["mflops_medio"].mean())
                    for n in ns
                ]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            barras_agrupadas(ax, cats, ser, ylabel="MFLOPS",
                             titulo=f"Escalonamentos OpenMP, {tmax} threads ({prec})")
            salva(fig, saida, f"comparacao_schedules_{prec}.png")

    # ---- double x float (melhor configuracao paralela e sequencial) ----
    precs = sorted(resumo["precisao"].unique())
    if len(precs) > 1:
        ser = {}
        for prec in precs:
            rp = resumo[resumo["precisao"] == prec]
            ser[f"Sequencial ({prec})"] = [
                (lambda m: np.nan if m is None else m["mflops_medio"])(
                    melhor(rp[rp["n"] == n], "seq", "mflops_medio")) for n in ns]
            ser[f"Melhor paralelo ({prec})"] = [
                rp[(rp["n"] == n) & (rp["versao"] != "seq")]["mflops_medio"].max() for n in ns]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        barras_agrupadas(ax, cats, ser, ylabel="MFLOPS",
                         titulo="Impacto da precisão (double x float) — CPU")
        salva(fig, saida, "comparacao_precisao_cpu.png")

    print(f"Gráficos salvos em: {saida}/")


if __name__ == "__main__":
    main()
