#!/usr/bin/env python3
"""
Analisa os experimentos da versao CUDA (GPU).

Para cada precisao (double/float), tamanho de matriz e bloco de threads:
  - tempo end-to-end e tempo so de kernel (media e desvio padrao)
  - MFLOPS (4 FLOPs por celula atualizada)
  - potencia, energia e MFLOPS/W, quando o nvidia-smi reportou power.draw

Energia so e medida na GPU. Se um CSV de CPU for informado, compara
sequencial x melhor CPU x melhor GPU por tamanho, em TEMPO e MFLOPS, mas
SOMENTE onde CPU e GPU usaram o mesmo N e o mesmo numero de iteracoes
(caso contrario avisa e pula).

Gera graficos em results/graficos_gpu/.

Uso:
    python3 scripts/analyze_gpu_results.py results/resultados_gpu_XXXX.csv [results/resultados_cpu_XXXX.csv]
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metricas import (carrega_cpu, carrega_gpu, rotulos_tamanho,
                      barras_agrupadas, salva, tem_energia)


def melhor_linha(df, col):
    if df.empty or not df[col].notna().any():
        return None
    return df.loc[df[col].idxmax()]


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 analyze_gpu_results.py <csv_gpu> [csv_cpu]")
        sys.exit(1)

    caminho = sys.argv[1]
    df = carrega_gpu(caminho)
    saida = os.path.join(os.path.dirname(caminho) or ".", "graficos_gpu")
    os.makedirs(saida, exist_ok=True)

    resumo = (
        df.groupby(["precisao", "n", "iteracoes", "bloco_x", "bloco_y",
                    "threads_por_bloco", "grid_x", "grid_y"])
        .agg(tempo_e2e_medio_s=("tempo_e2e_s", "mean"),
             desvio_e2e_s=("tempo_e2e_s", "std"),
             tempo_kernel_medio_s=("tempo_kernel_s", "mean"),
             mflops_kernel=("mflops_kernel", "mean"),
             mflops_e2e=("mflops_e2e", "mean"),
             potencia_media_w=("potencia_media_w", "mean"),
             energia_media_j=("energia_j", "mean"),
             mflops_por_watt=("mflops_por_watt", "mean"),
             n_execucoes=("tempo_e2e_s", "count"))
        .reset_index()
    )
    resumo["bloco"] = (resumo["bloco_x"].astype(int).astype(str) + "x"
                       + resumo["bloco_y"].astype(int).astype(str))
    resumo.to_csv(os.path.join(saida, "resumo_gpu.csv"), index=False)

    cols = ["bloco", "grid_x", "grid_y", "tempo_e2e_medio_s", "desvio_e2e_s",
            "tempo_kernel_medio_s", "mflops_kernel", "mflops_e2e",
            "potencia_media_w", "energia_media_j", "mflops_por_watt"]
    for (prec, n), g in resumo.groupby(["precisao", "n"]):
        print(f"\n=== GPU | {prec} | N={n} | iteracoes={int(g['iteracoes'].iloc[0])} ===")
        print(g.sort_values("tempo_e2e_medio_s")[cols].round(4).to_string(index=False))
    print(f"\nResumo completo salvo em: {os.path.join(saida, 'resumo_gpu.csv')}")

    ns = sorted(resumo["n"].unique())
    rot = rotulos_tamanho(ns)
    cats_tam = [rot[n] for n in ns]
    precs = sorted(resumo["precisao"].unique())
    houve_energia = tem_energia(resumo["mflops_por_watt"])
    blocos = list(dict.fromkeys(resumo.sort_values(["bloco_x", "bloco_y"])["bloco"]))

    # ---------------- graficos por precisao ----------------
    for prec in precs:
        rp = resumo[resumo["precisao"] == prec]

        def por_tamanho(nome_arq, titulo_fig, ylabel, series_cols, formato=None, log=False):
            fig, axs = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 4.2), squeeze=False)
            for ax, n in zip(axs[0], ns):
                g = rp[rp["n"] == n].set_index("bloco").reindex(blocos)
                series = {nome: g[col].tolist() for nome, col in series_cols.items()}
                barras_agrupadas(ax, blocos, series, titulo=rot[n].replace("\n", " "),
                                 formato=formato, log=log)
                ax.set_xlabel("Bloco de threads")
            axs[0][0].set_ylabel(ylabel)
            fig.suptitle(f"{titulo_fig} — CUDA ({prec})")
            salva(fig, saida, nome_arq)

        por_tamanho(f"tempo_por_bloco_{prec}.png", "Tempo por configuração de bloco",
                    "Tempo médio (s)",
                    {"end-to-end": "tempo_e2e_medio_s", "só kernel": "tempo_kernel_medio_s"},
                    formato="{:.2f}")
        por_tamanho(f"mflops_por_bloco_{prec}.png", "Desempenho por configuração de bloco",
                    "MFLOPS",
                    {"só kernel": "mflops_kernel", "end-to-end": "mflops_e2e"})

        if tem_energia(rp["mflops_por_watt"]):
            por_tamanho(f"energia_por_bloco_{prec}.png", "Energia por configuração de bloco",
                        "Energia média (J)", {"energia": "energia_media_j"}, formato="{:.0f}")
            por_tamanho(f"mflops_por_watt_por_bloco_{prec}.png",
                        "Eficiência energética por configuração de bloco",
                        "MFLOPS/W", {"MFLOPS/W": "mflops_por_watt"}, formato="{:.0f}")

        # desempenho x tamanho (uma curva por bloco)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for b in blocos:
            g = rp[rp["bloco"] == b].set_index("n").reindex(ns)
            ax.plot(range(len(ns)), g["mflops_kernel"], marker="o", label=f"bloco {b}")
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels(cats_tam)
        ax.set_ylabel("MFLOPS (só kernel)")
        ax.set_title(f"Desempenho vs. tamanho da matriz — CUDA ({prec})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        salva(fig, saida, f"mflops_vs_tamanho_{prec}.png")

    # ---------------- double x float na GPU ----------------
    if len(precs) > 1:
        ser, ser_w = {}, {}
        for prec in precs:
            rp = resumo[resumo["precisao"] == prec]
            ser[f"{prec}"] = []
            ser_w[f"{prec}"] = []
            for n in ns:
                m = melhor_linha(rp[rp["n"] == n], "mflops_kernel")
                ser[prec].append(np.nan if m is None else m["mflops_kernel"])
                m = melhor_linha(rp[rp["n"] == n], "mflops_por_watt")
                ser_w[prec].append(np.nan if m is None else m["mflops_por_watt"])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        barras_agrupadas(ax, cats_tam, ser, ylabel="MFLOPS (só kernel, melhor bloco)",
                         titulo="Impacto da precisão (double x float) — CUDA")
        salva(fig, saida, "comparacao_precisao_gpu.png")
        if houve_energia:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            barras_agrupadas(ax, cats_tam, ser_w, ylabel="MFLOPS/W (melhor bloco)",
                             titulo="Eficiência energética: double x float — CUDA")
            salva(fig, saida, "comparacao_precisao_gpu_mflops_por_watt.png")

    print(f"Gráficos salvos em: {saida}/")
    if not houve_energia:
        print("Obs.: o CSV não tem potência da GPU (nvidia-smi sem power.draw): sem gráficos de energia/MFLOPS/W.")

    # ---------------- comparacao CPU x GPU ----------------
    # So compara quando CPU e GPU usaram o MESMO N e o MESMO numero de
    # iteracoes; nesse caso os speedups sao razoes de TEMPO (mesmo trabalho).
    if len(sys.argv) > 2:
        import pandas as pd
        dc = carrega_cpu(sys.argv[2])
        rc = (dc.groupby(["precisao", "n", "iteracoes", "versao", "threads", "schedule"])
              .agg(tempo=("tempo_s", "mean"), mflops=("mflops", "mean"))
              .reset_index())
        print("\n=== Comparação CPU x GPU (mesmo N e mesmas iterações) ===")
        linhas = []
        for prec in precs:
            ns_ok = []
            for n in ns:
                g = resumo[(resumo["precisao"] == prec) & (resumo["n"] == n)]
                itg = int(g["iteracoes"].iloc[0])
                c = rc[(rc["precisao"] == prec) & (rc["n"] == n) & (rc["iteracoes"] == itg)]
                if c.empty or c[c["versao"] == "seq"].empty:
                    its = sorted(set(rc[(rc["precisao"] == prec) & (rc["n"] == n)]["iteracoes"]))
                    print(f"AVISO: {prec} N={n}: GPU usou {itg} iterações e a CPU {its or 'nenhuma'}; "
                          f"sem comparação direta (rode a CPU com ITER={itg}).")
                    continue
                ns_ok.append(n)
                seq = c[c["versao"] == "seq"]
                bc = melhor_linha(c[c["versao"] != "seq"], "mflops")
                bg = g.loc[g["tempo_e2e_medio_s"].idxmin()]
                bk = g.loc[g["tempo_kernel_medio_s"].idxmin()]
                t_seq = seq["tempo"].mean()
                t_cpu = np.nan if bc is None else bc["tempo"]
                linhas.append({
                    "precisao": prec, "n": n, "iteracoes": itg,
                    "t_seq_s": t_seq, "t_cpu_s": t_cpu,
                    "cpu_config": "-" if bc is None else f"{bc['versao']}/{int(bc['threads'])}t/{bc['schedule']}",
                    "t_gpu_kernel_s": bk["tempo_kernel_medio_s"], "bloco_kernel": bk["bloco"],
                    "t_gpu_e2e_s": bg["tempo_e2e_medio_s"], "bloco_e2e": bg["bloco"],
                    "speedup_gpu_kernel_vs_cpu": t_cpu / bk["tempo_kernel_medio_s"],
                    "speedup_gpu_e2e_vs_cpu": t_cpu / bg["tempo_e2e_medio_s"],
                    "speedup_gpu_e2e_vs_seq": t_seq / bg["tempo_e2e_medio_s"],
                    "mflops_seq": seq["mflops"].mean(),
                    "mflops_cpu": np.nan if bc is None else bc["mflops"],
                    "mflops_gpu_kernel": bk["mflops_kernel"],
                    "mflops_gpu_e2e": bg["mflops_e2e"],
                    "gpu_mflops_por_watt": bg["mflops_por_watt"],
                })
            if not ns_ok:
                continue
            L = pd.DataFrame([l for l in linhas if l["precisao"] == prec])
            cats = [rot[n] for n in L["n"]]
            fig, ax = plt.subplots(figsize=(9, 4.8))
            barras_agrupadas(ax, cats,
                             {"Sequencial": L["t_seq_s"].tolist(),
                              "Melhor CPU paralela": L["t_cpu_s"].tolist(),
                              "GPU (end-to-end)": L["t_gpu_e2e_s"].tolist(),
                              "GPU (só kernel)": L["t_gpu_kernel_s"].tolist()},
                             ylabel="Tempo (s, escala log)", log=True, formato="{:.2f}",
                             titulo=f"CPU x GPU — tempo com as mesmas iterações ({prec})")
            salva(fig, saida, f"comparacao_cpu_gpu_tempo_{prec}.png")
            fig, ax = plt.subplots(figsize=(9, 4.8))
            barras_agrupadas(ax, cats,
                             {"Sequencial": L["mflops_seq"].tolist(),
                              "Melhor CPU paralela": L["mflops_cpu"].tolist(),
                              "GPU (end-to-end)": L["mflops_gpu_e2e"].tolist(),
                              "GPU (só kernel)": L["mflops_gpu_kernel"].tolist()},
                             ylabel="MFLOPS (escala log)", log=True,
                             titulo=f"CPU x GPU — vazão ({prec})")
            salva(fig, saida, f"comparacao_cpu_gpu_mflops_{prec}.png")

        if linhas:
            tab = pd.DataFrame(linhas)
            print(tab.round(3).to_string(index=False))
            tab.to_csv(os.path.join(saida, "comparacao_cpu_gpu.csv"), index=False)
            print(f"\nTabela salva em: {os.path.join(saida, 'comparacao_cpu_gpu.csv')}")
            print("Escopos: tempo da CPU = laço principal (clock_gettime); GPU 'kernel' = laço de kernels "
                  "(cudaEvent); GPU 'end-to-end' = inclui alocação e transferências PCIe. "
                  "Energia/MFLOPS/W: só GPU (potência da placa via nvidia-smi).")
        else:
            print("Nenhuma configuração comparável (mesmo N e mesmas iterações). "
                  "Veja os avisos acima.")

if __name__ == "__main__":
    main()
