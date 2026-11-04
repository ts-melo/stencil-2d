#!/usr/bin/env bash
# Bateria de experimentos da versao CUDA (GPU): varia TAMANHO da matriz
# (pequena/media/grande), PRECISAO (double/float) e tamanho do bloco de
# threads; mede tempo e, quando o driver expoe, potencia/energia via nvidia-smi.
#
# Variaveis de ambiente opcionais:
#   PRECISOES="double float"   precisoes a testar
#   REPETICOES=10              repeticoes por configuracao
#   TAMANHOS="rot,N,ITER ..."  substitui a lista de tamanhos (teste rapido)
#   NVCC_CCBIN=/usr/bin/g++-12 compilador host aceito pelo nvcc
#
# Requisitos: GPU NVIDIA + driver (nvidia-smi) + CUDA Toolkit (nvcc).
# Se nvidia-smi nao reportar power.draw, o script continua e mede so o tempo.
set -euo pipefail
# Se qualquer comando falhar, mostra ONDE (evita sair em silencio).
trap 'echo "ERRO: script parou na linha $LINENO (comando: $BASH_COMMAND, codigo $?)" >&2' ERR
cd "$(dirname "$0")/.."

# Compilador host aceito pelo nvcc (gcc <= 14).
if [ -z "${NVCC_CCBIN:-}" ] && [ -x /usr/bin/g++-12 ]; then export NVCC_CCBIN=/usr/bin/g++-12; fi

# ---------------- parametros ----------------
REPETICOES="${REPETICOES:-10}"
PRECISOES="${PRECISOES:-double float}"
BLOCOS=("8 8" "16 16" "32 32" "32 8" "8 32")   # "BLOCO_X BLOCO_Y"

# "rotulo,N,ITERACOES". Iteracoes ajustadas para que cada execucao dure
# alguns segundos (a amostragem de potencia precisa de varias amostras).
# MFLOPS e MFLOPS/W sao taxas: comparaveis entre tamanhos e com a CPU.
if [ -n "${TAMANHOS:-}" ]; then
    read -ra TAMANHOS_ARR <<< "$TAMANHOS"
else
    TAMANHOS_ARR=("pequena,512,50000" "media,2000,5000" "grande,8000,400")
fi

SAIDA="results/resultados_gpu_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p results
echo "versao,bloco_x,bloco_y,threads_por_bloco,grid_x,grid_y,n,iteracoes,tempo_kernel_s,tempo_e2e_s,checksum,precisao,potencia_media_w,energia_j,amostras_potencia" > "$SAIDA"

# --- checa se nvidia-smi consegue reportar potencia neste hardware ---
MEDIR_POTENCIA=0
SMI_LOOP="-l 1"
if command -v nvidia-smi >/dev/null 2>&1; then
    # ATENCAO: no nvidia-smi, "-c" significa --compute-mode (ALTERA a GPU), nao "contagem".
    AMOSTRA=$(timeout 10 nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
    if [[ "$AMOSTRA" =~ ^[0-9.]+$ ]]; then
        MEDIR_POTENCIA=1
        TESTE_LMS=$(timeout 2 nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 100 2>/dev/null | head -1 || true)
        if [[ -n "$TESTE_LMS" ]]; then
            SMI_LOOP="-lms 100"
        fi
        echo ">> Medicao de potencia via nvidia-smi: OK (amostra teste: ${AMOSTRA} W, modo: ${SMI_LOOP})"
    else
        echo "AVISO: nvidia-smi nao reporta power.draw nesta GPU — energia sera 'NA' (apenas tempo é medido)."
    fi
else
    echo "AVISO: nvidia-smi nao encontrado — energia sera 'NA' (apenas tempo é medido)."
fi

# --- roda um comando medindo a potencia da GPU durante a execucao ---
medir() {
    local log="$1"; shift
    if [ "$MEDIR_POTENCIA" -eq 1 ]; then
        # consulta potencia E utilizacao: so amostras com a GPU ativa entram na media
        nvidia-smi --query-gpu=power.draw,utilization.gpu --format=csv,noheader,nounits $SMI_LOOP > "$log" 2>/dev/null &
        local pid_smi=$!
        "$@"
        sleep 0.3
        kill "$pid_smi" 2>/dev/null || true
        wait "$pid_smi" 2>/dev/null || true
    else
        "$@"
    fi
}

for PREC in $PRECISOES; do
    echo "=============================================="
    echo ">> PRECISAO: $PREC"
    echo ">> Compilando versao CUDA ($PREC)..."
    make cuda PRECISAO="$PREC"
    make -s PRECISAO="$PREC" "bin/$PREC/stencil_seq"
    BIN="bin/$PREC"

    # --- validacao de corretude (checksum CUDA x sequencial) ---
    if [ "$PREC" = "float" ]; then TOL="1e-5"; else TOL="1e-9"; fi
    echo ">> Validando corretude ($PREC, tolerancia relativa $TOL)..."
    # CSV seq: checksum = campo 8; CSV cuda: checksum = campo 11
    CHECK_SEQ=$("$BIN/stencil_seq" 300 300 2>/dev/null | cut -d',' -f8)
    CHECK_CUDA=$("$BIN/stencil_cuda" 300 300 16 16 2>/dev/null | cut -d',' -f11)
    python3 - "$CHECK_SEQ" "$CHECK_CUDA" "$TOL" << 'PYEOF'
import sys
seq, cuda, tol = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
dif = abs(seq - cuda) / max(abs(seq), 1e-12)
print(f"   checksum seq={seq:.6f} cuda={cuda:.6f} diferenca_relativa={dif:.2e}")
if dif > tol:
    print("   ATENCAO: diferenca maior que o esperado — revise o kernel antes de confiar nos tempos.")
    sys.exit(1)
print("   OK: checksums equivalentes dentro da tolerancia numerica esperada.")
PYEOF

    echo ">> Aquecimento (warm-up) da GPU..."
    "$BIN/stencil_cuda" 1000 200 16 16 >/dev/null 2>&1

    for tam in "${TAMANHOS_ARR[@]}"; do
        IFS=',' read -r ROTULO N ITER <<< "$tam"
        echo "----------------------------------------------"
        echo ">> [$PREC] matriz $ROTULO: N=$N, ITER=$ITER (blocos: ${BLOCOS[*]}; $REPETICOES repeticoes cada)"
        for par in "${BLOCOS[@]}"; do
            bx=$(echo "$par" | cut -d' ' -f1)
            by=$(echo "$par" | cut -d' ' -f2)
            for r in $(seq 1 "$REPETICOES"); do
                LOG=$(mktemp)
                LINHA=$(medir "$LOG" "$BIN/stencil_cuda" "$N" "$ITER" "$bx" "$by")
                # linha: versao,bx,by,threads,gx,gy,n,iter,tempo_kernel,tempo_e2e,checksum,precisao
                tempo_e2e=$(echo "$LINHA" | cut -d',' -f10)

                if [ "$MEDIR_POTENCIA" -eq 1 ]; then
                    read -r pot_media n_amostras < <(awk -F',' '
                        { gsub(/[ \r]/,"",$1); gsub(/[ \r]/,"",$2);
                          if ($1 ~ /^[0-9.]+$/) {
                              sa+=$1; na++;
                              if ($2 ~ /^[0-9]+$/ && ($2+0) > 0) { s+=$1; n++ }
                          } }
                        END {
                            if (n>0)       printf "%f %d\n", s/n, n;
                            else if (na>0) printf "%f %d\n", sa/na, na;
                            else           printf "0 0\n"
                        }' "$LOG")
                else
                    pot_media="0"; n_amostras="0"
                fi

                if [ "$n_amostras" -gt 0 ]; then
                    energia_j=$(python3 -c "import sys; print(f'{float(sys.argv[1])*float(sys.argv[2]):.4f}')" "$pot_media" "$tempo_e2e")
                else
                    pot_media="NA"; energia_j="NA"
                fi

                rm -f "$LOG"
                echo "${LINHA},${pot_media},${energia_j},${n_amostras}" >> "$SAIDA"
            done
            echo "   [$PREC/$ROTULO] bloco ${bx}x${by}: concluido"
        done
    done
done

echo ">> Concluido. Resultados salvos em: $SAIDA"
echo ">> Para analisar: python3 scripts/analyze_gpu_results.py $SAIDA [results/resultados_cpu_XXXX.csv]"
