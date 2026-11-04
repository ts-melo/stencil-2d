#!/usr/bin/env bash
# Bateria de experimentos de CPU (sequencial, Pthreads, OpenMP) do stencil 2D.
#
# Varia: TAMANHO da matriz (pequena/media/grande), PRECISAO (double/float),
# numero de threads e escalonamento OpenMP. Grava
# results/resultados_cpu_<timestamp>.csv
#
# Variaveis de ambiente opcionais:
#   PRECISOES="double float"   precisoes a testar
#   REPETICOES=10              repeticoes por configuracao
#   TAMANHOS="rot,N,ITER ..."  substitui a lista de tamanhos/iteracoes
#   THREADS_LISTA="1 2 4 8 12" numeros de threads (o ultimo e usado na
#                              comparacao de schedules do OpenMP)
#   SCHEDULES="static dynamic,64 guided"  escalonamentos do OpenMP
#
# Exemplo (comparacao CPU x GPU com as MESMAS iteracoes da GPU, so as
# configuracoes principais):
#   TAMANHOS="pequena,512,50000 media,2000,5000 grande,8000,400" \
#   THREADS_LISTA="4 8 12" SCHEDULES="static guided" bash scripts/run_experiments.sh
#
# Energia NAO e medida aqui: so a GPU tem medicao de energia (ver
# run_gpu_experiments.sh). Na CPU medimos tempo, speedup e MFLOPS.
set -euo pipefail
trap 'echo "ERRO: script parou na linha $LINENO (comando: $BASH_COMMAND)" >&2' ERR
cd "$(dirname "$0")/.."

# ---------------- parametros ----------------
REPETICOES="${REPETICOES:-10}"
PRECISOES="${PRECISOES:-double float}"
# ajuste THREADS_LISTA conforme os nucleos da maquina
read -ra THREADS_LISTA <<< "${THREADS_LISTA:-1 2 4 8 12}"
read -ra SCHEDULES <<< "${SCHEDULES:-static dynamic,64 guided}"

# "rotulo,N,ITERACOES". As iteracoes diferem por tamanho so para manter cada
# execucao em alguns segundos; MFLOPS e MFLOPS/W sao TAXAS e, portanto,
# comparaveis entre tamanhos mesmo com numero de iteracoes diferente.
if [ -n "${TAMANHOS:-}" ]; then
    read -ra TAMANHOS_ARR <<< "$TAMANHOS"
else
    TAMANHOS_ARR=("pequena,512,4000" "media,2000,1000" "grande,8000,100")
fi

SAIDA="results/resultados_cpu_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p results
echo "versao,threads,schedule,chunk,n,iteracoes,tempo_s,checksum,precisao" > "$SAIDA"

# Executa um comando de CPU e grava a linha no CSV.
# Linha do programa: versao,threads,schedule,chunk,n,iter,tempo_s,checksum,precisao
rodar() {
    "$@" >> "$SAIDA"
}

MAX_THREADS=${THREADS_LISTA[${#THREADS_LISTA[@]}-1]}

for PREC in $PRECISOES; do
    echo "=============================================="
    echo ">> PRECISAO: $PREC"
    echo ">> Compilando..."
    make -s PRECISAO="$PREC"
    BIN="bin/$PREC"

    for tam in "${TAMANHOS_ARR[@]}"; do
        IFS=',' read -r ROTULO N ITER <<< "$tam"
        echo "----------------------------------------------"
        echo ">> [$PREC] matriz $ROTULO: N=$N, ITER=$ITER, $REPETICOES repeticoes"

        echo "   [1/4] sequencial"
        for r in $(seq 1 "$REPETICOES"); do
            rodar "$BIN/stencil_seq" "$N" "$ITER"
        done

        echo "   [2/4] Pthreads — threads: ${THREADS_LISTA[*]}"
        for t in "${THREADS_LISTA[@]}"; do
            for r in $(seq 1 "$REPETICOES"); do
                rodar "$BIN/stencil_pthreads" "$N" "$ITER" "$t"
            done
        done

        echo "   [3/4] OpenMP (schedule=static) — threads: ${THREADS_LISTA[*]}"
        export OMP_SCHEDULE="static"
        for t in "${THREADS_LISTA[@]}"; do
            export OMP_NUM_THREADS=$t
            for r in $(seq 1 "$REPETICOES"); do
                rodar "$BIN/stencil_openmp" "$N" "$ITER"
            done
        done

        echo "   [4/4] OpenMP — escalonamentos com $MAX_THREADS threads"
        export OMP_NUM_THREADS=$MAX_THREADS
        for s in "${SCHEDULES[@]}"; do
            export OMP_SCHEDULE="$s"
            for r in $(seq 1 "$REPETICOES"); do
                rodar "$BIN/stencil_openmp" "$N" "$ITER"
            done
        done
    done
done

echo ">> Concluido. Resultados salvos em: $SAIDA"
echo ">> Para analisar: python3 scripts/analyze_results.py $SAIDA"
