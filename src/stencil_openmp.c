/*
 * stencil_openmp.c
 *
 * Versão paralela usando OpenMP.
 *
 * Uma única região paralela permanece aberta durante
 * todas as iterações.
 *
 * Cada thread mantém ponteiros locais para os dois buffers.
 *
 * A barreira implícita ao final de "omp for" garante
 * que todas as threads terminaram de escrever o próximo
 * estado antes da troca dos ponteiros.
 *
 * Número de threads:
 * OMP_NUM_THREADS
 *
 * Escalonamento:
 * OMP_SCHEDULE
 *
 * Exemplo:
 *
 * OMP_NUM_THREADS=8 \
 * OMP_SCHEDULE=static \
 * ./stencil_openmp 2000 1000
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "common.h"


int main(int argc, char **argv) {

    if (argc < 3) {

        fprintf(stderr,
                "Uso: %s <N> <ITERACOES>\n",
                argv[0]);

        fprintf(stderr,
                "Defina OMP_NUM_THREADS "
                "e OMP_SCHEDULE\n");

        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    int iteracoes = atoi(argv[2]);

    if (n < 3 ||
        iteracoes < 1) {

        fprintf(stderr,
                "N deve ser >= 3 e "
                "ITERACOES >= 1\n");

        return EXIT_FAILURE;
    }

    /*
     * Impede o runtime OpenMP de reduzir dinamicamente
     * o número de threads solicitado.
     */
    omp_set_dynamic(0);

    real_t *A =
        aloca_matriz(n);

    real_t *B =
        aloca_matriz(n);

    inicializa_matriz(A, n);
    inicializa_matriz(B, n);

    int nthreads_usadas = 1;

    double t0 =
        agora_segundos();

    #pragma omp parallel default(shared)
    {

        /*
         * Ponteiros privados de cada thread.
         */
        real_t *cur = A;
        real_t *prox = B;

        /*
         * Descobre o tamanho real da equipe OpenMP.
         */
        #pragma omp single nowait
        {
            nthreads_usadas =
                omp_get_num_threads();
        }

        for (int it = 0;
             it < iteracoes;
             it++) {

            /*
             * O schedule é definido em OMP_SCHEDULE.
             */
            #pragma omp for schedule(runtime)

            for (int i = 1;
                 i < n - 1;
                 i++) {

                for (int j = 1;
                     j < n - 1;
                     j++) {

                    prox[IDX(i, j, n)] =
                        CONST_QUARTO * (

                            cur[IDX(i - 1, j, n)] +
                            cur[IDX(i + 1, j, n)] +
                            cur[IDX(i, j - 1, n)] +
                            cur[IDX(i, j + 1, n)]

                        );
                }
            }

            /*
             * Existe uma barreira implícita no fim
             * do omp for.
             * Portanto todas as escritas em prox
             * já terminaram.
             * A troca é somente dos ponteiros privados
             * de cada thread.
             */
            real_t *tmp = cur;
            cur = prox;
            prox = tmp;
        }
    }

    double t1 =
        agora_segundos();

    double tempo =
        t1 - t0;

    /*
     * A e B compartilhados nunca foram trocados.
     * Portanto o resultado depende da paridade.
     */
    real_t *final =
        (iteracoes % 2 == 0)
        ? A
        : B;

    double soma =
        checksum(final, n);

    /*
     * Descobre qual schedule realmente foi usado.
     */
    omp_sched_t tipo;
    int chunk;

    omp_get_schedule(
        &tipo,
        &chunk
    );

    /*
     * Alguns runtimes adicionam o modificador
     * omp_sched_monotonic ao tipo.
     */
    omp_sched_t tipo_base =
        (omp_sched_t)(
            tipo &
            ~omp_sched_monotonic
        );

    const char *nome_sched =

        (tipo_base ==
         omp_sched_static)
        ? "static"

        : (tipo_base ==
           omp_sched_dynamic)
        ? "dynamic"

        : (tipo_base ==
           omp_sched_guided)
        ? "guided"

        : (tipo_base ==
           omp_sched_auto)
        ? "auto"

        : "unknown";

    fprintf(
        stderr,

        "[OPENMP] %s "
        "N=%d iter=%d "
        "threads=%d "
        "schedule=%s,%d "
        "tempo=%.6f s "
        "checksum=%.6f\n",

        NOME_PRECISAO,
        n,
        iteracoes,
        nthreads_usadas,
        nome_sched,
        chunk,
        tempo,
        soma
    );

    /*
     * CSV:
     * versao,threads,schedule,chunk,
     * n,iteracoes,tempo_s,checksum,precisao
     */
    printf(
        "openmp,%d,%s,%d,%d,%d,%.6f,%.6f,%s\n",

        nthreads_usadas,
        nome_sched,
        chunk,
        n,
        iteracoes,
        tempo,
        soma,
        NOME_PRECISAO
    );

    free(A);
    free(B);

    return EXIT_SUCCESS;
}