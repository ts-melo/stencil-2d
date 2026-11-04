/*
 * stencil_seq.c
 *
 * Versão sequencial (baseline) do stencil 2D.
 *
 * Cada célula interna é atualizada pela média dos
 * quatro vizinhos imediatos.
 *
 * As condições de contorno são fixas.
 * Como os dois buffers são inicializados com as bordas
 * corretas e o stencil nunca escreve nas bordas,
 * não é necessário reaplicá-las a cada iteração.
 *
 * Uso:
 * ./stencil_seq <N> <ITERACOES>
 */

#include <stdio.h>
#include <stdlib.h>

#include "common.h"

int main(int argc, char **argv) {

    if (argc < 3) {
        fprintf(stderr,
                "Uso: %s <N> <ITERACOES>\n",
                argv[0]);

        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    int iteracoes = atoi(argv[2]);

    if (n < 3 || iteracoes < 1) {
        fprintf(stderr,
                "N deve ser >= 3 e ITERACOES >= 1\n");

        return EXIT_FAILURE;
    }

    /*
     * Dois buffers:
     * cur  = estado atual
     * prox = estado da próxima iteração
     */
    real_t *cur = aloca_matriz(n);
    real_t *prox = aloca_matriz(n);

    /*
     * Importante:
     * os dois buffers recebem as condições de contorno
     * antes da medição.
     */
    inicializa_matriz(cur, n);
    inicializa_matriz(prox, n);

    double t0 = agora_segundos();

    for (int it = 0; it < iteracoes; it++) {

        for (int i = 1; i < n - 1; i++) {

            for (int j = 1; j < n - 1; j++) {

                prox[IDX(i, j, n)] = CONST_QUARTO * (

                    cur[IDX(i - 1, j, n)] +
                    cur[IDX(i + 1, j, n)] +
                    cur[IDX(i, j - 1, n)] +
                    cur[IDX(i, j + 1, n)]

                );
            }
        }

        /*
         * Não é necessário aplicar as bordas novamente.
         * O stencil escreve somente:
         * i = 1 ... n-2
         * j = 1 ... n-2
         * Portanto as bordas nunca são modificadas.
         */

        real_t *tmp = cur;
        cur = prox;
        prox = tmp;
    }

    double t1 = agora_segundos();

    double tempo = t1 - t0;

    double soma = checksum(cur, n);

    fprintf(stderr,
            "[SEQ] %s N=%d iter=%d "
            "tempo=%.6f s checksum=%.6f\n",
            NOME_PRECISAO,
            n,
            iteracoes,
            tempo,
            soma);

    /*
     * CSV:
     *
     * versao
     * threads
     * schedule
     * chunk
     * n
     * iteracoes
     * tempo_s
     * checksum
     * precisao
     */
    printf(
        "seq,1,-,0,%d,%d,%.6f,%.6f,%s\n",
        n,
        iteracoes,
        tempo,
        soma,
        NOME_PRECISAO
    );

    free(cur);
    free(prox);

    return EXIT_SUCCESS;
}