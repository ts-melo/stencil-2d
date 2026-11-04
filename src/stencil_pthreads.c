#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "common.h"

/* =========================================================
 * BARREIRA PORTÁVEL
 * Funciona em macOS e Linux/WSL
 * ========================================================= */

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;

    int total;
    int esperando;
    unsigned geracao;

} barreira_t;


static void erro_pthread(
    const char *operacao,
    int rc
) {
    if (rc != 0) {
        fprintf(
            stderr,
            "%s falhou: %s\n",
            operacao,
            strerror(rc)
        );

        exit(EXIT_FAILURE);
    }
}


static void barreira_init(
    barreira_t *b,
    int total
) {
    if (total < 1) {
        fprintf(
            stderr,
            "Numero invalido de threads.\n"
        );

        exit(EXIT_FAILURE);
    }

    int rc;

    rc = pthread_mutex_init(
        &b->mutex,
        NULL
    );

    erro_pthread(
        "pthread_mutex_init",
        rc
    );


    rc = pthread_cond_init(
        &b->cond,
        NULL
    );

    if (rc != 0) {
        pthread_mutex_destroy(
            &b->mutex
        );

        erro_pthread(
            "pthread_cond_init",
            rc
        );
    }


    b->total = total;
    b->esperando = 0;
    b->geracao = 0;
}


static void barreira_wait(
    barreira_t *b
) {
    int rc;

    rc = pthread_mutex_lock(
        &b->mutex
    );

    erro_pthread(
        "pthread_mutex_lock",
        rc
    );


    unsigned minha_geracao =
        b->geracao;


    b->esperando++;


    /*
     * Última thread que chegou.
     */
    if (
        b->esperando ==
        b->total
    ) {

        b->esperando = 0;
        b->geracao++;


        rc = pthread_cond_broadcast(
            &b->cond
        );

        erro_pthread(
            "pthread_cond_broadcast",
            rc
        );


        rc = pthread_mutex_unlock(
            &b->mutex
        );

        erro_pthread(
            "pthread_mutex_unlock",
            rc
        );

        return;
    }


    /*
     * As demais threads aguardam.
     */
    while (
        minha_geracao ==
        b->geracao
    ) {

        rc = pthread_cond_wait(
            &b->cond,
            &b->mutex
        );

        erro_pthread(
            "pthread_cond_wait",
            rc
        );
    }


    rc = pthread_mutex_unlock(
        &b->mutex
    );

    erro_pthread(
        "pthread_mutex_unlock",
        rc
    );
}


static void barreira_destroy(
    barreira_t *b
) {
    int rc;

    rc = pthread_cond_destroy(
        &b->cond
    );

    erro_pthread(
        "pthread_cond_destroy",
        rc
    );


    rc = pthread_mutex_destroy(
        &b->mutex
    );

    erro_pthread(
        "pthread_mutex_destroy",
        rc
    );
}


/* =========================================================
 * DADOS DE CADA THREAD
 * ========================================================= */

typedef struct {

    int n;
    int iteracoes;

    int linha_ini;
    int linha_fim;

    real_t *A;
    real_t *B;

    barreira_t *barreira;

} arg_thread_t;


/* =========================================================
 * TRABALHO DAS THREADS
 * ========================================================= */

static void *trabalho(
    void *arg
) {

    arg_thread_t *a =
        (arg_thread_t *)arg;


    real_t *cur =
        a->A;

    real_t *prox =
        a->B;


    for (
        int it = 0;
        it < a->iteracoes;
        it++
    ) {

        /*
         * Cada thread processa
         * apenas suas próprias linhas.
         */
        for (
            int i = a->linha_ini;
            i < a->linha_fim;
            i++
        ) {

            for (
                int j = 1;
                j < a->n - 1;
                j++
            ) {

                prox[IDX(i, j, a->n)] =
                    CONST_QUARTO * (

                        cur[IDX(i - 1, j, a->n)] +

                        cur[IDX(i + 1, j, a->n)] +

                        cur[IDX(i, j - 1, a->n)] +

                        cur[IDX(i, j + 1, a->n)]

                    );
            }
        }


        /*
         * Todas precisam terminar antes
         * da próxima iteração.
         */
        barreira_wait(
            a->barreira
        );


        /*
         * Troca local dos buffers.
         */
        real_t *tmp =
            cur;

        cur =
            prox;

        prox =
            tmp;
    }


    /*
     * Indica qual buffer ficou com
     * o resultado final.
     */
    a->A =
        cur;


    return NULL;
}


/* =========================================================
 * MAIN
 * ========================================================= */

int main(
    int argc,
    char **argv
) {

    if (argc != 4) {

        fprintf(
            stderr,
            "Uso: %s <N> <ITERACOES> <NTHREADS>\n",
            argv[0]
        );

        return EXIT_FAILURE;
    }


    int n =
        atoi(argv[1]);

    int iteracoes =
        atoi(argv[2]);

    int nthreads =
        atoi(argv[3]);


    if (
        n < 3 ||
        iteracoes < 1 ||
        nthreads < 1
    ) {

        fprintf(
            stderr,
            "N>=3, ITERACOES>=1 e NTHREADS>=1 sao obrigatorios\n"
        );

        return EXIT_FAILURE;
    }


    /* =====================================================
     * MATRIZES
     * ===================================================== */

    real_t *A =
        aloca_matriz(n);

    real_t *B =
        aloca_matriz(n);


    inicializa_matriz(
        A,
        n
    );

    inicializa_matriz(
        B,
        n
    );


    /* =====================================================
     * ESTRUTURAS DAS THREADS
     * ===================================================== */

    pthread_t *tids =
        malloc(
            (size_t)nthreads *
            sizeof(*tids)
        );


    arg_thread_t *args =
        malloc(
            (size_t)nthreads *
            sizeof(*args)
        );


    if (
        tids == NULL ||
        args == NULL
    ) {

        fprintf(
            stderr,
            "Erro ao alocar estruturas das threads.\n"
        );

        free(tids);
        free(args);
        free(A);
        free(B);

        return EXIT_FAILURE;
    }


    /* =====================================================
     * BARREIRA
     * ===================================================== */

    barreira_t barreira;

    barreira_init(
        &barreira,
        nthreads
    );


    /* =====================================================
     * DIVISÃO DAS LINHAS
     * ===================================================== */

    int linhas_internas =
        n - 2;


    int base =
        linhas_internas /
        nthreads;


    int resto =
        linhas_internas %
        nthreads;


    int inicio = 1;


    double t0 =
        agora_segundos();


    /* =====================================================
     * CRIAÇÃO DAS THREADS
     * ===================================================== */

    for (
        int t = 0;
        t < nthreads;
        t++
    ) {

        int qtd =
            base +
            (
                t < resto
                ? 1
                : 0
            );


        args[t].n =
            n;

        args[t].iteracoes =
            iteracoes;

        args[t].linha_ini =
            inicio;

        args[t].linha_fim =
            inicio +
            qtd;

        args[t].A =
            A;

        args[t].B =
            B;

        args[t].barreira =
            &barreira;


        inicio +=
            qtd;


        int rc =
            pthread_create(
                &tids[t],
                NULL,
                trabalho,
                &args[t]
            );


        if (rc != 0) {

            fprintf(
                stderr,
                "pthread_create falhou na thread %d: %s\n",
                t,
                strerror(rc)
            );

            exit(EXIT_FAILURE);
        }
    }


    /* =====================================================
     * JOIN
     * ===================================================== */

    for (
        int t = 0;
        t < nthreads;
        t++
    ) {

        int rc =
            pthread_join(
                tids[t],
                NULL
            );


        erro_pthread(
            "pthread_join",
            rc
        );
    }


    double t1 =
        agora_segundos();


    /* =====================================================
     * RESULTADOS
     * ===================================================== */

    double tempo =
        t1 - t0;


    real_t *final =
        args[0].A;


    double soma =
        checksum(
            final,
            n
        );


    fprintf(
        stderr,

        "[PTHREADS] %s "
        "N=%d "
        "iter=%d "
        "threads=%d "
        "tempo=%.6f s "
        "checksum=%.6f\n",

        NOME_PRECISAO,
        n,
        iteracoes,
        nthreads,
        tempo,
        soma
    );


    printf(
        "pthreads,%d,-,0,%d,%d,%.6f,%.6f,%s\n",

        nthreads,
        n,
        iteracoes,
        tempo,
        soma,
        NOME_PRECISAO
    );


    /* =====================================================
     * LIMPEZA
     * ===================================================== */

    barreira_destroy(
        &barreira
    );


    free(tids);
    free(args);

    free(A);
    free(B);


    return EXIT_SUCCESS;
}