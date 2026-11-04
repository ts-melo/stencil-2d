#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Precisão da matriz.
 *
 * Padrão: double (64 bits).
 * Compile com -DUSE_FLOAT (make PRECISAO=float) para usar float (32 bits).
 *
 * Todo o código usa real_t, portanto a troca vale para as versões
 * sequencial, Pthreads, OpenMP e CUDA.
 */
#ifdef USE_FLOAT
typedef float real_t;
#define NOME_PRECISAO "float"
#else
typedef double real_t;
#define NOME_PRECISAO "double"
#endif

/*
 * Constante do stencil já no tipo da matriz. Escrever "0.25 * x" com
 * x float promoveria a conta para double (mais lento na CPU e MUITO mais
 * lento em GPUs de consumo, que têm poucas unidades FP64).
 */
#define CONST_QUARTO ((real_t)0.25)

/*
 * Operações de ponto flutuante por célula atualizada:
 * 3 somas + 1 multiplicação = 4 FLOPs.
 * FLOPs totais = FLOPS_POR_CELULA * (N-2)^2 * ITERACOES
 */
#define FLOPS_POR_CELULA 4

/* Matriz NxN armazenada como vetor 1D contíguo (linha a linha). */
#define IDX(i, j, n) ((i) * (n) + (j))

#define TEMP_QUENTE 100.0
#define TEMP_FRIA     0.0

static inline real_t *aloca_matriz(int n) {
    size_t total = (size_t)n * (size_t)n;

    real_t *m = (real_t *)malloc(total * sizeof(real_t));

    if (!m) {
        fprintf(stderr, "Erro ao alocar matriz %dx%d\n", n, n);
        exit(EXIT_FAILURE);
    }

    return m;
}

/*
 * Condições de contorno de Dirichlet:
 *
 * borda superior = quente
 * borda inferior = fria
 * borda esquerda = fria
 * borda direita = fria
 *
 * As laterais são escritas depois, então os quatro cantos
 * terminam com TEMP_FRIA.
 */
static inline void aplica_bordas(real_t *m, int n) {
    for (int j = 0; j < n; j++) {
        m[IDX(0, j, n)] = TEMP_QUENTE;
        m[IDX(n - 1, j, n)] = TEMP_FRIA;
    }

    for (int i = 0; i < n; i++) {
        m[IDX(i, 0, n)] = TEMP_FRIA;
        m[IDX(i, n - 1, n)] = TEMP_FRIA;
    }
}

/*
 * Inicializa todo o interior com TEMP_FRIA
 * e depois aplica as condições de contorno.
 */
static inline void inicializa_matriz(real_t *m, int n) {
    size_t total = (size_t)n * (size_t)n;

    for (size_t k = 0; k < total; k++) {
        m[k] = TEMP_FRIA;
    }

    aplica_bordas(m, n);
}

/*
 * Checksum simples para validação das implementações.
 * A soma é acumulada em double mesmo quando a matriz é float.
 */
static inline double checksum(const real_t *m, int n) {
    size_t total = (size_t)n * (size_t)n;

    double soma = 0.0;

    for (size_t k = 0; k < total; k++) {
        soma += (double)m[k];   /* acumula sempre em double */
    }

    return soma;
}

/*
 * Relógio monotônico para medições de tempo.
 */
static inline double agora_segundos(void) {
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec +
           (double)ts.tv_nsec / 1e9;
}

#endif