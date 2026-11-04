/*
 * stencil_cuda.cu
 *
 * Versão CUDA do stencil 2D.
 *
 * Os dois buffers permanecem residentes na GPU
 * durante todas as iterações.
 *
 * Como os dois buffers começam com as condições
 * de contorno corretas e o kernel escreve somente
 * nas células internas, não são necessários kernels
 * adicionais para atualizar as bordas.
 *
 * São medidos:
 *
 * tempo_kernel_s:
 *     somente execução dos kernels
 *
 * tempo_end_to_end_s:
 *     H2D + kernels + D2H
 *
 * Uso:
 *
 * ./stencil_cuda <N> <ITERACOES>
 *
 * ou:
 *
 * ./stencil_cuda <N> <ITERACOES> <BLOCO_X> <BLOCO_Y>
 */
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "common.h"


#define CUDA_CHECK(chamada) do {                         \
                                                        \
    cudaError_t erro__ = (chamada);                     \
                                                        \
    if (erro__ != cudaSuccess) {                        \
                                                        \
        fprintf(                                        \
            stderr,                                     \
            "Erro CUDA em %s:%d: %s\n",                 \
            __FILE__,                                   \
            __LINE__,                                   \
            cudaGetErrorString(erro__)                   \
        );                                              \
                                                        \
        exit(EXIT_FAILURE);                             \
    }                                                   \
                                                        \
} while (0)


/*
 * Cada thread CUDA calcula exatamente
 * uma célula interna.
 */
__global__
void kernel_stencil(
    const real_t *__restrict__ cur,
    real_t *__restrict__ prox,
    int n
) {

    int j =
        (int)(
            blockIdx.x *
            blockDim.x +
            threadIdx.x
        ) + 1;

    int i =
        (int)(
            blockIdx.y *
            blockDim.y +
            threadIdx.y
        ) + 1;

    if (i < n - 1 &&
        j < n - 1) {

        prox[IDX(i, j, n)] =
            CONST_QUARTO * (

                cur[IDX(i - 1, j, n)] +
                cur[IDX(i + 1, j, n)] +
                cur[IDX(i, j - 1, n)] +
                cur[IDX(i, j + 1, n)]

            );
    }
}


int main(int argc, char **argv) {

    if (argc < 3) {

        fprintf(
            stderr,

            "Uso: %s "
            "<N> <ITERACOES> "
            "[BLOCO_X] [BLOCO_Y]\n",

            argv[0]
        );

        return EXIT_FAILURE;
    }

    int n =
        atoi(argv[1]);

    int iteracoes =
        atoi(argv[2]);

    int bloco_x =
        (argc > 3)
        ? atoi(argv[3])
        : 16;

    int bloco_y =
        (argc > 4)
        ? atoi(argv[4])
        : 16;

    if (n < 3 ||
        iteracoes < 1 ||
        bloco_x < 1 ||
        bloco_y < 1) {

        fprintf(
            stderr,

            "N>=3, ITERACOES>=1 e "
            "dimensões de bloco >=1 "
            "são obrigatórios\n"
        );

        return EXIT_FAILURE;
    }


    /*
     * Verifica se existe GPU CUDA.
     */
    int ndev = 0;

    CUDA_CHECK(
        cudaGetDeviceCount(
            &ndev
        )
    );

    if (ndev < 1) {

        fprintf(
            stderr,
            "Nenhuma GPU CUDA encontrada.\n"
        );

        return EXIT_FAILURE;
    }


    /*
     * Propriedades da GPU.
     */
    cudaDeviceProp prop;

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );


    /*
     * Valida as dimensões do bloco.
     */
    long long threads_por_bloco_ll =
        (long long)bloco_x *
        bloco_y;

    if (
        bloco_x >
            prop.maxThreadsDim[0]

        ||

        bloco_y >
            prop.maxThreadsDim[1]

        ||

        threads_por_bloco_ll >
            prop.maxThreadsPerBlock
    ) {

        fprintf(
            stderr,

            "Bloco %dx%d inválido "
            "para %s.\n"
            "Limites: "
            "x<=%d y<=%d "
            "threads/bloco<=%d\n",

            bloco_x,
            bloco_y,
            prop.name,

            prop.maxThreadsDim[0],
            prop.maxThreadsDim[1],
            prop.maxThreadsPerBlock
        );

        return EXIT_FAILURE;
    }


    size_t bytes =
        (size_t)n *
        (size_t)n *
        sizeof(real_t);


    /*
     * Memória do host.
     */
    real_t *h_ini =
        aloca_matriz(n);

    real_t *h_fim =
        aloca_matriz(n);

    inicializa_matriz(
        h_ini,
        n
    );


    /*
     * Memória da GPU.
     */
    real_t *d_cur = NULL;
    real_t *d_prox = NULL;

    CUDA_CHECK(
        cudaMalloc(
            (void **)&d_cur,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            (void **)&d_prox,
            bytes
        )
    );


    /*
     * Configuração 2D de bloco e grid.
     */
    dim3 blocoDim(
        (unsigned)bloco_x,
        (unsigned)bloco_y
    );

    dim3 gridDim(

        (unsigned)(
            (n - 2 +
             bloco_x - 1)
            /
            bloco_x
        ),

        (unsigned)(
            (n - 2 +
             bloco_y - 1)
            /
            bloco_y
        )

    );


    /*
     * Verifica limites do grid.
     */
    if (
        gridDim.x >
            (unsigned)prop.maxGridSize[0]

        ||

        gridDim.y >
            (unsigned)prop.maxGridSize[1]
    ) {

        fprintf(
            stderr,

            "Grid %ux%u excede "
            "o limite da GPU.\n",

            gridDim.x,
            gridDim.y
        );

        CUDA_CHECK(
            cudaFree(d_cur)
        );

        CUDA_CHECK(
            cudaFree(d_prox)
        );

        free(h_ini);
        free(h_fim);

        return EXIT_FAILURE;
    }


    /*
     * Eventos CUDA para medir somente
     * a região dos kernels.
     */
    cudaEvent_t ev_ini;
    cudaEvent_t ev_fim;

    CUDA_CHECK(
        cudaEventCreate(
            &ev_ini
        )
    );

    CUDA_CHECK(
        cudaEventCreate(
            &ev_fim
        )
    );


    /*
     * Tempo E2E
     * Começa ANTES das transferências H2D.
     */
    double t_total_ini =
        agora_segundos();


    /*
     * Os dois buffers precisam começar
     * com as mesmas condições de contorno.
     */
    CUDA_CHECK(
        cudaMemcpy(
            d_cur,
            h_ini,
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_prox,
            h_ini,
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    /*
     * Tempo de Kernel
     */
    CUDA_CHECK(
        cudaEventRecord(
            ev_ini
        )
    );


    for (int it = 0;
         it < iteracoes;
         it++) {

        kernel_stencil
        <<<gridDim, blocoDim>>>
        (
            d_cur,
            d_prox,
            n
        );

        /*
         * NÃO existem mais:
         *
         * kernel_borda_linhas
         * kernel_borda_colunas
         */

        real_t *tmp =
            d_cur;

        d_cur =
            d_prox;

        d_prox =
            tmp;
    }


    CUDA_CHECK(
        cudaEventRecord(
            ev_fim
        )
    );

    CUDA_CHECK(
        cudaEventSynchronize(
            ev_fim
        )
    );

    /*
     * Detecta erros de lançamento.
     */
    CUDA_CHECK(
        cudaGetLastError()
    );


    /*
     * Tempo dos kernels.
     */
    float tempo_kernel_ms =
        0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &tempo_kernel_ms,
            ev_ini,
            ev_fim
        )
    );


    /*
     * Resultado volta para o host.
     */
    CUDA_CHECK(
        cudaMemcpy(
            h_fim,
            d_cur,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    /*
     * Fim do tempo end-to-end.
     */
    double t_total_fim =
        agora_segundos();


    double tempo_kernel_s =
        (double)tempo_kernel_ms /
        1000.0;

    double tempo_end_to_end_s =
        t_total_fim -
        t_total_ini;


    /*
     * Checksum.
     */
    double soma =
        checksum(
            h_fim,
            n
        );


    int threads_por_bloco =
        bloco_x *
        bloco_y;


    fprintf(
        stderr,

        "[CUDA] %s "
        "GPU=%s "
        "N=%d "
        "iter=%d "
        "bloco=%dx%d "
        "threads/bloco=%d "
        "grid=%ux%u "
        "kernel=%.6f s "
        "end-to-end=%.6f s "
        "checksum=%.6f\n",

        NOME_PRECISAO,
        prop.name,

        n,
        iteracoes,

        bloco_x,
        bloco_y,

        threads_por_bloco,

        gridDim.x,
        gridDim.y,

        tempo_kernel_s,
        tempo_end_to_end_s,

        soma
    );


    /*
     * CSV:
     *
     * versao
     * block_x
     * block_y
     * threads_por_bloco
     * grid_x
     * grid_y
     * n
     * iteracoes
     * tempo_kernel_s
     * tempo_end_to_end_s
     * checksum
     * precisao
     */
    printf(
        "cuda,"
        "%d,%d,%d,"
        "%u,%u,"
        "%d,%d,"
        "%.6f,%.6f,"
        "%.6f,%s\n",

        bloco_x,
        bloco_y,

        threads_por_bloco,

        gridDim.x,
        gridDim.y,

        n,
        iteracoes,

        tempo_kernel_s,
        tempo_end_to_end_s,

        soma,
        NOME_PRECISAO
    );


    CUDA_CHECK(
        cudaEventDestroy(
            ev_ini
        )
    );

    CUDA_CHECK(
        cudaEventDestroy(
            ev_fim
        )
    );

    CUDA_CHECK(
        cudaFree(
            d_cur
        )
    );

    CUDA_CHECK(
        cudaFree(
            d_prox
        )
    );

    free(h_ini);
    free(h_fim);

    return EXIT_SUCCESS;
}
