# Stencil 2D — difusão de calor com Pthreads, OpenMP e CUDA

Simulação de difusão de calor em malha 2D (stencil de 5 pontos, condições de
contorno de Dirichlet) implementada em quatro versões, usada para estudar o
impacto de interface de paralelização, escalonamento do OpenMP, tamanho de
bloco CUDA, tamanho da malha e precisão numérica (`double`/`float`) sobre
tempo, vazão (MFLOPS) e, na GPU, energia.

> Amanda Dantas, Bruna Surur Bergara, Luís Felipe Salgado Mitidieri, Rafaela de Paula Vilela, Thaís Melo

> Programação Concorrente e Distribuída

> UNIFESP - Universidade Fededral de São Paulo

## Versões

| Arquivo | Versão |
|---|---|
| `src/stencil_seq.c` | Sequencial (baseline) |
| `src/stencil_pthreads.c` | Pthreads (decomposição por linhas + barreira com mutex/condvar) |
| `src/stencil_openmp.c` | OpenMP (`schedule(runtime)`, definido por `OMP_SCHEDULE`) |
| `src/stencil_cuda.cu` | CUDA (grid/bloco 2D, um thread por célula) |
| `src/common.h` | Tipo `real_t`, alocação, inicialização, checksum, relógio |

A matriz é lógica 2D (N×N), armazenada num vetor contíguo por linhas
(`IDX(i,j,n)`). Cada célula interna recebe a média dos 4 vizinhos
(3 somas + 1 multiplicação = **4 FLOPs/célula**).

## Requisitos

- Linux ou WSL2 (testado em Ubuntu no WSL2), `gcc`, `make`, OpenMP (`-fopenmp`)
- Para a GPU: GPU NVIDIA + driver, CUDA Toolkit (`nvcc`) e um `g++` aceito
  pelo `nvcc` (para o CUDA 12.8, gcc ≤ 14)
- Python 3 com `pandas` e `matplotlib` para a análise
  (`sudo apt install python3-pandas python3-matplotlib`)

## Compilar

A precisão é escolhida em tempo de compilação; cada uma usa seu diretório.

```bash
make                         # CPU, double  -> bin/double/
make PRECISAO=float          # CPU, float   -> bin/float/
make cuda PRECISAO=float     # GPU, float   -> bin/float/stencil_cuda
```

Para outra GPU, ajuste a arquitetura (padrão `sm_86`, RTX 2050):
`make cuda CUDA_ARCH=sm_75`. Se o `nvcc` recusar a versão do gcc:
`make cuda NVCC_CCBIN=/usr/bin/g++-12`.

`make clean` remove só `bin/`. `make clean-results` remove também os
resultados e gráficos.

## Executar um binário

```bash
./bin/double/stencil_seq 2000 1000                       # N ITERACOES
./bin/double/stencil_pthreads 2000 1000 8                # N ITERACOES THREADS
OMP_NUM_THREADS=8 OMP_SCHEDULE=guided ./bin/double/stencil_openmp 2000 1000
./bin/double/stencil_cuda 2000 1000 32 8                 # N ITERACOES BLOCO_X BLOCO_Y
```

Cada execução imprime uma linha CSV no `stdout` (e um resumo legível no `stderr`).

## Experimentos

Os scripts compilam sozinhos (não precisa rodar `make` antes) e gravam em
`results/`. Por padrão rodam `double` e `float`.

```bash
# GPU: 3 tamanhos x 5 blocos x 2 precisões x 10 repetições
export NVCC_CCBIN=/usr/bin/g++-12          # se necessário
bash scripts/run_gpu_experiments.sh

# CPU: varredura completa (threads 1 2 4 8 12; static/dynamic,64/guided)
bash scripts/run_experiments.sh

# CPU com as MESMAS iterações da GPU (comparação CPU x GPU)
TAMANHOS="pequena,512,50000 media,2000,5000 grande,8000,400" \
THREADS_LISTA="4 8 12" SCHEDULES="static guided" \
bash scripts/run_experiments.sh
```

Variáveis úteis: `PRECISOES="float"`, `REPETICOES=3` (teste rápido),
`TAMANHOS="rotulo,N,ITER ..."`, `THREADS_LISTA`, `SCHEDULES`.

### Análise

```bash
python3 scripts/analyze_results.py results/resultados_cpu_<data>.csv
python3 scripts/analyze_gpu_results.py results/resultados_gpu_<data>.csv \
                                       results/resultados_cpu_<data>.csv
```

Gera tabelas (`results/graficos/resumo.csv`,
`results/graficos_gpu/resumo_gpu.csv`, `results/graficos_gpu/comparacao_cpu_gpu.csv`)
e gráficos em `results/graficos/` e `results/graficos_gpu/`. A comparação
CPU × GPU só é feita onde N **e** número de iterações coincidem.

## Métricas

- **Speedup** = T_seq / T_paralelo; **Eficiência** = speedup / threads
- **MFLOPS** = 4·(N−2)²·ITER / (10⁶ · t)
- **Energia (somente GPU)**: potência da placa via `nvidia-smi` (`power.draw`),
  média só das amostras com a GPU ativa; `E = P̄ × t_end-to-end`;
  **MFLOPS/W** = MFLOPS / P̄. A energia da CPU **não** é medida (o WSL2 não
  expõe RAPL).

### Escopo dos tempos

| Medida | Escopo |
|---|---|
| Tempo CPU | laço principal (`clock_gettime`) |
| Tempo GPU (kernel) | laço de kernels (`cudaEvent`) |
| Tempo GPU (end-to-end) | inclui alocação e transferências PCIe |

### Validação

O checksum (soma da matriz, acumulada em `double`) de Pthreads e OpenMP deve ser
idêntico ao do sequencial. O script de GPU valida o checksum CUDA contra o
sequencial (N=300, 300 iterações) com tolerância relativa de 1e-9 (`double`) ou
1e-5 (`float`) antes de medir.

## Resultados incluídos

| Arquivo | Conteúdo |
|---|---|
| `results/resultados_gpu_*.csv` | Dados brutos da GPU (N=512/2000/8000; 50000/5000/400 iterações) |
| `results/resultados_cpu_*.csv` | Dados brutos da CPU com as mesmas iterações da GPU |
| `results/graficos*/resumo*.csv`, `comparacao_cpu_gpu.csv` | Tabelas agregadas |
| `results/graficos*/*.png` | Gráficos |

Ambiente dos experimentos: ver artigo (CPU Intel Core i5-12450H, GPU
NVIDIA GeForce RTX 2050, WSL2 Ubuntu, CUDA 12.8).

## Solução de problemas (CUDA no WSL2)

- `nvcc: command not found`: o `CUDA Version` do `nvidia-smi` é só o suportado
  pelo driver. Instale o **CUDA Toolkit** (repositório WSL-Ubuntu da NVIDIA) e
  adicione `/usr/local/cuda/bin` ao `PATH`. Não instale driver dentro do WSL.
- `unsupported GNU version! gcc versions later than 14`: instale `g++-12` (ou 13/14)
  e use `NVCC_CCBIN=/usr/bin/g++-12`.
- `exception specification is incompatible ... cospi/sinpi/rsqrt`: incompatibilidade
  entre o CUDA 12.8 e o glibc recente. Contornos: adicionar `noexcept (true)` a essas
  declarações em `crt/math_functions.h` do toolkit, ou usar CUDA ≥ 12.9 (não verificado).
  Não use `-D__MATH_FUNCTIONS_H__`: isso quebra `min`/`max`.
- `nvidia-smi -c` **não** é contagem de amostras (é `--compute-mode`); os scripts não o usam.


