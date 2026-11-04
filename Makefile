UNAME_S := $(shell uname -s)

SRC = src

# Precisao da matriz: double (padrao) ou float.
#   make                      -> bin/double/...
#   make PRECISAO=float       -> bin/float/...
#   make cuda PRECISAO=float
PRECISAO ?= double
BIN_ROOT = bin
BIN = $(BIN_ROOT)/$(PRECISAO)

CC = gcc
NVCC = nvcc

CFLAGS = -O3 -Wall -Wextra -std=gnu11

ifeq ($(PRECISAO),float)
PREC_FLAG = -DUSE_FLOAT
else ifeq ($(PRECISAO),double)
PREC_FLAG =
else
$(error PRECISAO deve ser 'double' ou 'float' (recebido: '$(PRECISAO)'))
endif

CFLAGS += $(PREC_FLAG)

# RTX 2050 (Ampere, compute capability 8.6). Para outra GPU: make cuda CUDA_ARCH=sm_75
CUDA_ARCH ?= sm_86

# Compilador host do nvcc. Vazio = usa o g++ padrao do sistema.
# Se o nvcc reclamar da versao do gcc: make cuda NVCC_CCBIN=/usr/bin/g++-12
NVCC_CCBIN ?=

# Flags extras para o nvcc (ex.: NVCC_EXTRA="-D__MATH_FUNCTIONS_H__")
NVCC_EXTRA ?=


# macOS

ifeq ($(UNAME_S),Darwin)

CC = clang

OMP_PREFIX := $(shell brew --prefix libomp)

OPENMP_FLAGS = \
	-Xpreprocessor -fopenmp \
	-I$(OMP_PREFIX)/include \
	-L$(OMP_PREFIX)/lib \
	-Wl,-rpath,$(OMP_PREFIX)/lib \
	-lomp

else

# Linux / WSL

OPENMP_FLAGS = -fopenmp

endif


# CPU

all: \
	$(BIN)/stencil_seq \
	$(BIN)/stencil_pthreads \
	$(BIN)/stencil_openmp


$(BIN):
	mkdir -p $(BIN)


# Sequencial

$(BIN)/stencil_seq: \
	$(SRC)/stencil_seq.c \
	$(SRC)/common.h | $(BIN)

	$(CC) $(CFLAGS) \
	-o $@ \
	$(SRC)/stencil_seq.c \
	-lm


# Pthreads

$(BIN)/stencil_pthreads: \
	$(SRC)/stencil_pthreads.c \
	$(SRC)/common.h | $(BIN)

	$(CC) $(CFLAGS) \
	-pthread \
	-o $@ \
	$(SRC)/stencil_pthreads.c \
	-lm

# OpenMP

$(BIN)/stencil_openmp: \
	$(SRC)/stencil_openmp.c \
	$(SRC)/common.h | $(BIN)

	$(CC) $(CFLAGS) \
	$(OPENMP_FLAGS) \
	-o $@ \
	$(SRC)/stencil_openmp.c \
	-lm


# CUDA

cuda: $(BIN)/stencil_cuda


$(BIN)/stencil_cuda: \
	$(SRC)/stencil_cuda.cu \
	$(SRC)/common.h | $(BIN)

	$(NVCC) \
	-O3 \
	$(PREC_FLAG) \
	$(if $(NVCC_CCBIN),-ccbin $(NVCC_CCBIN),) \
	$(NVCC_EXTRA) \
	-Xcompiler -Wall,-Wextra \
	-arch=$(CUDA_ARCH) \
	-I$(SRC) \
	-o $@ \
	$(SRC)/stencil_cuda.cu


# Limpeza
#   make clean          -> remove so os binarios (bin/)
#   make clean-results  -> remove TAMBEM os resultados e graficos (cuidado: se
#                          estiverem versionados no git, recupere com git checkout)

clean:
	rm -rf $(BIN_ROOT)

clean-results:
	rm -rf \
	results/*.csv \
	results/graficos \
	results/graficos_gpu


.PHONY: all cuda clean clean-results
