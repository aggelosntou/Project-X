# Makefile — Parallel Matrix Multiplication
#
# Standard Linux build:
#   make
#   make run
#
# macOS with Homebrew libomp:
#   OpenMP is not included in Apple Clang by default.
#   Install libomp first:
#     brew install libomp
#   Then build with:
#     make MACOS=1
#   Which adds: -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include
#               -L$(brew --prefix libomp)/lib -lomp
#   You can also use GCC from Homebrew (brew install gcc) which ships with
#   full OpenMP support without extra flags:
#     make CC=gcc-14   (or whatever version brew installed)
#
# If you see a linker error about "omp_get_max_threads" being undefined,
# you are using Apple Clang without libomp — use one of the macOS options
# above.

CC     = gcc
CFLAGS = -Wall -Wextra -std=c11 -O3 -fopenmp
LDFLAGS = -lm -fopenmp

# macOS override: uncomment and adjust the libomp prefix if needed.
# LIBOMP_PREFIX = $(shell brew --prefix libomp)
# CFLAGS  = -Wall -Wextra -std=c11 -O3 -Xpreprocessor -fopenmp \
#            -I$(LIBOMP_PREFIX)/include
# LDFLAGS = -lm -L$(LIBOMP_PREFIX)/lib -lomp

all: benchmark

benchmark: src/matmul.c src/benchmark.c
	$(CC) $(CFLAGS) -o benchmark $^ $(LDFLAGS)

run: benchmark
	./benchmark

clean:
	rm -f benchmark
