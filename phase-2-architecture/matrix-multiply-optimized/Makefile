CC     = gcc
CFLAGS = -Wall -Wextra -std=c11 -O3 -g

# Detect architecture and enable appropriate SIMD flags
ARCH  := $(shell uname -m)
ifeq ($(ARCH),x86_64)
    SIMD_FLAGS = -mavx2 -mfma
else
    # Apple Silicon (arm64) — no AVX2, but has NEON (auto-vectorized by compiler)
    SIMD_FLAGS = -mcpu=native
endif

SRCS   = src/naive.c src/transposed.c src/tiled.c src/simd.c \
         src/tiled_simd.c src/benchmark.c
TARGET = matmul_benchmark

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) src/matmul.h
	$(CC) $(CFLAGS) $(SIMD_FLAGS) -o $@ $(SRCS) -lm

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
