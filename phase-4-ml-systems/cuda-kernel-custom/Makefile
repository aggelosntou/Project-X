NVCC      = nvcc
NVCCFLAGS = -O2 -arch=sm_70 -std=c++14 -DBENCHMARK_MAIN

SRCS = src/attention_naive.cu \
       src/attention_shared_mem.cu \
       src/flash_attention.cu \
       src/layer_norm_kernel.cu \
       src/benchmark.cu

all: benchmark

benchmark: $(SRCS)
	$(NVCC) $(NVCCFLAGS) -o benchmark $^

# Convenience targets to compile individual kernels stand-alone
naive: src/attention_naive.cu
	$(NVCC) -O2 -arch=sm_70 -std=c++14 -o naive_test $<

shared: src/attention_shared_mem.cu
	$(NVCC) -O2 -arch=sm_70 -std=c++14 -o shared_test $<

flash: src/flash_attention.cu
	$(NVCC) -O2 -arch=sm_70 -std=c++14 -o flash_test $<

layernorm: src/layer_norm_kernel.cu
	$(NVCC) -O2 -arch=sm_70 -std=c++14 -o layernorm_test $<

clean:
	rm -f benchmark naive_test shared_test flash_test layernorm_test

.PHONY: all naive shared flash layernorm clean
