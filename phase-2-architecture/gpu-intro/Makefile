NVCC      = nvcc
NVCCFLAGS = -O2 -arch=sm_70

TARGETS = 01_hello_gpu 02_vector_add 03_matrix_multiply \
          04_memory_patterns 05_reduction

all: $(TARGETS)

01_hello_gpu: src/01_hello_gpu.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

02_vector_add: src/02_vector_add.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

03_matrix_multiply: src/03_matrix_multiply.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

04_memory_patterns: src/04_memory_patterns.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

05_reduction: src/05_reduction.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
