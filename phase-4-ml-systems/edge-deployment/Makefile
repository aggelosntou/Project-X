CC      = gcc
CFLAGS  = -O3 -Wall -std=c11
LDFLAGS = -lm

# Default target: compile C inference engine
all: c_inference

c_inference: src/c_inference.c
	$(CC) $(CFLAGS) -o c_inference $< $(LDFLAGS)

# Export model weights (creates model.bin and model.npz)
export:
	python3 src/export_weights.py

# Run benchmark (requires c_inference binary and model.bin)
bench: c_inference export
	python3 src/benchmark_inference.py

clean:
	rm -f c_inference model.bin model.npz

.PHONY: all export bench clean
