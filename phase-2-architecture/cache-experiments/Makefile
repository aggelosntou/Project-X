CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -O2 -g
LDFLAGS = -lpthread

BUILD   = build
SRCS    = src/01_cache_size.c \
          src/02_cache_line_size.c \
          src/03_false_sharing.c \
          src/04_sequential_vs_random.c \
          src/05_matrix_traversal.c

TARGETS = $(BUILD)/cache_size \
          $(BUILD)/cache_line \
          $(BUILD)/false_sharing \
          $(BUILD)/seq_vs_random \
          $(BUILD)/matrix_traversal

.PHONY: all clean run-all run-01 run-02 run-03 run-04 run-05 benchmark

all: $(BUILD) $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/cache_size: src/01_cache_size.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD)/cache_line: src/02_cache_line_size.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD)/false_sharing: src/03_false_sharing.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/seq_vs_random: src/04_sequential_vs_random.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD)/matrix_traversal: src/05_matrix_traversal.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

run-01: $(BUILD)/cache_size
	./$(BUILD)/cache_size

run-02: $(BUILD)/cache_line
	./$(BUILD)/cache_line

run-03: $(BUILD)/false_sharing
	./$(BUILD)/false_sharing

run-04: $(BUILD)/seq_vs_random
	./$(BUILD)/seq_vs_random

run-05: $(BUILD)/matrix_traversal
	./$(BUILD)/matrix_traversal

run-all: all
	@echo "=== Experiment 1: Cache Size ==="
	./$(BUILD)/cache_size
	@echo ""
	@echo "=== Experiment 2: Cache Line Size ==="
	./$(BUILD)/cache_line
	@echo ""
	@echo "=== Experiment 3: False Sharing ==="
	./$(BUILD)/false_sharing
	@echo ""
	@echo "=== Experiment 4: Sequential vs Random ==="
	./$(BUILD)/seq_vs_random
	@echo ""
	@echo "=== Experiment 5: Matrix Traversal ==="
	./$(BUILD)/matrix_traversal

benchmark: all
	bash src/benchmark.sh

clean:
	rm -rf $(BUILD) results
