# Makefile for matrix-library-c
CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -O2 -Isrc
LDFLAGS = -lm

SRC_DIR = src
BUILD   = build

.PHONY: all test clean

all: $(BUILD)/test_matrix

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_matrix: $(BUILD) $(SRC_DIR)/matrix.c $(SRC_DIR)/test_matrix.c
	$(CC) $(CFLAGS) $(SRC_DIR)/matrix.c $(SRC_DIR)/test_matrix.c -o $@ $(LDFLAGS)

test: $(BUILD)/test_matrix
	./$(BUILD)/test_matrix

clean:
	rm -rf $(BUILD)
