# Makefile — file-system-toy

CC     = cc
CFLAGS = -Wall -Wextra -O2 -g -std=c11
SRC    = src
BIN    = bin

.PHONY: all clean test

all: $(BIN)/fs

$(BIN):
	mkdir -p $(BIN)

$(BIN)/fs.o: $(SRC)/fs.c $(SRC)/fs.h | $(BIN)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN)/main.o: $(SRC)/main.c $(SRC)/fs.h | $(BIN)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN)/fs: $(BIN)/fs.o $(BIN)/main.o
	$(CC) $(CFLAGS) $^ -o $@

test: $(BIN)/fs
	@echo "=== Toy filesystem smoke test ==="
	@rm -f test.img
	@echo "mkdir /docs"    | ./$(BIN)/fs test.img
	@echo "create /docs/hello.txt" | ./$(BIN)/fs test.img
	@echo "write /docs/hello.txt Hello, filesystem!" | ./$(BIN)/fs test.img
	@echo "read /docs/hello.txt"  | ./$(BIN)/fs test.img
	@echo "ls /docs"       | ./$(BIN)/fs test.img
	@echo "stat /docs/hello.txt"  | ./$(BIN)/fs test.img
	@rm -f test.img

clean:
	rm -rf $(BIN) test.img
