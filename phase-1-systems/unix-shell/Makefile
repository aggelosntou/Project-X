# Makefile — unix-shell

CC     = cc
CFLAGS = -Wall -Wextra -O2 -g -std=c11
SRC    = src
BIN    = bin

.PHONY: all clean

all: $(BIN)/shell

$(BIN):
	mkdir -p $(BIN)

$(BIN)/shell: $(SRC)/shell.c | $(BIN)
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(BIN)
