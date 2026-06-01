# Makefile — http-server

CC     = cc
CFLAGS = -Wall -Wextra -O2 -g -std=c11
SRC    = src
BIN    = bin

.PHONY: all clean

all: $(BIN)/server

$(BIN):
	mkdir -p $(BIN)

$(BIN)/http_parser.o: $(SRC)/http_parser.c $(SRC)/http_parser.h | $(BIN)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN)/server.o: $(SRC)/server.c $(SRC)/http_parser.h | $(BIN)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN)/server: $(BIN)/http_parser.o $(BIN)/server.o
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -rf $(BIN)
