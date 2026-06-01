# Makefile — memory-allocator
#
# WHY separate compilation?
#   Building allocator.o independently means the test binary links against our
#   object file, not the system allocator. The linker resolves my_malloc etc.
#   from allocator.o before it ever looks at libc.

CC      = cc
CFLAGS  = -Wall -Wextra -O2 -g -std=c11
SRC     = src
BIN     = bin

.PHONY: all test clean

all: $(BIN)/test_allocator

$(BIN):
	mkdir -p $(BIN)

# Compile the allocator implementation into an object file
$(BIN)/allocator.o: $(SRC)/allocator.c $(SRC)/allocator.h | $(BIN)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile the test driver
$(BIN)/test_allocator.o: $(SRC)/test_allocator.c $(SRC)/allocator.h | $(BIN)
	$(CC) $(CFLAGS) -c $< -o $@

# Link both object files into the test binary
$(BIN)/test_allocator: $(BIN)/allocator.o $(BIN)/test_allocator.o
	$(CC) $(CFLAGS) $^ -o $@

test: $(BIN)/test_allocator
	./$(BIN)/test_allocator

clean:
	rm -rf $(BIN)
