# string-library/Makefile

CC       = gcc
CSTD     = -std=c11
WARNINGS = -Wall -Wextra
CFLAGS   = $(CSTD) $(WARNINGS) -O2

SRCDIR   = src
LIB      = libmystring.a
TEST_BIN = test_mystring

.PHONY: all test clean

all: $(LIB)

# Build object files
$(SRCDIR)/mystring.o: $(SRCDIR)/mystring.c $(SRCDIR)/mystring.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Build the static library archive
# ar rc: 'r' = insert with replacement, 'c' = create archive if missing
# ranlib: generate an index for fast linking (required on some systems)
$(LIB): $(SRCDIR)/mystring.o
	ar rc $@ $^
	ranlib $@
	@echo "Built static library: $@"
	@echo "  Link it with: gcc your_program.c -L. -lmystring"

# Build and run the test suite
$(TEST_BIN): $(SRCDIR)/test_mystring.c $(SRCDIR)/mystring.c $(SRCDIR)/mystring.h
	$(CC) $(CSTD) $(WARNINGS) -g -fsanitize=address,undefined \
	    -o $@ $(SRCDIR)/test_mystring.c $(SRCDIR)/mystring.c
	@echo "Built test binary: $@"

test: $(TEST_BIN)
	@echo "Running tests..."
	./$(TEST_BIN)

clean:
	rm -f $(SRCDIR)/mystring.o $(LIB) $(TEST_BIN)
	@echo "Cleaned."
