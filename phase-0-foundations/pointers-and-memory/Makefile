# pointers-and-memory/Makefile

CC       = gcc
CSTD     = -std=c11
WARNINGS = -Wall -Wextra
RELEASE  = $(CSTD) $(WARNINGS) -O2
DEBUG    = $(CSTD) $(WARNINGS) -g -O0 -fsanitize=address,undefined

SRCDIR   = src

TARGETS  = pointers_basics stack_vs_heap arrays_and_pointers function_pointers

.PHONY: all debug clean

all: $(TARGETS)

pointers_basics: $(SRCDIR)/pointers_basics.c
	$(CC) $(RELEASE) -o $@ $<
	@echo "Built: $@"

stack_vs_heap: $(SRCDIR)/stack_vs_heap.c
	$(CC) $(RELEASE) -o $@ $<
	@echo "Built: $@"

arrays_and_pointers: $(SRCDIR)/arrays_and_pointers.c
	$(CC) $(RELEASE) -o $@ $<
	@echo "Built: $@"

function_pointers: $(SRCDIR)/function_pointers.c
	# -lm links the math library (needed for fabs, sqrt on some systems)
	$(CC) $(RELEASE) -o $@ $< -lm
	@echo "Built: $@"

# Debug builds with AddressSanitizer + UBSan — catches memory bugs at runtime
debug:
	$(CC) $(DEBUG) -o pointers_basics_dbg    $(SRCDIR)/pointers_basics.c
	$(CC) $(DEBUG) -o stack_vs_heap_dbg      $(SRCDIR)/stack_vs_heap.c
	$(CC) $(DEBUG) -o arrays_and_pointers_dbg $(SRCDIR)/arrays_and_pointers.c
	$(CC) $(DEBUG) -o function_pointers_dbg  $(SRCDIR)/function_pointers.c -lm
	@echo "Debug builds complete (run them to see ASan/UBSan output)"

clean:
	rm -f $(TARGETS) pointers_basics_dbg stack_vs_heap_dbg \
	      arrays_and_pointers_dbg function_pointers_dbg
	@echo "Cleaned."
