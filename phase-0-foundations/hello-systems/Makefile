# hello-systems/Makefile
#
# A Makefile describes a dependency graph: "to build TARGET, run RECIPE,
# provided PREREQUISITES are up to date."
#
# Make only re-runs a recipe when a prerequisite is NEWER than the target
# (it compares modification timestamps).  This is why large projects don't
# recompile every file on every change.
#
# Syntax rules:
#   target : prerequisites
#   <TAB>  recipe line          ← the leading TAB is mandatory, not spaces
#
# Variables are expanded with $(VAR).
# $@ = the target name
# $< = the first prerequisite
# $^ = all prerequisites

# ── Compiler and flags ────────────────────────────────────────────────────

CC      = gcc
# C11 is required for _Static_assert and other modern features
CSTD    = -std=c11

# Warning flags:
#   -Wall   : enable a large set of common warnings
#   -Wextra : enable extra warnings not covered by -Wall
WARNINGS = -Wall -Wextra

# Release flags: -O2 asks the compiler to optimise for speed
RELEASE_FLAGS  = $(CSTD) $(WARNINGS) -O2

# Debug flags:
#   -g               : embed DWARF debug info so gdb/lldb can show source lines
#   -O0              : disable all optimisation (so variable values are predictable)
#   -fsanitize=address : AddressSanitizer — detects heap/stack buffer overflows,
#                        use-after-free, etc. at runtime with ~2x overhead
DEBUG_FLAGS    = $(CSTD) $(WARNINGS) -g -O0 -fsanitize=address

# Strict flags:
#   -Werror   : treat every warning as a compile error (good for CI)
#   -pedantic : reject any GNU extensions; demand standard C only
STRICT_FLAGS   = $(CSTD) -Wall -Wextra -Werror -pedantic

SRC     = src/main.c
TARGET  = hello

# ── Targets ───────────────────────────────────────────────────────────────

# .PHONY tells Make these are not real files — run the recipe unconditionally
.PHONY: all debug strict asm clean

# 'all' is the DEFAULT target (first target in the file)
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(RELEASE_FLAGS) -o $@ $^
	@echo "Built release binary: $@"
	@echo "  (optimised, no debug info, warnings enabled)"

debug: $(SRC)
	$(CC) $(DEBUG_FLAGS) -o $(TARGET)_debug $^
	@echo "Built debug binary: $(TARGET)_debug"
	@echo "  (AddressSanitizer enabled — will detect memory bugs at runtime)"

strict: $(SRC)
	$(CC) $(STRICT_FLAGS) -o $(TARGET)_strict $^
	@echo "Built strict binary: $(TARGET)_strict"
	@echo "  (-Werror: any warning is a compile error)"

# Generate human-readable assembly listing.
# Useful for understanding what the compiler ACTUALLY produces.
# -S     : stop after compiling, emit .s assembly file
# -masm=intel : use Intel syntax (optional, but more readable for many people)
asm: $(SRC)
	$(CC) $(RELEASE_FLAGS) -S -fverbose-asm -o src/main.s $<
	@echo "Assembly written to src/main.s"
	@echo "  Open it and look for the function names — you can read the machine code."

clean:
	rm -f $(TARGET) $(TARGET)_debug $(TARGET)_strict src/main.s
	@echo "Cleaned build artifacts."
