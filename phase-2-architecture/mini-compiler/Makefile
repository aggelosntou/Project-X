CC     = gcc
CFLAGS = -Wall -Wextra -std=c11 -g

SRCS = src/lexer.c src/parser.c src/codegen.c src/main.c

# macOS and Linux require different linker flags when assembling
# the generated .s files.
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    LINK_FLAGS =
else
    LINK_FLAGS = -no-pie
endif

all: calc

calc: $(SRCS)
	$(CC) $(CFLAGS) -o calc $(SRCS)

# --- Test targets ------------------------------------------------
test: calc test_add test_loop test_fizzbuzz

test_add: calc
	./calc programs/add.calc
	gcc $(LINK_FLAGS) add.s -o add_prog
	@echo "add_prog output (expected: 42):"
	./add_prog

test_loop: calc
	./calc programs/loop.calc
	gcc $(LINK_FLAGS) loop.s -o loop_prog
	@echo "loop_prog output (expected: 5050):"
	./loop_prog

test_fizzbuzz: calc
	./calc programs/fizzbuzz.calc
	gcc $(LINK_FLAGS) fizzbuzz.s -o fizzbuzz_prog
	@echo "fizzbuzz_prog output:"
	./fizzbuzz_prog

test_expressions: calc
	./calc programs/expressions.calc
	gcc $(LINK_FLAGS) expressions.s -o expressions_prog
	@echo "expressions_prog output:"
	./expressions_prog

# --- Dump AST for a program (export CALC_DUMP_AST=1) -------------
dump_ast:
	CALC_DUMP_AST=1 ./calc programs/fizzbuzz.calc

clean:
	rm -f calc *.s *_prog src/*.o

.PHONY: all test test_add test_loop test_fizzbuzz test_expressions \
        dump_ast clean
