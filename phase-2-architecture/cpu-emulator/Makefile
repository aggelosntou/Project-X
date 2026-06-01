CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -g -O2
SRCS    = src/main.c src/cpu.c src/assembler.c
TARGET  = cpu_emulator

.PHONY: all clean run-demo run-fibonacci run-factorial run-bubblesort

all: $(TARGET)

$(TARGET): $(SRCS) src/cpu.h src/assembler.h
	$(CC) $(CFLAGS) -o $@ $(SRCS)

run-demo: $(TARGET)
	./$(TARGET) --demo

run-fibonacci: $(TARGET)
	@echo "=== Fibonacci ==="
	./$(TARGET) src/programs/fibonacci.asm

run-factorial: $(TARGET)
	@echo "=== Factorial ==="
	./$(TARGET) src/programs/factorial.asm

run-bubblesort: $(TARGET)
	@echo "=== Bubble Sort ==="
	./$(TARGET) src/programs/bubble_sort.asm

run-all: run-fibonacci run-factorial run-bubblesort

clean:
	rm -f $(TARGET)
