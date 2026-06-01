CXX      = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -g -O2
SRCS     = src/main.cpp src/vm.cpp src/compiler.cpp \
           src/scanner.cpp src/chunk.cpp
TARGET   = lox_mini

.PHONY: all clean run-fib run-fizzbuzz run-counter

all: $(TARGET)

$(TARGET): $(SRCS) src/vm.hpp src/chunk.hpp src/compiler.hpp \
            src/scanner.hpp src/opcodes.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS)

run-fib: $(TARGET)
	./$(TARGET) programs/fib.lox

run-fizzbuzz: $(TARGET)
	./$(TARGET) programs/fizzbuzz.lox

run-counter: $(TARGET)
	./$(TARGET) programs/counter.lox

run-all: run-fib run-fizzbuzz run-counter

repl: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
