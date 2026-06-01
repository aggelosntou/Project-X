# Makefile for neural-net-c
CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -O2 -Isrc
LDFLAGS = -lm

SRC = src/matrix.c src/nn.c src/data.c

.PHONY: all train_xor train_mnist_lite clean

all: build/train_xor build/train_mnist_lite

build:
	mkdir -p build

build/train_xor: build $(SRC) src/train_xor.c
	$(CC) $(CFLAGS) $(SRC) src/train_xor.c -o $@ $(LDFLAGS)

build/train_mnist_lite: build $(SRC) src/train_mnist_lite.c
	$(CC) $(CFLAGS) $(SRC) src/train_mnist_lite.c -o $@ $(LDFLAGS)

train_xor: build/train_xor
	./build/train_xor

train_mnist_lite: build/train_mnist_lite
	./build/train_mnist_lite

clean:
	rm -rf build
