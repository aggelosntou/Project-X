CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g
LDFLAGS = -lpthread

all: test_pool

test_pool: src/thread_pool.c src/test_pool.c
	$(CC) $(CFLAGS) -o test_pool $^ $(LDFLAGS)

run: test_pool
	./test_pool

clean:
	rm -f test_pool
