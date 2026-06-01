CC     = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
LDFLAGS = -lpthread

all: test_lfq bench_lfq

test_lfq: src/lfq.c src/test_lfq.c
	$(CC) $(CFLAGS) -o test_lfq $^ $(LDFLAGS)

bench_lfq: src/lfq.c src/bench_lfq.c
	$(CC) $(CFLAGS) -o bench_lfq $^ $(LDFLAGS)

clean:
	rm -f test_lfq bench_lfq
