CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g

all: test_all

test_all: src/dynamic_array.c src/linked_list.c src/hash_map.c src/test_all.c
	$(CC) $(CFLAGS) -o test_all $^

run: test_all
	./test_all

clean:
	rm -f test_all
