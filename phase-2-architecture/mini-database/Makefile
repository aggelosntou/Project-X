CC     = gcc
CFLAGS = -Wall -Wextra -std=c11 -g -O2

SRCS = src/btree.c src/pager.c src/table.c src/sql_parser.c src/main.c
TARGET = minidb

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

# Quick smoke test
test: $(TARGET)
	@echo "=== Creating fresh database ==="
	rm -f test_smoke.db
	@printf "INSERT INTO rows VALUES (1, 'Alice', 95.5);\nINSERT INTO rows VALUES (2, 'Bob', 87.0);\nINSERT INTO rows VALUES (3, 'Carol', 91.3);\nSELECT * FROM rows;\nSELECT * FROM rows WHERE id = 2;\nexit\n" | ./minidb test_smoke.db
	@echo ""
	@echo "=== Reopening database (persistence test) ==="
	@printf "SELECT * FROM rows;\nexit\n" | ./minidb test_smoke.db
	rm -f test_smoke.db

# B-tree unit test
test_btree: src/btree.c
	$(CC) $(CFLAGS) -DBTREE_TEST -o btree_test src/btree.c
	./btree_test

clean:
	rm -f $(TARGET) btree_test *.db

.PHONY: all test test_btree clean
