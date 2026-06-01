# Mini Database — B-Tree Backed REPL

A from-scratch relational database with a B-tree index, page-based storage,
and a SQL REPL — approximately 1,000 lines of C.

## Build & Run

```bash
make              # build the minidb binary
./minidb          # opens test.db (creates if absent)
./minidb mydb.db  # use a custom file
make test         # smoke test: insert + select + persist
```

## SQL Dialect

```sql
-- Insert a row (id must be a positive integer, name single-quoted)
INSERT INTO rows VALUES (1, 'Alice', 95.5);
INSERT INTO rows VALUES (2, 'Bob', 87.0);

-- Select all rows (ordered by id via B-tree in-order traversal)
SELECT * FROM rows;

-- Lookup by primary key (O(log n) B-tree search)
SELECT * FROM rows WHERE id = 1;

-- Exit and save
exit
```

## Example Session

```
$ ./minidb students.db
Mini-Database REPL
Database file: students.db
(0 row(s) loaded from disk)

db> INSERT INTO rows VALUES (1, 'Alice', 95.5);
Inserted row (id=1, name='Alice', score=95.50)
db> INSERT INTO rows VALUES (2, 'Bob', 87.0);
Inserted row (id=2, name='Bob', score=87.00)
db> INSERT INTO rows VALUES (3, 'Carol', 91.3);
Inserted row (id=3, name='Carol', score=91.30)
db> SELECT * FROM rows;
     id | name                             |    score
-------+----------------------------------+----------
      1 | Alice                            |    95.50
      2 | Bob                              |    87.00
      3 | Carol                            |    91.30
  3 row(s)
db> SELECT * FROM rows WHERE id = 2;
     id | name                             |    score
-------+----------------------------------+----------
      2 | Bob                              |    87.00
db> exit
Database saved to 'students.db'. Goodbye.

$ ./minidb students.db
(3 row(s) loaded from disk)
db> SELECT * FROM rows;
...  (all 3 rows still there)
```

## Architecture

```
main.c         REPL: read line, parse, dispatch
sql_parser.c   Recognise INSERT / SELECT / WHERE
table.c        Row storage: serialise to pager pages;
               maintain B-tree index (id → row_number)
pager.c        Page cache: load pages lazily, flush on close
btree.c        B-tree (order 3): insert, search, delete, print
```

## Source Files

| File              | Lines | Purpose                                    |
|-------------------|-------|--------------------------------------------|
| `src/btree.h/.c`  | ~380  | Order-3 B-tree with full insert/delete      |
| `src/pager.h/.c`  | ~120  | 4 KiB page cache over a binary file         |
| `src/table.h/.c`  | ~160  | Row serialisation, index, select all        |
| `src/sql_parser.h/.c` | ~170 | Minimal INSERT/SELECT SQL parser         |
| `src/main.c`      | ~80   | REPL driver                                 |

## Persistence

Rows are stored at fixed 48-byte offsets in pages (85 rows/page).
On close, all modified pages are flushed to disk.
On open, the index is rebuilt by scanning every page slot — O(n) startup
is acceptable for this toy database; production systems store the index
itself on disk (SQLite's B+ tree stores data in the tree nodes).
