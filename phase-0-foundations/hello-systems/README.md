# hello-systems

A minimal C program that reveals what happens when you type `./hello` —
argument parsing, environment variables, exit codes, compile-time macros,
and the concept of undefined behaviour.

## How to build

```bash
make          # release build  (-O2 -Wall -Wextra)
make debug    # debug build    (-g -O0 -fsanitize=address)
make strict   # strict build   (-Wall -Wextra -Werror -pedantic)
make asm      # emit src/main.s assembly listing
make clean    # remove all generated files
```

## Running

```bash
./hello
./hello --verbose foo bar
./hello --fail ; echo "Exit status: $?"
```

## Make targets explained

| Target    | Flags                              | Purpose                                       |
|-----------|------------------------------------|-----------------------------------------------|
| `all`     | `-O2 -Wall -Wextra`                | Production build: fast binary, warnings on    |
| `debug`   | `-g -O0 -fsanitize=address`        | Debug build: symbols, no opt, memory checker  |
| `strict`  | `-Wall -Wextra -Werror -pedantic`  | CI build: any warning becomes a compile error |
| `asm`     | `-S -fverbose-asm`                 | Read the actual assembly the compiler emits   |
| `clean`   | —                                  | Delete all build artifacts                    |

## The compilation pipeline

Every `.c` file goes through four stages before you get an executable:

```
main.c
  │
  ▼ Stage 1: PREPROCESSOR  (cpp)
  │   - Expands #include, #define, #ifdef directives
  │   - Produces a single expanded .i file (pure C, no directives)
  │   - Run: gcc -E src/main.c -o main.i  (then open main.i to see it)
  │
  ▼ Stage 2: COMPILER  (cc1)
  │   - Translates C to assembly language (.s file)
  │   - This is where -O2 and -Wall happen
  │   - Run: gcc -S src/main.c -o main.s
  │
  ▼ Stage 3: ASSEMBLER  (as)
  │   - Translates assembly to machine code in an object file (.o)
  │   - Symbols are still unresolved (printf is just a name here)
  │   - Run: gcc -c src/main.c -o main.o
  │
  ▼ Stage 4: LINKER  (ld)
      - Combines .o files + libraries → final executable
      - Resolves all symbol references (matches printf call → libc code)
      - Run: gcc main.o -o hello
```

You can inspect any intermediate stage:
```bash
gcc -E  src/main.c -o main.i   # see preprocessed output
gcc -S  src/main.c -o main.s   # see assembly
gcc -c  src/main.c -o main.o   # see object file (use objdump -d main.o)
objdump -d hello               # disassemble the final binary
```
