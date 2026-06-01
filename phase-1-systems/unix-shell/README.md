# unix-shell

A complete interactive Unix shell written in C, covering the full fork/exec
lifecycle, pipes, I/O redirection, signal handling, and background jobs.

## Features

| Feature | Example |
|---------|---------|
| Run any program | `ls -la` |
| Built-in: `cd` | `cd /tmp` |
| Built-in: `pwd` | `pwd` |
| Built-in: `history` | `history` |
| Built-in: `exit` | `exit 0` |
| Background jobs | `sleep 10 &` |
| Pipes (N stages) | `ls \| grep .c \| wc -l` |
| Output redirect | `ls > files.txt` |
| Append redirect | `echo hi >> log.txt` |
| Input redirect | `wc < file.txt` |
| Quoted args | `echo "hello world"` |
| Ctrl+C kills child | (shell stays alive) |

## How to build

```bash
make all
```

## How to run

```bash
./bin/shell
```

Then type commands at the `shell> ` prompt.  Press `Ctrl+D` (EOF) or type
`exit` to quit.

## Files

| File | Purpose |
|------|---------|
| `src/shell.c` | Complete shell implementation |
| `Makefile` | Build system |
| `README.md` | This file |
| `LEARNINGS.md` | Deep explanations of fork, exec, pipes, signals |
