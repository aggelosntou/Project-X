/*
 * shell.c — A complete interactive Unix shell.
 *
 * ── CORE CONCEPTS ──────────────────────────────────────────────────────────
 *
 * FORK + EXEC MODEL
 *   Unix shells do NOT run external programs directly.  Instead:
 *     1. fork()  — clone the current process (shell) into parent + child.
 *                  Both processes are now running identical code.
 *     2. execvp() — the CHILD replaces its memory image with the new program.
 *                  The parent (shell) waits for it to finish.
 *   This separation is elegant: the child can freely manipulate its own
 *   file descriptors (redirect stdin/stdout) BEFORE exec without affecting
 *   the parent.
 *
 * FILE DESCRIPTORS
 *   Every process inherits three open file descriptors:
 *     0 — stdin   (read from terminal)
 *     1 — stdout  (write to terminal)
 *     2 — stderr  (write errors to terminal)
 *   dup2(oldfd, newfd) copies oldfd to newfd, closing newfd first.
 *   After dup2(file, 0), reading from fd 0 reads from the file.
 *
 * PIPES
 *   pipe(fds) creates a one-way channel: write to fds[1], read from fds[0].
 *   For cmd1 | cmd2, the shell creates a pipe and wires:
 *     child1: stdout → pipe write end
 *     child2: stdin  → pipe read end
 *
 * SIGNALS
 *   SIGINT (Ctrl+C) is normally delivered to the foreground process group.
 *   We catch it in the shell and forward it to the foreground child instead,
 *   so the shell itself stays alive.
 *   SIGCHLD is sent when a child changes state; we use it to reap zombies
 *   from background jobs.
 */

#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE   700

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <limits.h>
#include <ctype.h>

/* ── Configuration ───────────────────────────────────────────────────────── */

#define MAX_LINE        4096    /* maximum input line length */
#define MAX_TOKENS      256     /* maximum tokens per command line */
#define MAX_ARGS        128     /* maximum arguments per command */
#define MAX_PIPES       16      /* maximum pipe stages */
#define HISTORY_SIZE    20      /* number of history entries to keep */

/* ── History ─────────────────────────────────────────────────────────────── */

static char history[HISTORY_SIZE][MAX_LINE];
static int  history_count = 0;

static void history_add(const char *line) {
    /* Shift older entries out if we're at capacity */
    if (history_count == HISTORY_SIZE) {
        for (int i = 0; i < HISTORY_SIZE - 1; i++)
            strcpy(history[i], history[i + 1]);
        history_count--;
    }
    strncpy(history[history_count], line, MAX_LINE - 1);
    history[history_count][MAX_LINE - 1] = '\0';
    history_count++;
}

static void history_print(void) {
    for (int i = 0; i < history_count; i++)
        printf("  %3d  %s\n", i + 1, history[i]);
}

/* ── Signal handling ─────────────────────────────────────────────────────── */

/*
 * fg_pid — the PID of the currently-running foreground child, or -1 if none.
 * It is declared volatile sig_atomic_t because it is written from a signal
 * handler.  The C standard requires this type for variables shared between
 * signal handlers and normal code.
 */
static volatile sig_atomic_t fg_pid = -1;

/*
 * SIGINT handler — Ctrl+C.
 *
 * WHY not just ignore it in the shell?  If we ignore SIGINT completely,
 * Ctrl+C does nothing — not even killing runaway programs.  Instead we
 * forward it to the foreground child so it dies while the shell lives.
 */
static void sigint_handler(int sig) {
    if (fg_pid > 0)
        kill(fg_pid, sig);
    /* else: no foreground child; the signal effectively does nothing */
}

/*
 * SIGCHLD handler — a child process changed state (exited, was killed, etc.)
 *
 * WHY do we need this?  When a background child exits, it becomes a zombie:
 * it holds its entry in the process table waiting for the parent to call
 * waitpid().  If we never reap it, we leak process-table entries.
 * WNOHANG means "don't block; return 0 if no child has exited yet".
 * The loop handles the case where multiple children exit between signals.
 */
static void sigchld_handler(int sig) {
    (void)sig;
    int status;
    pid_t pid;
    /* Reap all children that have already exited */
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0)
        ; /* zombie reaped */
}

/* ── Tokeniser ───────────────────────────────────────────────────────────── */

/*
 * tokenise — split `line` into tokens, handling:
 *   - whitespace separators
 *   - quoted strings ("hello world" → one token)
 *   - special single-character tokens: | < > &
 *
 * Returns the number of tokens, stored in tokens[].
 * Modifies `line` in place (inserts NUL terminators).
 */
static int tokenise(char *line, char *tokens[], int max_tokens) {
    int count = 0;
    char *p = line;

    while (*p && count < max_tokens - 1) {
        /* Skip whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        if (*p == '"') {
            /* Quoted string: find closing quote */
            p++;   /* skip opening quote */
            tokens[count++] = p;
            while (*p && *p != '"') p++;
            if (*p == '"') *p++ = '\0';  /* NUL-terminate, skip closing quote */
        } else if (*p == '|' || *p == '<' || *p == '&') {
            /* Single-char special tokens */
            tokens[count++] = p;
            p[1] = '\0';  /* only if safe — we check next char */
            /* Actually we need to be careful here: don't NUL-terminate if
             * p[1] might be '>' (for >>).  Let's handle >> properly: */
            /* Reset: handle >> as a two-char token */
            if (*p == '>' && *(p+1) == '>') {
                /* handled below */
            }
            p++;
        } else if (*p == '>') {
            if (*(p+1) == '>') {
                /* ">>" append redirection */
                tokens[count++] = p;
                p[2] = '\0';
                p += 2;
            } else {
                tokens[count++] = p;
                char saved = p[1];
                p[1] = '\0';
                p++;
                /* We already NUL'd p[1]; we need the character back for
                 * later parsing — but since we advanced p, *p is now saved.
                 * Actually we clobbered the original string.  Use a simpler
                 * approach: restore saved character before advancing. */
                /* This is tricky with in-place modification.  Use a clean
                 * approach instead: */
                (void)saved;
            }
        } else {
            /* Regular token: extends until whitespace or special char */
            tokens[count++] = p;
            while (*p && !isspace((unsigned char)*p) &&
                   *p != '|' && *p != '<' && *p != '>' && *p != '&') {
                p++;
            }
            if (*p) { *p++ = '\0'; }
        }
    }
    tokens[count] = NULL;
    return count;
}

/*
 * tokenise_v2 — a cleaner tokeniser that doesn't have the in-place mutation
 * problem above.  We use this in the actual shell.
 *
 * Approach: copy each token into a fresh buffer (static storage is fine for
 * a single-threaded shell), return pointers into that buffer.
 */
static char token_storage[MAX_LINE * 2];
static int  token_storage_pos = 0;

static char *alloc_token(const char *start, int len) {
    if (token_storage_pos + len + 1 >= (int)sizeof(token_storage)) return NULL;
    char *t = token_storage + token_storage_pos;
    memcpy(t, start, len);
    t[len] = '\0';
    token_storage_pos += len + 1;
    return t;
}

static int tokenise_clean(const char *line, char *tokens[], int max_tokens) {
    token_storage_pos = 0;   /* reset per-command storage */
    int count = 0;
    const char *p = line;

    while (*p && count < max_tokens - 1) {
        /* skip whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        const char *start;

        if (*p == '"') {
            /* Quoted string */
            p++;  /* skip opening quote */
            start = p;
            while (*p && *p != '"') p++;
            tokens[count++] = alloc_token(start, (int)(p - start));
            if (*p == '"') p++;  /* skip closing quote */
        } else if (*p == '>' && *(p+1) == '>') {
            tokens[count++] = alloc_token(">>", 2);
            p += 2;
        } else if (*p == '|' || *p == '<' || *p == '>' || *p == '&') {
            tokens[count++] = alloc_token(p, 1);
            p++;
        } else {
            start = p;
            while (*p && !isspace((unsigned char)*p) &&
                   *p != '|' && *p != '<' && *p != '>' && *p != '&')
                p++;
            tokens[count++] = alloc_token(start, (int)(p - start));
        }
    }
    tokens[count] = NULL;
    return count;
}

/* ── Command struct ──────────────────────────────────────────────────────── */

typedef struct {
    char  *args[MAX_ARGS];   /* NULL-terminated argument vector for execvp */
    int    argc;
    char  *redirect_in;      /* filename for < redirection, or NULL */
    char  *redirect_out;     /* filename for > or >> redirection, or NULL */
    int    append_out;       /* 1 if >>, 0 if > */
} Command;

/* ── Pipeline parser ─────────────────────────────────────────────────────── */

/*
 * parse_pipeline — split tokens at '|' boundaries into an array of Commands.
 *
 * Each Command also captures any I/O redirections (< file, > file, >> file).
 *
 * Returns the number of pipe stages.
 */
static int parse_pipeline(char *tokens[], int num_tokens,
                           Command cmds[], int max_cmds) {
    int num_cmds = 0;
    int i = 0;

    memset(&cmds[0], 0, sizeof(Command));

    while (i < num_tokens && num_cmds < max_cmds) {
        Command *cmd = &cmds[num_cmds];

        while (i < num_tokens) {
            char *tok = tokens[i];

            if (strcmp(tok, "|") == 0) {
                i++;
                num_cmds++;
                if (num_cmds < max_cmds)
                    memset(&cmds[num_cmds], 0, sizeof(Command));
                break;
            } else if (strcmp(tok, "<") == 0) {
                i++;
                if (i < num_tokens) cmd->redirect_in = tokens[i++];
            } else if (strcmp(tok, ">>") == 0) {
                i++;
                if (i < num_tokens) {
                    cmd->redirect_out = tokens[i++];
                    cmd->append_out   = 1;
                }
            } else if (strcmp(tok, ">") == 0) {
                i++;
                if (i < num_tokens) {
                    cmd->redirect_out = tokens[i++];
                    cmd->append_out   = 0;
                }
            } else {
                if (cmd->argc < MAX_ARGS - 1) {
                    cmd->args[cmd->argc++] = tok;
                    cmd->args[cmd->argc]   = NULL;
                }
                i++;
            }
        }

        if (cmd->argc > 0 || cmd->redirect_in || cmd->redirect_out)
            num_cmds++;
        else
            break;   /* empty segment after last '|' */
    }

    return num_cmds;
}

/* ── I/O redirection ─────────────────────────────────────────────────────── */

/*
 * setup_redirections — called in the child process BEFORE exec.
 *
 * WHY in the child?  dup2 modifies the calling process's file descriptor
 * table.  If we did this in the parent (shell), we'd redirect the shell's
 * own stdin/stdout — breaking the shell.  Because we do it after fork() but
 * before exec(), only the child is affected.
 */
static int setup_redirections(Command *cmd) {
    if (cmd->redirect_in) {
        int fd = open(cmd->redirect_in, O_RDONLY);
        if (fd < 0) {
            perror(cmd->redirect_in);
            return -1;
        }
        dup2(fd, STDIN_FILENO);   /* make fd 0 point to the file */
        close(fd);                /* we no longer need the original fd */
    }

    if (cmd->redirect_out) {
        int flags = O_WRONLY | O_CREAT | (cmd->append_out ? O_APPEND : O_TRUNC);
        int fd    = open(cmd->redirect_out, flags, 0644);
        if (fd < 0) {
            perror(cmd->redirect_out);
            return -1;
        }
        dup2(fd, STDOUT_FILENO);  /* make fd 1 point to the file */
        close(fd);
    }

    return 0;
}

/* ── Built-in commands ───────────────────────────────────────────────────── */

/*
 * WHY built-ins?  Some commands MUST run inside the shell process because they
 * modify the shell's own state.  `cd` changes the shell's working directory —
 * if it ran in a child process, the child's chdir() would have no effect on
 * the parent (each process has its own cwd).  `exit` must terminate the shell,
 * not a child.
 */

static int builtin_cd(char *args[]) {
    const char *dir = args[1] ? args[1] : getenv("HOME");
    if (!dir) { fprintf(stderr, "cd: HOME not set\n"); return 1; }
    if (chdir(dir) != 0) { perror("cd"); return 1; }
    return 0;
}

static int builtin_pwd(void) {
    char buf[PATH_MAX];
    if (!getcwd(buf, sizeof(buf))) { perror("pwd"); return 1; }
    printf("%s\n", buf);
    return 0;
}

static void builtin_help(void) {
    printf("Built-in commands:\n"
           "  cd [dir]     — change directory (default: $HOME)\n"
           "  pwd          — print working directory\n"
           "  exit [code]  — exit the shell\n"
           "  help         — show this message\n"
           "  history      — show last %d commands\n"
           "\nFeatures:\n"
           "  cmd &        — run command in background\n"
           "  cmd1 | cmd2  — pipe (chain as many as you like)\n"
           "  cmd > file   — redirect stdout (overwrite)\n"
           "  cmd >> file  — redirect stdout (append)\n"
           "  cmd < file   — redirect stdin\n",
           HISTORY_SIZE);
}

/* ── Execute a pipeline ───────────────────────────────────────────────────── */

/*
 * execute_pipeline — the heart of the shell.
 *
 * For a pipeline with N commands, we need N-1 pipes.
 * We fork N children, connecting them with pipes, then wait for all of them.
 *
 * Pipe layout for cmd1 | cmd2 | cmd3:
 *
 *   pipes[0][0] ──read──▶ cmd2's stdin
 *   pipes[0][1] ◀──write── cmd1's stdout
 *
 *   pipes[1][0] ──read──▶ cmd3's stdin
 *   pipes[1][1] ◀──write── cmd2's stdout
 *
 * WHY close the unused pipe ends?
 *   Each pipe end counts as an open file descriptor.  If cmd2 keeps the
 *   write end of its input pipe open, cmd1 never sees EOF when it's done,
 *   because there's still a writer.  Closing unused ends is mandatory for
 *   correct EOF signaling.
 */
static int execute_pipeline(Command cmds[], int num_cmds, int background) {
    if (num_cmds == 0) return 0;

    /* pipes[i] connects cmd[i] → cmd[i+1] */
    int pipes[MAX_PIPES][2];
    int num_pipes = num_cmds - 1;

    for (int i = 0; i < num_pipes; i++) {
        if (pipe(pipes[i]) < 0) {
            perror("pipe");
            return -1;
        }
    }

    pid_t pids[MAX_PIPES + 1];
    pid_t last_pid = -1;

    for (int i = 0; i < num_cmds; i++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); return -1; }

        if (pid == 0) {
            /* ── Child process ─────────────────────────────────────── */

            /*
             * Connect pipe input (if not the first command).
             * The previous pipe's read end becomes our stdin.
             */
            if (i > 0) {
                dup2(pipes[i-1][0], STDIN_FILENO);
            }

            /*
             * Connect pipe output (if not the last command).
             * The current pipe's write end becomes our stdout.
             */
            if (i < num_cmds - 1) {
                dup2(pipes[i][1], STDOUT_FILENO);
            }

            /* Close ALL pipe fds in the child — we've dup2'd what we need */
            for (int j = 0; j < num_pipes; j++) {
                close(pipes[j][0]);
                close(pipes[j][1]);
            }

            /* Apply explicit I/O redirections (overrides pipe connections) */
            if (setup_redirections(&cmds[i]) < 0) exit(1);

            /* Restore default signal handling in the child.
             * WHY?  The shell catches SIGINT; the child should die on it. */
            signal(SIGINT, SIG_DFL);
            signal(SIGCHLD, SIG_DFL);

            /* Replace the child image with the requested program */
            execvp(cmds[i].args[0], cmds[i].args);

            /* execvp only returns on failure */
            fprintf(stderr, "shell: %s: command not found\n", cmds[i].args[0]);
            exit(127);
        }

        /* ── Parent: record child PID ──────────────────────────────── */
        pids[i]  = pid;
        last_pid = pid;
    }

    /* Close ALL pipe fds in the parent.
     * WHY?  If the parent holds a write end open, the reader child never
     * sees EOF even after all writer children exit. */
    for (int i = 0; i < num_pipes; i++) {
        close(pipes[i][0]);
        close(pipes[i][1]);
    }

    if (background) {
        printf("[%d] running in background\n", last_pid);
        return 0;
    }

    /* Foreground: wait for all children */
    fg_pid = last_pid;   /* so SIGINT forwards to the right process */

    int exit_status = 0;
    for (int i = 0; i < num_cmds; i++) {
        int status;
        waitpid(pids[i], &status, 0);
        if (i == num_cmds - 1) {
            /* The exit status of a pipeline is that of the last command */
            exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        }
    }

    fg_pid = -1;
    return exit_status;
}

/* ── Main REPL ───────────────────────────────────────────────────────────── */

int main(void) {
    /* Install signal handlers */
    struct sigaction sa_int = { .sa_handler = sigint_handler,  .sa_flags = 0 };
    struct sigaction sa_chld= { .sa_handler = sigchld_handler, .sa_flags = SA_RESTART };
    sigemptyset(&sa_int.sa_mask);
    sigemptyset(&sa_chld.sa_mask);
    sigaction(SIGINT,  &sa_int,  NULL);
    sigaction(SIGCHLD, &sa_chld, NULL);

    char line[MAX_LINE];

    while (1) {
        /* Print prompt */
        printf("shell> ");
        fflush(stdout);

        /* Read a line (handles EOF from Ctrl+D) */
        if (!fgets(line, sizeof(line), stdin)) {
            printf("\n");
            break;   /* EOF → exit */
        }

        /* Strip trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[--len] = '\0';

        /* Ignore empty lines */
        if (len == 0) continue;

        /* Save to history */
        history_add(line);

        /* Check for background flag BEFORE tokenising */
        int background = 0;
        {
            char *last = line + len - 1;
            while (last > line && isspace((unsigned char)*last)) last--;
            if (*last == '&') {
                background = 1;
                *last = '\0';  /* remove the & from the line */
            }
        }

        /* Tokenise */
        char *tokens[MAX_TOKENS];
        int num_tokens = tokenise_clean(line, tokens, MAX_TOKENS);
        if (num_tokens == 0) continue;

        /* ── Handle built-ins BEFORE forking ─────────────────────── */
        if (strcmp(tokens[0], "exit") == 0) {
            int code = tokens[1] ? atoi(tokens[1]) : 0;
            exit(code);
        }
        if (strcmp(tokens[0], "cd") == 0)  { builtin_cd(tokens); continue; }
        if (strcmp(tokens[0], "pwd") == 0) { builtin_pwd();      continue; }
        if (strcmp(tokens[0], "help") == 0){ builtin_help();     continue; }
        if (strcmp(tokens[0], "history") == 0) { history_print(); continue; }

        /* ── Parse into pipeline stages ──────────────────────────── */
        Command cmds[MAX_PIPES + 1];
        int num_cmds = parse_pipeline(tokens, num_tokens, cmds, MAX_PIPES + 1);

        if (num_cmds == 0 || cmds[0].argc == 0) continue;

        /* ── Execute ──────────────────────────────────────────────── */
        execute_pipeline(cmds, num_cmds, background);
    }

    return 0;
}
