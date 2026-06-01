# LEARNINGS — unix-shell

## fork() and exec() — why two syscalls?

The Unix designers could have provided a single `spawn()` call that creates a
process running a new program.  Instead they split it into two:

```
fork()   — clone the current process
exec()   — replace the current process image with a new program
```

The genius of this split is that code between `fork()` and `exec()` runs in
the child but not the parent.  This window is where the shell:
- sets up redirections (`dup2`)
- sets up pipe connections (`dup2`)
- sets environment variables
- changes the process group

All of these changes are made to the child's copies of the OS resources, so
the parent (the shell) is unaffected.  A monolithic `spawn()` would require
passing all of this configuration as arguments, which becomes unwieldy.

### What fork() actually copies

`fork()` creates a child that is a near-exact copy of the parent:
- Same code (text segment)
- Same data, stack, heap (copy-on-write — not actually copied until written)
- Same open file descriptors (shared underlying file descriptions)
- Same signal handlers
- **Different** PID, different parent PID

The shared file descriptors are key: a pipe created before `fork()` is
visible to both parent and child, enabling communication.

## File descriptors and dup2()

Every open file, socket, or pipe is identified by a small integer called a
file descriptor.  The kernel maintains a per-process table mapping fd numbers
to underlying "file descriptions" (offset, mode, reference count).

```
Process fd table:
  0 → terminal (stdin)
  1 → terminal (stdout)
  2 → terminal (stderr)
  3 → open file "data.txt"
```

`dup2(old, new)` makes `new` point to the same underlying file as `old`,
closing `new` first if it was open.

```c
int fd = open("output.txt", O_WRONLY|O_CREAT|O_TRUNC, 0644);
dup2(fd, 1);   // fd 1 (stdout) now writes to output.txt
close(fd);     // fd 3 is no longer needed
// Now: printf() and write(1,...) go to the file
```

## Pipes

`pipe(fds)` creates a kernel buffer with two ends:
- `fds[0]` — read end
- `fds[1]` — write end

Data written to `fds[1]` can be read from `fds[0]`.  The kernel buffer is
typically 64 KB.  Writes block when full; reads block when empty.

For `cmd1 | cmd2`:

```
shell:  pipe(fds)   → fds[0]=3, fds[1]=4

fork child1 (cmd1):
  dup2(fds[1], 1)   → stdout → pipe write end
  close(fds[0])
  close(fds[1])
  exec cmd1

fork child2 (cmd2):
  dup2(fds[0], 0)   → stdin  → pipe read end
  close(fds[0])
  close(fds[1])
  exec cmd2

parent:
  close(fds[0])     ← CRITICAL: see below
  close(fds[1])     ← CRITICAL
  waitpid(child1)
  waitpid(child2)
```

**Why must the parent close its pipe ends?**

The pipe's write end (`fds[1]`) has a reference count of 2 after the fork:
one in the parent, one in child1.  `cmd2` reads from the pipe until it sees
EOF.  EOF is only signaled when ALL write-end references are closed.  If the
parent keeps `fds[1]` open, `cmd2` hangs forever waiting for more input even
after `cmd1` exits.

## Signals

Signals are asynchronous notifications sent to processes.  Common ones:

| Signal | Number | Default action | Cause |
|--------|--------|----------------|-------|
| SIGINT | 2 | Terminate | Ctrl+C |
| SIGCHLD | 17 | Ignore | Child state change |
| SIGTERM | 15 | Terminate | `kill PID` |
| SIGKILL | 9 | Terminate (unblockable) | `kill -9 PID` |

### SIGINT and the foreground child

When you press Ctrl+C, the terminal sends SIGINT to the *foreground process
group*.  By default both the shell and its children are in the same group.

Our shell installs a handler that forwards SIGINT to the foreground child's
PID.  A more complete shell would put each child in its own process group
(`setpgid`), so SIGINT automatically reaches only that group.

### SIGCHLD and zombie prevention

When a child exits, it becomes a "zombie": its PID and exit status are
preserved in the kernel until the parent calls `waitpid`.  If the parent
never does, the zombie persists until the parent exits.  For background jobs,
we install a SIGCHLD handler that calls `waitpid(-1, ..., WNOHANG)` to reap
all newly-terminated children.

## Background jobs

The `&` suffix tells the shell NOT to call `waitpid` on the child.  The child
runs concurrently with the shell's read-eval-print loop.  When it eventually
exits, the SIGCHLD handler reaps it.

A production shell (bash, zsh) also implements job control: `jobs`, `fg`,
`bg`, suspend with Ctrl+Z (SIGTSTP), etc.  We omit these for brevity.
