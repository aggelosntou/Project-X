# string-library

A complete reimplementation of `<string.h>` from scratch, plus a dynamic
string type (stretchy buffer).  Every function is written in ~10-30 lines
of clear, commented C so you can see exactly how they work.

## Functions implemented

### String functions
| Function | What it does |
|----------|--------------|
| `my_strlen` | Count bytes until `\0` |
| `my_strcpy` | Copy string including terminator |
| `my_strncpy` | Bounded copy (padded with `\0`) |
| `my_strcat` | Append src to dst |
| `my_strcmp` | Lexicographic compare |
| `my_strncmp` | Bounded compare |
| `my_strchr` | First occurrence of char |
| `my_strrchr` | Last occurrence of char |
| `my_strstr` | Find substring |

### Memory functions
| Function | What it does |
|----------|--------------|
| `my_memcpy` | Copy n bytes (no overlap) |
| `my_memmove` | Copy n bytes (overlap-safe) |
| `my_memset` | Fill n bytes with value |
| `my_memcmp` | Compare n bytes |

### Dynamic string
| Function | What it does |
|----------|--------------|
| `dynstr_new()` | Create empty dynamic string |
| `dynstr_append_char()` | Append one character (grows if needed) |
| `dynstr_append_str()` | Append a C string (grows if needed) |
| `dynstr_free()` | Release memory |

## Build and test

```bash
make          # build libmystring.a
make test     # build and run test suite (with AddressSanitizer)
make clean
```

All tests are in `src/test_mystring.c` and use `assert()`.
If the program exits 0, all tests passed.

## File layout

```
src/
  mystring.h        — declarations (include this in your code)
  mystring.c        — implementations
  test_mystring.c   — test suite
```
