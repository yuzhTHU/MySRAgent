---
name: workspace-shell
description: Use this skill to understand what commands are available in the workspace_shell tool, their syntax, limitations, and security constraints. Read this when you need to manipulate files, run scripts, or explore data in the workspace.
---

# Workspace Shell Tool Reference

## Overview

The `workspace_shell` tool provides a restricted shell environment inside an isolated workspace directory. All operations are sandboxed — you cannot access files outside the workspace.

## Supported Commands

### File Viewing
| Command | Syntax | Description |
|---------|--------|-------------|
| `ls` | `ls [path]` | List directory contents |
| `cat` | `cat <file>` | Print file contents |
| `head` | `head [-n N] <file>` | Print first N lines (default 10) |
| `tail` | `tail [-n N] <file>` | Print last N lines (default 10) |
| `wc` | `wc <file>` | Print line, word, and character counts |

### Text Processing
| Command | Syntax | Description |
|---------|--------|-------------|
| `grep` | `grep <pattern> [file]` | Filter lines containing pattern |
| `sort` | `sort [file]` | Sort lines alphabetically |
| `cut` | `cut -d <delim> -f <fields> [file]` | Extract columns |

### File Operations
| Command | Syntax | Description |
|---------|--------|-------------|
| `cp` | `cp <src> <dst>` | Copy a file or directory |
| `mv` | `mv <src> <dst>` | Move/rename a file |
| `rm` | `rm <file>` | Remove a file or directory |
| `mkdir` | `mkdir <dir>` | Create a directory |

### Compression
| Command | Syntax | Description |
|---------|--------|-------------|
| `gunzip` | `gunzip <file.gz>` | Decompress gzip file |
| `gzip` | `gzip <file>` | Compress file with gzip |
| `unzip` | `unzip <file.zip>` | Extract zip archive |
| `tar` | `tar -xf <file.tar>` or `tar -xzf <file.tar.gz>` | Extract tar archive |


## Pipes

You can chain commands with `|` (pipe). The stdout of the previous command becomes the input of the next:

```
cat data.csv | head -5
cat data.csv | grep "pattern" | wc
```

## Security Constraints

1. **No path traversal:** Paths like `../`, `/etc/passwd`, or any absolute path are rejected.
2. **No escape:** Symbolic links pointing outside the workspace are blocked.
3. **Command whitelist:** Only the commands listed above are allowed. All others are rejected.
4. **Read-only source files:** Files linked from outside the workspace are read-only.
5. **No network access:** Python scripts cannot import networking modules.
6. **No process spawning:** subprocess, os.system, etc. are forbidden in Python scripts.

## Common Patterns

### Explore a compressed dataset
```
ls
gunzip data.csv.gz
head -20 data.csv
wc data.csv
```

### Check specific columns
```
head -1 data.csv
cut -d , -f 1,3 data.csv | head -5
```

## Limitations

- No interactive commands (vim, nano, less, etc.)
- No background processes or job control
- No environment variables or shell expansion ($VAR, *, ?, etc.)
- No redirection operators (>, >>, <) — use Python scripts for file writing
- `grep` does simple string matching, not full regex
- `tar` only supports extraction (-x), not creation (-c)
