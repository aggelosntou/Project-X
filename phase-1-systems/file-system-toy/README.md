# file-system-toy

A toy filesystem stored in a single flat file (disk image), implementing
superblock, bitmap-based block allocation, an inode table, and directory
entries — the core concepts behind ext4, FAT32, and APFS.

## How to build

```bash
make all
```

## How to run

```bash
./bin/fs myfs.img
```

If `myfs.img` doesn't exist it is formatted automatically.

## Interactive commands

```
fs> mkdir /home
fs> mkdir /home/user
fs> create /home/user/notes.txt
fs> write /home/user/notes.txt Hello from the toy FS!
fs> read /home/user/notes.txt
fs> ls /home/user
fs> stat /home/user/notes.txt
fs> rm /home/user/notes.txt
fs> ls /home/user
fs> exit
```

## Disk image layout

```
Offset 0         : Superblock (magic, block count, bitmap)
Offset 512       : Inode table (MAX_INODES × sizeof(Inode))
Offset 512+table : Data blocks (BLOCK_SIZE = 512 bytes each)
```

## Files

| File | Purpose |
|------|---------|
| `src/fs.h` | Structs and API declaration |
| `src/fs.c` | Full filesystem implementation |
| `src/main.c` | Interactive CLI |
| `Makefile` | Build + smoke test |
| `README.md` | This file |
| `LEARNINGS.md` | Inodes, block allocation, bitmap, why metadata/data split |
