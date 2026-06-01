# LEARNINGS — file-system-toy

## Inodes: separating metadata from data

An **inode** (index node) is the core data structure of Unix filesystems.
Every file and directory has exactly one inode.  The inode stores:

- File type (regular file, directory, symlink, device, ...)
- Size in bytes
- Timestamps (created, modified, accessed)
- Ownership (uid, gid) — we omit for simplicity
- Permissions — we omit for simplicity
- **Block pointers**: which disk blocks hold the data

Critically, the inode does **not** store the filename.  That mapping lives in
directory entries.

### Why separate metadata from data?

1. **Hard links**: multiple filenames can point to the same inode.
   `ls -i` shows inode numbers.  `ln file1 file2` creates two names for one
   inode.  The inode's "link count" tracks how many directory entries point to
   it; it is freed only when the count reaches zero.

2. **Rename is O(1)**: `rename("old", "new")` updates only the directory entry
   (from "old" → inode to "new" → inode).  No data blocks are touched.

3. **Fast metadata queries**: `ls -la` can display name, size, and timestamps
   without reading any data blocks.  The inode is small and cached.

4. **Directory entries are just mappings**: a directory block contains
   `(name, inode_number)` pairs.  The filesystem code to traverse paths is
   generic — it doesn't know about file types until it reads the inode.

## Block allocation with a bitmap

We track free/used blocks with a **bitmap**: an array of bits where bit `i`
represents block `i`.  0 = free, 1 = used.

```
blocks:   0  1  2  3  4  5  6  7
bitmap:   0  0  1  1  0  1  0  0
               ↑  ↑     ↑
             used     used
```

To allocate a block: scan until we find a 0 bit, set it to 1.
To free a block: clear the bit.

For a 1024-block filesystem the bitmap is only 1024/8 = 128 bytes — tiny
enough to keep in the superblock.

Real filesystems (ext4) use **block group descriptors**: the disk is divided
into groups, each with its own superblock copy, block bitmap, inode bitmap,
and inode table.  This locality means allocating a new block for a file is
likely to land near existing blocks for the same file → fewer disk seeks.

## Direct vs indirect block pointers

Our inodes store up to `MAX_DIRECT_BLOCKS = 8` data block indices directly.
At 512 bytes/block that's 8 × 512 = 4 KB maximum file size.

Real filesystems extend this with indirect pointers:

```
Direct blocks    (12 pointers × 4 KB = 48 KB)
Single indirect  (1 pointer → block of 1024 pointers → 4 MB)
Double indirect  (1 pointer → block → blocks of pointers → 4 GB)
Triple indirect  (→ 4 TB)
```

ext4 (and btrfs, XFS) also support **extents**: instead of listing individual
block numbers, store (start_block, length) pairs.  A 1 GB sequential file
might need only a single extent, not millions of block pointers.

## The superblock

The superblock is the "table of contents" of the filesystem.  It must be read
before any other operation.  It contains:

- Magic number — identifies this as our format (prevents mounting the wrong image)
- Total and free block counts
- Inode count
- Block bitmap (for small filesystems)

ext4 stores multiple superblock copies at well-known offsets (block groups 0,
1, 3, 5, 7, ...) so that if the primary superblock is corrupted, `fsck` can
recover from a backup.

## What `fsck` does

`fsck` (filesystem check) walks the inode table and directory tree, verifying:
- Every inode's block pointers point to in-use blocks
- Every in-use block is referenced by exactly one inode
- Directory entries point to valid inodes
- Link counts match actual directory references

Inconsistencies arise from crashes between metadata updates (e.g., a block is
allocated but the inode is not updated).  Journaling filesystems (ext4, APFS,
NTFS) prevent this by writing changes to a log first.

## Path resolution

`resolve_path(fs, "/home/user/notes.txt")` works like this:

1. Start at inode 0 (root directory).
2. Look up "home" in root's directory entries → inode N.
3. Look up "user" in inode N's directory entries → inode M.
4. Look up "notes.txt" in inode M's directory entries → inode K.
5. Return K.

Each directory lookup requires reading the directory's data blocks.  Real
filesystems cache frequently-used directory entries in the kernel's **dentry
cache** (dcache) so repeated `open("/usr/bin/python3")` calls are fast.
