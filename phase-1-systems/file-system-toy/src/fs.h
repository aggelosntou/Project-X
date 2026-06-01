/*
 * fs.h — A toy filesystem implemented on top of a flat disk image file.
 *
 * ── WHAT IS A FILESYSTEM? ─────────────────────────────────────────────────
 *
 * A filesystem is a data structure stored on a block device (disk, SSD, file)
 * that organises raw bytes into named files and directories.  It provides:
 *   - A namespace (paths like /home/user/notes.txt)
 *   - Metadata (size, timestamps, permissions)
 *   - Efficient allocation of storage blocks
 *
 * ── OUR LAYOUT ────────────────────────────────────────────────────────────
 *
 *  Block 0: Superblock (filesystem metadata)
 *  Block 1: Inode table (MAX_INODES × sizeof(Inode))
 *  Blocks 2+: Data blocks
 *
 *  Total image size = MAX_BLOCKS × BLOCK_SIZE bytes
 *
 * ── WHAT IS AN INODE? ─────────────────────────────────────────────────────
 *
 * An inode (index node) is a fixed-size struct holding a file's METADATA:
 *   - type (file or directory)
 *   - size in bytes
 *   - block numbers where the data lives
 *   - timestamps
 *
 * Critically, inodes do NOT store the filename.  Filenames live in directory
 * entries, which map name → inode number.  This separation enables:
 *   - Hard links: multiple names pointing to the same inode
 *   - Renaming: just update the directory entry, not the inode
 *   - Efficient metadata access: the inode is small, no need to read data
 *
 * ── BLOCK ALLOCATION ──────────────────────────────────────────────────────
 *
 * We use a BITMAP to track which blocks are free.  Each bit corresponds to
 * one data block.  Finding a free block = finding the first 0 bit.
 * This is O(MAX_BLOCKS/8) in the worst case.
 *
 * ext4, NTFS, btrfs use more sophisticated structures (b-trees, extent trees)
 * for large filesystems.
 */

#ifndef FS_H
#define FS_H

#include <stdint.h>
#include <time.h>

/* ── Constants ───────────────────────────────────────────────────────────── */

#define BLOCK_SIZE        512
#define MAX_BLOCKS        1024
#define MAX_INODES        128
#define MAX_DIRECT_BLOCKS   8
#define FS_MAGIC       0xDEADC0DE   /* magic number identifies our format */

/* ── Superblock ──────────────────────────────────────────────────────────── */

/*
 * The superblock is the first thing read when mounting a filesystem.
 * It describes the geometry and current state of the whole volume.
 * Real filesystems (ext4, FAT32) also store copies of the superblock
 * in multiple locations as a backup against corruption.
 */
typedef struct {
    uint32_t magic;                      /* must equal FS_MAGIC */
    uint32_t total_blocks;               /* total data blocks */
    uint32_t free_blocks;                /* number of free data blocks */
    uint8_t  block_bitmap[MAX_BLOCKS/8]; /* 1 bit per block: 0=free, 1=used */
    uint32_t inode_count;                /* number of allocated inodes */
} Superblock;

/* ── Inode types ─────────────────────────────────────────────────────────── */

typedef enum {
    INODE_FREE = 0,   /* unused inode slot */
    INODE_FILE = 1,   /* regular file */
    INODE_DIR  = 2    /* directory */
} InodeType;

/* ── Inode ───────────────────────────────────────────────────────────────── */

typedef struct {
    InodeType type;
    uint32_t  size;                       /* bytes of content */
    uint32_t  blocks[MAX_DIRECT_BLOCKS];  /* data block indices (0 = unused) */
    time_t    created;
    time_t    modified;
} Inode;

/* ── Directory entry ─────────────────────────────────────────────────────── */

/*
 * Each directory is stored as an array of DirEntry structs.
 * A directory with N entries occupies ceil(N * sizeof(DirEntry) / BLOCK_SIZE)
 * data blocks.
 *
 * size is chosen so sizeof(DirEntry) == 32, giving 16 entries per 512-byte block.
 */
typedef struct {
    uint32_t inode_num;   /* 0 = empty/deleted slot */
    char     name[28];    /* filename (null-terminated) — 32 bytes total */
} DirEntry;

/* ── FS handle ────────────────────────────────────────────────────────────── */

/*
 * FS is an in-memory handle to a mounted filesystem.
 * All operations go through this pointer.
 */
typedef struct {
    int        fd;           /* file descriptor for the disk image */
    Superblock sb;           /* cached superblock */
    Inode      inodes[MAX_INODES];  /* cached inode table */
    int        dirty;        /* 1 if in-memory state differs from disk */
} FS;

/* ── Public API ─────────────────────────────────────────────────────────── */

/** fs_format — create a new, empty filesystem image at `filename`. */
int  fs_format(const char *filename, uint32_t total_blocks);

/** fs_mount — open an existing filesystem image.  Returns NULL on error. */
FS  *fs_mount(const char *filename);

/** fs_unmount — flush changes to disk and free the FS handle. */
void fs_unmount(FS *fs);

/** fs_mkdir — create a directory.  Parent must exist. */
int  fs_mkdir(FS *fs, const char *path);

/** fs_create — create an empty file.  Parent directory must exist. */
int  fs_create(FS *fs, const char *path);

/** fs_write — write `len` bytes from `data` to file at `path`. */
int  fs_write(FS *fs, const char *path, const void *data, uint32_t len);

/** fs_read — read up to `maxlen` bytes from file at `path` into `buf`. */
int  fs_read(FS *fs, const char *path, void *buf, uint32_t maxlen);

/** fs_ls — list the contents of directory `path`. */
int  fs_ls(FS *fs, const char *path);

/** fs_stat — print inode metadata for `path`. */
int  fs_stat(FS *fs, const char *path);

/** fs_rm — remove a file (not a directory). */
int  fs_rm(FS *fs, const char *path);

#endif /* FS_H */
