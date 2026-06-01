/*
 * fs.c — Implementation of our toy filesystem.
 *
 * All data is stored in a single flat file (the "disk image").
 * We read/write individual blocks using pread/pwrite for simplicity.
 *
 * Layout on disk:
 *   Offset 0               : Superblock (one BLOCK_SIZE sector)
 *   Offset BLOCK_SIZE      : Inode table (MAX_INODES × sizeof(Inode))
 *   Offset INODE_TABLE_END : Data blocks 0 .. total_blocks-1
 *
 * We cache the superblock and entire inode table in memory.
 * On unmount (or any write), we flush them back to disk.
 */

#define _POSIX_C_SOURCE 200809L

#include "fs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>

/* ── Disk layout helpers ─────────────────────────────────────────────────── */

#define SUPERBLOCK_OFFSET   0
/* We pad the superblock and inode table to whole blocks */
#define INODE_TABLE_OFFSET  BLOCK_SIZE
#define INODE_TABLE_SIZE    (MAX_INODES * sizeof(Inode))
/* Data blocks start after superblock block + inode table blocks */
#define DATA_OFFSET         (BLOCK_SIZE + INODE_TABLE_SIZE)

/* Convert a data block index to its byte offset on disk */
static off_t block_offset(uint32_t block_idx) {
    return (off_t)DATA_OFFSET + (off_t)block_idx * BLOCK_SIZE;
}

/* ── Low-level block I/O ──────────────────────────────────────────────────── */

static int write_block(FS *fs, uint32_t idx, const void *data) {
    if (pwrite(fs->fd, data, BLOCK_SIZE, block_offset(idx)) != BLOCK_SIZE) {
        perror("write_block"); return -1;
    }
    return 0;
}

static int read_block(FS *fs, uint32_t idx, void *data) {
    if (pread(fs->fd, data, BLOCK_SIZE, block_offset(idx)) != BLOCK_SIZE) {
        perror("read_block"); return -1;
    }
    return 0;
}

/* ── Flush metadata to disk ──────────────────────────────────────────────── */

static int flush_superblock(FS *fs) {
    if (pwrite(fs->fd, &fs->sb, sizeof(Superblock),
               SUPERBLOCK_OFFSET) != sizeof(Superblock)) {
        perror("flush_superblock"); return -1;
    }
    return 0;
}

static int flush_inodes(FS *fs) {
    if (pwrite(fs->fd, fs->inodes,
               MAX_INODES * sizeof(Inode),
               INODE_TABLE_OFFSET) != (ssize_t)(MAX_INODES * sizeof(Inode))) {
        perror("flush_inodes"); return -1;
    }
    return 0;
}

/* ── Bitmap helpers ──────────────────────────────────────────────────────── */

/*
 * A bitmap is an array of bytes used as a compact array of bits.
 * Bit i corresponds to block i.
 * byte = i/8, bit within byte = i%8
 *
 * This is the same technique used by CPU registers, IP address masks, etc.
 */

static void bitmap_set(uint8_t *bm, uint32_t i) {
    bm[i / 8] |= (1u << (i % 8));
}

static void bitmap_clear(uint8_t *bm, uint32_t i) {
    bm[i / 8] &= ~(1u << (i % 8));
}

static int bitmap_test(const uint8_t *bm, uint32_t i) {
    return (bm[i / 8] >> (i % 8)) & 1;
}

/* ── Block allocation ────────────────────────────────────────────────────── */

/* Allocate a free data block.  Returns block index or -1 if full. */
static int alloc_block(FS *fs) {
    for (uint32_t i = 0; i < fs->sb.total_blocks; i++) {
        if (!bitmap_test(fs->sb.block_bitmap, i)) {
            bitmap_set(fs->sb.block_bitmap, i);
            fs->sb.free_blocks--;
            /* Zero-fill the new block (like real filesystems do for security) */
            uint8_t zeroes[BLOCK_SIZE] = {0};
            write_block(fs, i, zeroes);
            return (int)i;
        }
    }
    return -1;  /* disk full */
}

static void free_block(FS *fs, uint32_t idx) {
    if (bitmap_test(fs->sb.block_bitmap, idx)) {
        bitmap_clear(fs->sb.block_bitmap, idx);
        fs->sb.free_blocks++;
    }
}

/* ── Inode allocation ────────────────────────────────────────────────────── */

/* Allocate a free inode slot.  Returns inode index or -1. */
static int alloc_inode(FS *fs) {
    for (int i = 0; i < MAX_INODES; i++) {
        if (fs->inodes[i].type == INODE_FREE) {
            memset(&fs->inodes[i], 0, sizeof(Inode));
            fs->sb.inode_count++;
            return i;
        }
    }
    return -1;
}

static void free_inode(FS *fs, int idx) {
    fs->inodes[idx].type = INODE_FREE;
    fs->sb.inode_count--;
}

/* ── Path parsing ────────────────────────────────────────────────────────── */

/*
 * split_path — split "/a/b/c" into parent="/a/b" and name="c".
 * Writes into parent_buf and name_buf (caller provides storage).
 */
static void split_path(const char *path,
                        char *parent_buf, int parent_sz,
                        char *name_buf,   int name_sz) {
    const char *last_slash = strrchr(path, '/');
    if (!last_slash || last_slash == path) {
        strncpy(parent_buf, "/", parent_sz - 1);
        parent_buf[parent_sz - 1] = '\0';
        strncpy(name_buf, last_slash ? last_slash + 1 : path, name_sz - 1);
        name_buf[name_sz - 1] = '\0';
    } else {
        int plen = (int)(last_slash - path);
        if (plen >= parent_sz) plen = parent_sz - 1;
        strncpy(parent_buf, path, plen);
        parent_buf[plen] = '\0';
        strncpy(name_buf, last_slash + 1, name_sz - 1);
        name_buf[name_sz - 1] = '\0';
    }
}

/* ── Directory lookup ────────────────────────────────────────────────────── */

/*
 * lookup_in_dir — search directory inode `dir_idx` for an entry named `name`.
 * Returns the inode number of the found entry, or -1 if not found.
 *
 * A directory's data is an array of DirEntry structs packed into blocks.
 */
static int lookup_in_dir(FS *fs, int dir_idx, const char *name) {
    Inode *dir = &fs->inodes[dir_idx];
    if (dir->type != INODE_DIR) return -1;

    uint8_t block[BLOCK_SIZE];
    uint32_t entries_per_block = BLOCK_SIZE / sizeof(DirEntry);

    for (int b = 0; b < MAX_DIRECT_BLOCKS; b++) {
        if (dir->blocks[b] == 0) continue;
        if (read_block(fs, dir->blocks[b], block) < 0) return -1;

        DirEntry *entries = (DirEntry *)block;
        for (uint32_t e = 0; e < entries_per_block; e++) {
            if (entries[e].inode_num != 0 &&
                strncmp(entries[e].name, name, 27) == 0) {
                return (int)entries[e].inode_num;
            }
        }
    }
    return -1;
}

/*
 * resolve_path — walk a path like "/home/user/notes.txt" component by
 * component, starting from inode 0 (root directory).
 * Returns the inode index of the final component, or -1.
 */
static int resolve_path(FS *fs, const char *path) {
    if (strcmp(path, "/") == 0) return 0;  /* inode 0 is root */

    /* Copy path so we can tokenise it */
    char buf[256];
    strncpy(buf, path, sizeof(buf) - 1);
    buf[255] = '\0';

    int current = 0;  /* start at root inode */
    char *token = strtok(buf[0] == '/' ? buf + 1 : buf, "/");

    while (token) {
        int next = lookup_in_dir(fs, current, token);
        if (next < 0) return -1;  /* component not found */
        current = next;
        token = strtok(NULL, "/");
    }
    return current;
}

/* ── Add an entry to a directory ─────────────────────────────────────────── */

static int dir_add_entry(FS *fs, int dir_idx, const char *name, uint32_t ino) {
    Inode *dir = &fs->inodes[dir_idx];
    uint8_t block[BLOCK_SIZE];
    uint32_t entries_per_block = BLOCK_SIZE / sizeof(DirEntry);

    /* Search for an empty slot in existing blocks */
    for (int b = 0; b < MAX_DIRECT_BLOCKS; b++) {
        if (dir->blocks[b] == 0) continue;
        if (read_block(fs, dir->blocks[b], block) < 0) return -1;

        DirEntry *entries = (DirEntry *)block;
        for (uint32_t e = 0; e < entries_per_block; e++) {
            if (entries[e].inode_num == 0) {
                /* Found an empty slot */
                entries[e].inode_num = ino;
                strncpy(entries[e].name, name, 27);
                entries[e].name[27] = '\0';
                write_block(fs, dir->blocks[b], block);
                dir->size += sizeof(DirEntry);
                return 0;
            }
        }
    }

    /* Allocate a new block for this directory */
    int new_blk = alloc_block(fs);
    if (new_blk < 0) { fprintf(stderr, "disk full\n"); return -1; }

    /* Find an empty direct block slot in the inode */
    for (int b = 0; b < MAX_DIRECT_BLOCKS; b++) {
        if (dir->blocks[b] == 0) {
            dir->blocks[b] = (uint32_t)new_blk;
            memset(block, 0, BLOCK_SIZE);
            DirEntry *entries = (DirEntry *)block;
            entries[0].inode_num = ino;
            strncpy(entries[0].name, name, 27);
            entries[0].name[27] = '\0';
            write_block(fs, new_blk, block);
            dir->size += sizeof(DirEntry);
            return 0;
        }
    }

    fprintf(stderr, "directory too large\n");
    free_block(fs, new_blk);
    return -1;
}

/* ── Remove an entry from a directory ─────────────────────────────────────── */

static int dir_remove_entry(FS *fs, int dir_idx, const char *name) {
    Inode *dir = &fs->inodes[dir_idx];
    uint8_t block[BLOCK_SIZE];
    uint32_t entries_per_block = BLOCK_SIZE / sizeof(DirEntry);

    for (int b = 0; b < MAX_DIRECT_BLOCKS; b++) {
        if (dir->blocks[b] == 0) continue;
        if (read_block(fs, dir->blocks[b], block) < 0) return -1;

        DirEntry *entries = (DirEntry *)block;
        for (uint32_t e = 0; e < entries_per_block; e++) {
            if (entries[e].inode_num != 0 &&
                strncmp(entries[e].name, name, 27) == 0) {
                entries[e].inode_num = 0;
                memset(entries[e].name, 0, 28);
                write_block(fs, dir->blocks[b], block);
                return 0;
            }
        }
    }
    return -1;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

int fs_format(const char *filename, uint32_t total_blocks) {
    if (total_blocks > MAX_BLOCKS) {
        fprintf(stderr, "fs_format: max %d blocks\n", MAX_BLOCKS);
        return -1;
    }

    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("fs_format: open"); return -1; }

    /* Size the file: superblock + inode table + data blocks */
    off_t total_size = DATA_OFFSET + (off_t)total_blocks * BLOCK_SIZE;
    if (ftruncate(fd, total_size) < 0) { perror("ftruncate"); close(fd); return -1; }

    /* Build and write superblock */
    Superblock sb = {0};
    sb.magic        = FS_MAGIC;
    sb.total_blocks = total_blocks;
    sb.free_blocks  = total_blocks;
    /* All bits in bitmap start as 0 (free) */

    if (pwrite(fd, &sb, sizeof(Superblock), SUPERBLOCK_OFFSET)
            != sizeof(Superblock)) {
        perror("write superblock"); close(fd); return -1;
    }

    /* Initialise inode table (all INODE_FREE) */
    Inode inodes[MAX_INODES] = {0};
    if (pwrite(fd, inodes, sizeof(inodes), INODE_TABLE_OFFSET)
            != (ssize_t)sizeof(inodes)) {
        perror("write inodes"); close(fd); return -1;
    }

    close(fd);

    /*
     * Mount, create root directory at inode 0, then unmount.
     * We do this through fs_mount so the root inode is properly set up.
     */
    FS *fs = fs_mount(filename);
    if (!fs) return -1;

    /* Manually set up root inode (inode 0) */
    fs->inodes[0].type     = INODE_DIR;
    fs->inodes[0].size     = 0;
    fs->inodes[0].created  = time(NULL);
    fs->inodes[0].modified = fs->inodes[0].created;
    fs->sb.inode_count     = 1;

    flush_superblock(fs);
    flush_inodes(fs);
    fs_unmount(fs);

    printf("Formatted filesystem: %s (%u blocks, %zu bytes)\n",
           filename, total_blocks, (size_t)total_size);
    return 0;
}

FS *fs_mount(const char *filename) {
    int fd = open(filename, O_RDWR);
    if (fd < 0) { perror("fs_mount: open"); return NULL; }

    FS *fs = calloc(1, sizeof(FS));
    if (!fs) { close(fd); return NULL; }
    fs->fd = fd;

    /* Read superblock */
    if (pread(fd, &fs->sb, sizeof(Superblock), SUPERBLOCK_OFFSET)
            != sizeof(Superblock)) {
        perror("read superblock"); goto err;
    }

    if (fs->sb.magic != FS_MAGIC) {
        fprintf(stderr, "fs_mount: bad magic (not a valid filesystem)\n");
        goto err;
    }

    /* Read inode table */
    if (pread(fd, fs->inodes, MAX_INODES * sizeof(Inode), INODE_TABLE_OFFSET)
            != (ssize_t)(MAX_INODES * sizeof(Inode))) {
        perror("read inodes"); goto err;
    }

    return fs;

err:
    close(fd);
    free(fs);
    return NULL;
}

void fs_unmount(FS *fs) {
    if (!fs) return;
    flush_superblock(fs);
    flush_inodes(fs);
    close(fs->fd);
    free(fs);
}

int fs_mkdir(FS *fs, const char *path) {
    char parent[256], name[64];
    split_path(path, parent, sizeof(parent), name, sizeof(name));

    int parent_idx = resolve_path(fs, parent);
    if (parent_idx < 0) {
        fprintf(stderr, "mkdir: parent not found: %s\n", parent);
        return -1;
    }
    if (fs->inodes[parent_idx].type != INODE_DIR) {
        fprintf(stderr, "mkdir: parent is not a directory\n");
        return -1;
    }

    /* Check for name collision */
    if (lookup_in_dir(fs, parent_idx, name) >= 0) {
        fprintf(stderr, "mkdir: already exists: %s\n", path);
        return -1;
    }

    int new_ino = alloc_inode(fs);
    if (new_ino < 0) { fprintf(stderr, "mkdir: no inodes left\n"); return -1; }

    fs->inodes[new_ino].type     = INODE_DIR;
    fs->inodes[new_ino].size     = 0;
    fs->inodes[new_ino].created  = time(NULL);
    fs->inodes[new_ino].modified = fs->inodes[new_ino].created;

    if (dir_add_entry(fs, parent_idx, name, (uint32_t)new_ino) < 0) {
        free_inode(fs, new_ino);
        return -1;
    }

    flush_superblock(fs);
    flush_inodes(fs);
    return 0;
}

int fs_create(FS *fs, const char *path) {
    char parent[256], name[64];
    split_path(path, parent, sizeof(parent), name, sizeof(name));

    int parent_idx = resolve_path(fs, parent);
    if (parent_idx < 0) {
        fprintf(stderr, "create: parent not found: %s\n", parent);
        return -1;
    }

    if (lookup_in_dir(fs, parent_idx, name) >= 0) {
        fprintf(stderr, "create: already exists: %s\n", path);
        return -1;
    }

    int new_ino = alloc_inode(fs);
    if (new_ino < 0) { fprintf(stderr, "create: no inodes left\n"); return -1; }

    fs->inodes[new_ino].type     = INODE_FILE;
    fs->inodes[new_ino].size     = 0;
    fs->inodes[new_ino].created  = time(NULL);
    fs->inodes[new_ino].modified = fs->inodes[new_ino].created;

    if (dir_add_entry(fs, parent_idx, name, (uint32_t)new_ino) < 0) {
        free_inode(fs, new_ino);
        return -1;
    }

    flush_superblock(fs);
    flush_inodes(fs);
    return 0;
}

int fs_write(FS *fs, const char *path, const void *data, uint32_t len) {
    int ino = resolve_path(fs, path);
    if (ino < 0) { fprintf(stderr, "write: not found: %s\n", path); return -1; }
    if (fs->inodes[ino].type != INODE_FILE) {
        fprintf(stderr, "write: not a file: %s\n", path);
        return -1;
    }

    Inode *inode = &fs->inodes[ino];

    /* Free any existing data blocks (overwrite semantics) */
    for (int b = 0; b < MAX_DIRECT_BLOCKS; b++) {
        if (inode->blocks[b]) {
            free_block(fs, inode->blocks[b]);
            inode->blocks[b] = 0;
        }
    }

    uint32_t written = 0;
    int block_idx = 0;

    while (written < len && block_idx < MAX_DIRECT_BLOCKS) {
        int blk = alloc_block(fs);
        if (blk < 0) { fprintf(stderr, "write: disk full\n"); break; }

        uint32_t chunk = len - written;
        if (chunk > BLOCK_SIZE) chunk = BLOCK_SIZE;

        uint8_t block[BLOCK_SIZE] = {0};
        memcpy(block, (const uint8_t *)data + written, chunk);
        write_block(fs, (uint32_t)blk, block);

        inode->blocks[block_idx++] = (uint32_t)blk;
        written += chunk;
    }

    inode->size     = written;
    inode->modified = time(NULL);

    flush_superblock(fs);
    flush_inodes(fs);
    return (int)written;
}

int fs_read(FS *fs, const char *path, void *buf, uint32_t maxlen) {
    int ino = resolve_path(fs, path);
    if (ino < 0) { fprintf(stderr, "read: not found: %s\n", path); return -1; }
    if (fs->inodes[ino].type != INODE_FILE) {
        fprintf(stderr, "read: not a file: %s\n", path);
        return -1;
    }

    Inode *inode = &fs->inodes[ino];
    uint32_t total   = inode->size < maxlen ? inode->size : maxlen;
    uint32_t read_so_far = 0;
    uint8_t  block[BLOCK_SIZE];

    for (int b = 0; b < MAX_DIRECT_BLOCKS && read_so_far < total; b++) {
        if (inode->blocks[b] == 0) break;
        if (read_block(fs, inode->blocks[b], block) < 0) return -1;

        uint32_t chunk = total - read_so_far;
        if (chunk > BLOCK_SIZE) chunk = BLOCK_SIZE;
        memcpy((uint8_t *)buf + read_so_far, block, chunk);
        read_so_far += chunk;
    }

    return (int)read_so_far;
}

int fs_ls(FS *fs, const char *path) {
    int ino = resolve_path(fs, path);
    if (ino < 0) { fprintf(stderr, "ls: not found: %s\n", path); return -1; }
    if (fs->inodes[ino].type != INODE_DIR) {
        fprintf(stderr, "ls: not a directory: %s\n", path);
        return -1;
    }

    Inode *dir = &fs->inodes[ino];
    uint8_t block[BLOCK_SIZE];
    uint32_t entries_per_block = BLOCK_SIZE / sizeof(DirEntry);
    int count = 0;

    printf("Contents of %s:\n", path);
    printf("  %-8s  %-6s  %s\n", "INODE", "TYPE", "NAME");

    for (int b = 0; b < MAX_DIRECT_BLOCKS; b++) {
        if (dir->blocks[b] == 0) continue;
        if (read_block(fs, dir->blocks[b], block) < 0) return -1;

        DirEntry *entries = (DirEntry *)block;
        for (uint32_t e = 0; e < entries_per_block; e++) {
            if (entries[e].inode_num == 0) continue;
            int ent_ino = (int)entries[e].inode_num;
            const char *type = "???";
            if (ent_ino < MAX_INODES) {
                type = (fs->inodes[ent_ino].type == INODE_DIR) ? "DIR" : "FILE";
            }
            printf("  %-8u  %-6s  %s\n",
                   entries[e].inode_num, type, entries[e].name);
            count++;
        }
    }

    if (count == 0) printf("  (empty)\n");
    return 0;
}

int fs_stat(FS *fs, const char *path) {
    int ino = resolve_path(fs, path);
    if (ino < 0) { fprintf(stderr, "stat: not found: %s\n", path); return -1; }

    Inode *inode = &fs->inodes[ino];
    char ctime_buf[64], mtime_buf[64];

    struct tm *tm;
    tm = localtime(&inode->created);
    strftime(ctime_buf, sizeof(ctime_buf), "%Y-%m-%d %H:%M:%S", tm);
    tm = localtime(&inode->modified);
    strftime(mtime_buf, sizeof(mtime_buf), "%Y-%m-%d %H:%M:%S", tm);

    printf("Inode %d:\n", ino);
    printf("  Type    : %s\n", inode->type == INODE_DIR ? "directory" : "file");
    printf("  Size    : %u bytes\n", inode->size);
    printf("  Created : %s\n", ctime_buf);
    printf("  Modified: %s\n", mtime_buf);
    printf("  Blocks  :");
    for (int b = 0; b < MAX_DIRECT_BLOCKS; b++) {
        if (inode->blocks[b]) printf(" %u", inode->blocks[b]);
    }
    printf("\n");
    return 0;
}

int fs_rm(FS *fs, const char *path) {
    char parent[256], name[64];
    split_path(path, parent, sizeof(parent), name, sizeof(name));

    int parent_idx = resolve_path(fs, parent);
    if (parent_idx < 0) { fprintf(stderr, "rm: parent not found\n"); return -1; }

    int ino = lookup_in_dir(fs, parent_idx, name);
    if (ino < 0) { fprintf(stderr, "rm: not found: %s\n", path); return -1; }

    if (fs->inodes[ino].type == INODE_DIR) {
        fprintf(stderr, "rm: cannot remove directory (use rmdir)\n");
        return -1;
    }

    /* Free data blocks */
    Inode *inode = &fs->inodes[ino];
    for (int b = 0; b < MAX_DIRECT_BLOCKS; b++) {
        if (inode->blocks[b]) {
            free_block(fs, inode->blocks[b]);
            inode->blocks[b] = 0;
        }
    }

    /* Free inode */
    free_inode(fs, ino);

    /* Remove from directory */
    dir_remove_entry(fs, parent_idx, name);

    flush_superblock(fs);
    flush_inodes(fs);
    return 0;
}
