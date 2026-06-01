/*
 * main.c — Interactive CLI for the toy filesystem.
 *
 * Usage:
 *   ./bin/fs <disk_image>
 *
 * If disk_image does not exist, it is formatted automatically.
 *
 * Commands:
 *   mkdir <path>
 *   create <path>
 *   write <path> <data>
 *   read <path>
 *   ls [path]
 *   stat <path>
 *   rm <path>
 *   help
 *   exit
 */

#define _POSIX_C_SOURCE 200809L

#include "fs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_LINE  1024

static void print_help(void) {
    printf("Commands:\n"
           "  mkdir <path>         create directory\n"
           "  create <path>        create empty file\n"
           "  write <path> <data>  write data to file\n"
           "  read <path>          print file contents\n"
           "  ls [path]            list directory (default: /)\n"
           "  stat <path>          show inode metadata\n"
           "  rm <path>            remove file\n"
           "  help                 show this message\n"
           "  exit                 save and quit\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <disk_image>\n", argv[0]);
        return 1;
    }

    const char *image = argv[1];

    /* Format if the image doesn't exist */
    if (access(image, F_OK) != 0) {
        printf("Image not found, formatting new filesystem: %s\n", image);
        if (fs_format(image, 256) < 0) return 1;
    }

    FS *fs = fs_mount(image);
    if (!fs) {
        fprintf(stderr, "Failed to mount %s\n", image);
        return 1;
    }

    printf("Mounted filesystem: %s\n", image);
    printf("Type 'help' for a list of commands.\n\n");

    char line[MAX_LINE];

    while (1) {
        printf("fs> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            printf("\n");
            break;
        }

        /* Strip trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[--len] = '\0';
        if (len == 0) continue;

        /* Tokenise: cmd [arg1] [arg2] */
        char *cmd  = strtok(line, " ");
        char *arg1 = strtok(NULL, " ");
        char *arg2 = strtok(NULL, "");  /* rest of line for write data */

        if (!cmd) continue;

        if (strcmp(cmd, "exit") == 0) {
            break;
        } else if (strcmp(cmd, "help") == 0) {
            print_help();
        } else if (strcmp(cmd, "mkdir") == 0) {
            if (!arg1) { fprintf(stderr, "Usage: mkdir <path>\n"); continue; }
            fs_mkdir(fs, arg1);
        } else if (strcmp(cmd, "create") == 0) {
            if (!arg1) { fprintf(stderr, "Usage: create <path>\n"); continue; }
            fs_create(fs, arg1);
        } else if (strcmp(cmd, "write") == 0) {
            if (!arg1 || !arg2) {
                fprintf(stderr, "Usage: write <path> <data>\n"); continue;
            }
            int n = fs_write(fs, arg1, arg2, (uint32_t)strlen(arg2));
            if (n >= 0) printf("Wrote %d bytes.\n", n);
        } else if (strcmp(cmd, "read") == 0) {
            if (!arg1) { fprintf(stderr, "Usage: read <path>\n"); continue; }
            char buf[8192] = {0};
            int n = fs_read(fs, arg1, buf, sizeof(buf) - 1);
            if (n >= 0) {
                buf[n] = '\0';
                printf("%s\n", buf);
            }
        } else if (strcmp(cmd, "ls") == 0) {
            fs_ls(fs, arg1 ? arg1 : "/");
        } else if (strcmp(cmd, "stat") == 0) {
            if (!arg1) { fprintf(stderr, "Usage: stat <path>\n"); continue; }
            fs_stat(fs, arg1);
        } else if (strcmp(cmd, "rm") == 0) {
            if (!arg1) { fprintf(stderr, "Usage: rm <path>\n"); continue; }
            fs_rm(fs, arg1);
        } else {
            fprintf(stderr, "Unknown command: %s (type 'help')\n", cmd);
        }
    }

    fs_unmount(fs);
    printf("Filesystem unmounted.\n");
    return 0;
}
