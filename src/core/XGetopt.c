#include <stdio.h>
#include <string.h>
#include "getopt.h"

char *optarg;
int optind = 1, opterr = 1, optopt;

int getopt(int argc, char * const argv[], const char *optstring) {
    static int optpos;
    const char *opt;

    if (optind >= argc || argv[optind][0] != '-' || !argv[optind][1]) {
        return -1;
    }
    if (strcmp(argv[optind], "--") == 0) {
        optind++;
        return -1;
    }

    optopt = argv[optind][optpos = optpos ? optpos : 1];
    opt = strchr(optstring, optopt);
    if (!opt) {
        if (opterr) fprintf(stderr, "Unknown option '-%c'\n", optopt);
        if (!argv[optind][++optpos]) {
            optind++;
            optpos = 0;
        }
        return '?';
    }
    if (opt[1] == ':') {
        if (argv[optind][optpos+1]) {
            optarg = &argv[optind++][optpos+1];
        } else if (++optind >= argc) {
            if (opterr) fprintf(stderr, "Option '-%c' requires an argument\n", optopt);
            optpos = 0;
            return '?';
        } else {
            optarg = argv[optind++];
        }
        optpos = 0;
    } else {
        if (!argv[optind][++optpos]) {
            optind++;
            optpos = 0;
        }
        optarg = NULL;
    }
    return optopt;
}
