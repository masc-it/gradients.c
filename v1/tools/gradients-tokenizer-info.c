#include "gradients/tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_SPECIALS 64

static void usage(FILE *f)
{
    fprintf(f,
            "usage: gradients-tokenizer-info --tokenizer tokenizer.json "
            "[--special TOKEN ...]\n");
}

int main(int argc, char **argv)
{
    const char *tokenizer_path = NULL;
    const char *specials[MAX_SPECIALS];
    int n_specials = 0;
    gd_tokenizer_config cfg;
    gd_tokenizer *tok = NULL;
    gd_status status;
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            return 0;
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--special") == 0) {
            if (i + 1 >= argc || n_specials == MAX_SPECIALS) {
                usage(stderr);
                return 2;
            }
            specials[n_specials] = argv[++i];
            n_specials += 1;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(stderr);
            return 2;
        }
    }
    if (tokenizer_path == NULL) {
        usage(stderr);
        return 2;
    }

    cfg.split_digits = 1;
    cfg.allow_special = 1;
    status = gd_bpe_tokenizer_load(tokenizer_path, &cfg, &tok);
    if (status != GD_OK) {
        fprintf(stderr, "load failed: %s\n", gd_last_error());
        return 1;
    }

    printf("tokenizer: %s\n", tokenizer_path);
    printf("vocab_size: %d\n", gd_tokenizer_vocab_size(tok));
    printf("hash: %016llx\n", (unsigned long long)gd_tokenizer_hash(tok));
    for (i = 0; i < n_specials; ++i) {
        int32_t id = -1;
        status = gd_tokenizer_id(tok, specials[i], &id);
        if (status == GD_OK) {
            printf("special %s: %d\n", specials[i], (int)id);
        } else {
            printf("special %s: missing\n", specials[i]);
        }
    }

    gd_tokenizer_destroy(tok);
    return 0;
}
