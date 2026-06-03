#include "gradients/tokenizer.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_INPUTS 256
#define MAX_SPECIALS 64

static void usage(FILE *f)
{
    fprintf(f,
            "usage: gradients-train-bpe --input PATH [--input PATH ...] "
            "--output tokenizer.json --vocab-size N [options]\n"
            "\n"
            "options:\n"
            "  --min-frequency N   minimum pair frequency (default: 2)\n"
            "  --split-digits      prevent digit-digit merges (recommended)\n"
            "  --special TOKEN     reserve special token; repeatable\n"
            "  --seed N            recorded deterministic seed (default: 0)\n"
            "  --help              show this help\n");
}

static int parse_int(const char *s, int *out)
{
    char *end = NULL;
    long v;
    if (s == NULL || out == NULL) {
        return 1;
    }
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 0 || v > INT_MAX) {
        return 1;
    }
    *out = (int)v;
    return 0;
}

static int parse_u64(const char *s, uint64_t *out)
{
    char *end = NULL;
    unsigned long long v;
    if (s == NULL || out == NULL) {
        return 1;
    }
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return 1;
    }
    *out = (uint64_t)v;
    return 0;
}

int main(int argc, char **argv)
{
    const char *inputs[MAX_INPUTS];
    const char *specials[MAX_SPECIALS];
    int n_inputs = 0;
    int n_specials = 0;
    const char *output = NULL;
    gd_bpe_train_config cfg;
    gd_tokenizer *tok = NULL;
    gd_status status;
    int i;

    memset(&cfg, 0, sizeof(cfg));
    cfg.min_frequency = 2;
    cfg.split_digits = 0;
    cfg.seed = 0U;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            return 0;
        } else if (strcmp(argv[i], "--input") == 0) {
            if (i + 1 >= argc || n_inputs == MAX_INPUTS) {
                usage(stderr);
                return 2;
            }
            inputs[n_inputs] = argv[++i];
            n_inputs += 1;
        } else if (strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            output = argv[++i];
        } else if (strcmp(argv[i], "--vocab-size") == 0) {
            if (i + 1 >= argc || parse_int(argv[++i], &cfg.vocab_size) != 0) {
                fprintf(stderr, "invalid --vocab-size\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--min-frequency") == 0) {
            if (i + 1 >= argc || parse_int(argv[++i], &cfg.min_frequency) != 0) {
                fprintf(stderr, "invalid --min-frequency\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--split-digits") == 0) {
            cfg.split_digits = 1;
        } else if (strcmp(argv[i], "--special") == 0) {
            if (i + 1 >= argc || n_specials == MAX_SPECIALS) {
                usage(stderr);
                return 2;
            }
            specials[n_specials] = argv[++i];
            n_specials += 1;
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc || parse_u64(argv[++i], &cfg.seed) != 0) {
                fprintf(stderr, "invalid --seed\n");
                return 2;
            }
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(stderr);
            return 2;
        }
    }

    if (n_inputs == 0 || output == NULL || cfg.vocab_size == 0) {
        usage(stderr);
        return 2;
    }

    cfg.n_special_tokens = n_specials;
    cfg.special_tokens = specials;
    status = gd_bpe_tokenizer_train(inputs, n_inputs, &cfg, &tok);
    if (status != GD_OK) {
        fprintf(stderr, "train failed: %s\n", gd_last_error());
        return 1;
    }
    status = gd_bpe_tokenizer_save(tok, output);
    if (status != GD_OK) {
        fprintf(stderr, "save failed: %s\n", gd_last_error());
        gd_tokenizer_destroy(tok);
        return 1;
    }

    printf("{\"tokenizer\":\"%s\",\"vocab_size\":%d,\"hash\":\"%016llx\"}\n",
           output,
           gd_tokenizer_vocab_size(tok),
           (unsigned long long)gd_tokenizer_hash(tok));
    gd_tokenizer_destroy(tok);
    return 0;
}
