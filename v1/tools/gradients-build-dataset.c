#include "gradients/dataset.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_INPUTS 256

static void usage(FILE *f)
{
    fprintf(f,
            "usage: gradients-build-dataset --tokenizer tokenizer.json "
            "--input PATH [--input PATH ...] --output DIR [options]\n"
            "\n"
            "options:\n"
            "  --block-len N           block length (default: 512)\n"
            "  --train-ratio R         train split ratio (default: 0.9)\n"
            "  --val-ratio R           validation split ratio (default: 0.1)\n"
            "  --seed N                deterministic split seed (default: 1234)\n"
            "  --wrap-plain-text       wrap paragraphs as <|im_start|>...<|im_end|>\n"
            "  --no-shuffle-split      preserve record order before split\n"
            "  --split-digits          accepted for config compatibility; tokenizer owns it\n"
            "  --special im_start=STR  override im_start token\n"
            "  --special im_end=STR    override im_end token\n"
            "  --help                  show this help\n");
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
    if (errno != 0 || end == s || *end != '\0' || v < 0 || v > INT32_MAX) {
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

static int parse_double(const char *s, double *out)
{
    char *end = NULL;
    double v;
    if (s == NULL || out == NULL) {
        return 1;
    }
    errno = 0;
    v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0') {
        return 1;
    }
    *out = v;
    return 0;
}

static int parse_special(const char *arg, const char **im_start, const char **im_end)
{
    const char prefix_start[] = "im_start=";
    const char prefix_end[] = "im_end=";
    if (strncmp(arg, prefix_start, sizeof(prefix_start) - 1U) == 0) {
        *im_start = arg + sizeof(prefix_start) - 1U;
        return 0;
    }
    if (strncmp(arg, prefix_end, sizeof(prefix_end) - 1U) == 0) {
        *im_end = arg + sizeof(prefix_end) - 1U;
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    const char *inputs[MAX_INPUTS];
    gd_dataset_build_config cfg;
    gd_dataset_build_result result;
    gd_status status;
    int n_inputs = 0;
    int i;

    memset(&cfg, 0, sizeof(cfg));
    memset(&result, 0, sizeof(result));
    cfg.block_len = 512;
    cfg.train_ratio = 0.9;
    cfg.val_ratio = 0.1;
    cfg.seed = 1234U;
    cfg.im_start = "<|im_start|>";
    cfg.im_end = "<|im_end|>";

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            return 0;
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            cfg.tokenizer_path = argv[++i];
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
            cfg.output_dir = argv[++i];
        } else if (strcmp(argv[i], "--block-len") == 0) {
            if (i + 1 >= argc || parse_int(argv[++i], &cfg.block_len) != 0) {
                fprintf(stderr, "invalid --block-len\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--train-ratio") == 0) {
            if (i + 1 >= argc || parse_double(argv[++i], &cfg.train_ratio) != 0) {
                fprintf(stderr, "invalid --train-ratio\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--val-ratio") == 0) {
            if (i + 1 >= argc || parse_double(argv[++i], &cfg.val_ratio) != 0) {
                fprintf(stderr, "invalid --val-ratio\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc || parse_u64(argv[++i], &cfg.seed) != 0) {
                fprintf(stderr, "invalid --seed\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--wrap-plain-text") == 0) {
            cfg.wrap_plain_text = 1;
        } else if (strcmp(argv[i], "--no-shuffle-split") == 0) {
            cfg.no_shuffle_split = 1;
        } else if (strcmp(argv[i], "--split-digits") == 0) {
            /* Tokenizer file already records this. Flag kept for config compatibility. */
        } else if (strcmp(argv[i], "--special") == 0) {
            if (i + 1 >= argc || parse_special(argv[++i], &cfg.im_start, &cfg.im_end) != 0) {
                fprintf(stderr, "invalid --special; expected im_start=STR or im_end=STR\n");
                return 2;
            }
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(stderr);
            return 2;
        }
    }

    if (cfg.tokenizer_path == NULL || cfg.output_dir == NULL || n_inputs == 0) {
        usage(stderr);
        return 2;
    }
    cfg.input_paths = inputs;
    cfg.n_input_paths = n_inputs;

    status = gd_dataset_build(&cfg, &result);
    if (status != GD_OK) {
        fprintf(stderr, "build dataset failed: %s\n", gd_last_error());
        return 1;
    }

    printf("{\"output\":\"%s\",\"records\":%d,"
           "\"train_samples\":%llu,\"val_samples\":%llu,"
           "\"train_dropped_tail_tokens\":%llu,\"val_dropped_tail_tokens\":%llu}\n",
           cfg.output_dir,
           result.n_records_total,
           (unsigned long long)result.train.n_samples,
           (unsigned long long)result.val.n_samples,
           (unsigned long long)result.train.dropped_tail_tokens,
           (unsigned long long)result.val.dropped_tail_tokens);
    gd_dataset_build_result_clear(&result);
    return 0;
}
