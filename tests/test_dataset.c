#include "gradients/dataset.h"
#include "gradients/tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_OK(expr)                                                           \
    do {                                                                         \
        gd_status status_ = (expr);                                              \
        if (status_ != GD_OK) {                                                  \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());        \
            return 1;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_STATUS(expr, expected)                                             \
    do {                                                                         \
        gd_status status_ = (expr);                                              \
        if (status_ != (expected)) {                                             \
            fprintf(stderr,                                                       \
                    "%s got %s expected %s; last_error=%s\n",                   \
                    #expr,                                                       \
                    gd_status_name(status_),                                     \
                    gd_status_name(expected),                                    \
                    gd_last_error());                                            \
            return 1;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_TRUE(expr)                                                         \
    do {                                                                         \
        if (!(expr)) {                                                           \
            fprintf(stderr, "%s failed at %s:%d\n", #expr, __FILE__, __LINE__); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

static int write_file(const char *path, const char *text)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        return 1;
    }
    if (fwrite(text, 1U, strlen(text), f) != strlen(text)) {
        (void)fclose(f);
        return 1;
    }
    if (fclose(f) != 0) {
        return 1;
    }
    return 0;
}

static int train_tokenizer(const char *input_path, const char *tokenizer_path)
{
    const char *inputs[1];
    const char *specials[3] = {"<|pad|>", "<|im_start|>", "<|im_end|>"};
    gd_bpe_train_config cfg;
    gd_tokenizer *tok = NULL;

    inputs[0] = input_path;
    cfg.vocab_size = 384;
    cfg.min_frequency = 2;
    cfg.split_digits = 1;
    cfg.n_special_tokens = 3;
    cfg.special_tokens = specials;
    cfg.seed = 1234U;
    CHECK_OK(gd_bpe_tokenizer_train(inputs, 1, &cfg, &tok));
    CHECK_OK(gd_bpe_tokenizer_save(tok, tokenizer_path));
    gd_tokenizer_destroy(tok);
    return 0;
}

static int arrays_equal(const int *a, int na, const int *b, int nb)
{
    int i;
    if (na != nb) {
        return 0;
    }
    for (i = 0; i < na; ++i) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}

static int arrays_disjoint(const int *a, int na, const int *b, int nb)
{
    int i;
    int j;
    for (i = 0; i < na; ++i) {
        for (j = 0; j < nb; ++j) {
            if (a[i] == b[j]) {
                return 0;
            }
        }
    }
    return 1;
}

static int read_first_u16_payload(const char *path, uint16_t *out)
{
    FILE *f = fopen(path, "rb");
    unsigned char b[2];
    if (f == NULL) {
        return 1;
    }
    if (fseek(f, (long)GD_GDTOK_HEADER_SIZE, SEEK_SET) != 0) {
        (void)fclose(f);
        return 1;
    }
    if (fread(b, 1U, sizeof(b), f) != sizeof(b)) {
        (void)fclose(f);
        return 1;
    }
    if (fclose(f) != 0) {
        return 1;
    }
    *out = (uint16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8U));
    return 0;
}

static int test_formatted_dataset_split_and_header(void)
{
    const char *corpus_path = "/tmp/gd_dataset_formatted.txt";
    const char *tokenizer_path = "/tmp/gd_dataset_formatted_tokenizer.json";
    const char *corpus =
        "<|im_start|>alpha alpha alpha one one one one one<|im_end|>\n"
        "<|im_start|>beta beta beta two two two two two<|im_end|>\n"
        "<|im_start|>gamma gamma gamma three three three three three<|im_end|>\n"
        "<|im_start|>delta delta delta four four four four four<|im_end|>\n"
        "<|im_start|>epsilon epsilon epsilon five five five five five<|im_end|>\n"
        "<|im_start|>zeta zeta zeta six six six six six<|im_end|>\n";
    const char *inputs[1];
    gd_dataset_build_config cfg;
    gd_dataset_build_result a;
    gd_dataset_build_result b;
    gd_gdtok_header h;

    memset(&cfg, 0, sizeof(cfg));
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    CHECK_TRUE(write_file(corpus_path, corpus) == 0);
    CHECK_TRUE(train_tokenizer(corpus_path, tokenizer_path) == 0);

    inputs[0] = corpus_path;
    cfg.tokenizer_path = tokenizer_path;
    cfg.input_paths = inputs;
    cfg.n_input_paths = 1;
    cfg.output_dir = "/tmp/gd_dataset_a";
    cfg.block_len = 8;
    cfg.train_ratio = 0.5;
    cfg.val_ratio = 0.5;
    cfg.seed = 99U;
    cfg.wrap_plain_text = 0;
    cfg.im_start = "<|im_start|>";
    cfg.im_end = "<|im_end|>";
    CHECK_OK(gd_dataset_build(&cfg, &a));
    cfg.output_dir = "/tmp/gd_dataset_b";
    CHECK_OK(gd_dataset_build(&cfg, &b));

    CHECK_TRUE(a.n_records_total == 6);
    CHECK_TRUE(a.train.n_record_indices == 3);
    CHECK_TRUE(a.val.n_record_indices == 3);
    CHECK_TRUE(arrays_equal(a.train.record_indices,
                            a.train.n_record_indices,
                            b.train.record_indices,
                            b.train.n_record_indices));
    CHECK_TRUE(arrays_equal(a.val.record_indices,
                            a.val.n_record_indices,
                            b.val.record_indices,
                            b.val.n_record_indices));
    CHECK_TRUE(arrays_disjoint(a.train.record_indices,
                               a.train.n_record_indices,
                               a.val.record_indices,
                               a.val.n_record_indices));
    CHECK_TRUE(a.train.n_tokens_written == a.train.n_samples * 8U + 1U);
    CHECK_TRUE(a.train.dropped_tail_tokens ==
               a.train.n_tokens_total - a.train.n_tokens_written);

    CHECK_OK(gd_gdtok_read_header(a.train.shard_path, &h));
    CHECK_TRUE(h.version == GD_GDTOK_VERSION);
    CHECK_TRUE(h.block_len == 8U);
    CHECK_TRUE(h.dtype == GD_GDTOK_DTYPE_U16);
    CHECK_TRUE(h.n_tokens == a.train.n_tokens_written);
    CHECK_TRUE(h.n_samples == a.train.n_samples);
    CHECK_TRUE(h.payload_offset == GD_GDTOK_HEADER_SIZE);

    gd_dataset_build_result_clear(&a);
    gd_dataset_build_result_clear(&b);
    (void)remove(corpus_path);
    (void)remove(tokenizer_path);
    return 0;
}

static int test_plain_text_wrap_and_special_boundary(void)
{
    const char *plain_path = "/tmp/gd_dataset_plain.txt";
    const char *tokenizer_path = "/tmp/gd_dataset_plain_tokenizer.json";
    const char *plain =
        "primo paragrafo con molte parole molte parole 123\n"
        "ancora primo paragrafo parole parole parole\n"
        "\n"
        "secondo paragrafo con molte parole molte parole 456\n"
        "\n"
        "terzo paragrafo con molte parole molte parole 789\n";
    const char *inputs[1];
    gd_dataset_build_config cfg;
    gd_dataset_build_result result;
    gd_tokenizer_config tok_cfg;
    gd_tokenizer *tok = NULL;
    int32_t im_start_id = -1;
    uint16_t first_token = 0U;

    memset(&cfg, 0, sizeof(cfg));
    memset(&result, 0, sizeof(result));
    CHECK_TRUE(write_file(plain_path, plain) == 0);
    CHECK_TRUE(train_tokenizer(plain_path, tokenizer_path) == 0);

    inputs[0] = plain_path;
    cfg.tokenizer_path = tokenizer_path;
    cfg.input_paths = inputs;
    cfg.n_input_paths = 1;
    cfg.output_dir = "/tmp/gd_dataset_wrap";
    cfg.block_len = 4;
    cfg.train_ratio = 0.67;
    cfg.val_ratio = 0.33;
    cfg.seed = 7U;
    cfg.wrap_plain_text = 1;
    cfg.im_start = "<|im_start|>";
    cfg.im_end = "<|im_end|>";
    CHECK_OK(gd_dataset_build(&cfg, &result));
    CHECK_TRUE(result.n_records_total == 3);
    CHECK_TRUE(result.train.n_record_indices + result.val.n_record_indices == 3);
    CHECK_TRUE(result.train.n_samples > 0U);

    tok_cfg.split_digits = 1;
    tok_cfg.allow_special = 1;
    CHECK_OK(gd_bpe_tokenizer_load(tokenizer_path, &tok_cfg, &tok));
    CHECK_OK(gd_tokenizer_id(tok, "<|im_start|>", &im_start_id));
    CHECK_TRUE(read_first_u16_payload(result.train.shard_path, &first_token) == 0);
    CHECK_TRUE((int32_t)first_token == im_start_id);

    gd_tokenizer_destroy(tok);
    gd_dataset_build_result_clear(&result);
    (void)remove(plain_path);
    (void)remove(tokenizer_path);
    return 0;
}

static int test_invalid_formatted_record_rejected(void)
{
    const char *train_path = "/tmp/gd_dataset_invalid_train.txt";
    const char *bad_path = "/tmp/gd_dataset_invalid_bad.txt";
    const char *tokenizer_path = "/tmp/gd_dataset_invalid_tokenizer.json";
    const char *inputs[1];
    gd_dataset_build_config cfg;
    gd_dataset_build_result result;

    memset(&cfg, 0, sizeof(cfg));
    memset(&result, 0, sizeof(result));
    CHECK_TRUE(write_file(train_path,
                          "<|im_start|>valid valid valid<|im_end|>\n"
                          "<|im_start|>more more more<|im_end|>\n") == 0);
    CHECK_TRUE(write_file(bad_path, "<|im_start|>missing end\n") == 0);
    CHECK_TRUE(train_tokenizer(train_path, tokenizer_path) == 0);

    inputs[0] = bad_path;
    cfg.tokenizer_path = tokenizer_path;
    cfg.input_paths = inputs;
    cfg.n_input_paths = 1;
    cfg.output_dir = "/tmp/gd_dataset_bad";
    cfg.block_len = 8;
    cfg.train_ratio = 0.9;
    cfg.val_ratio = 0.1;
    cfg.seed = 1U;
    cfg.im_start = "<|im_start|>";
    cfg.im_end = "<|im_end|>";
    CHECK_STATUS(gd_dataset_build(&cfg, &result), GD_ERR_INVALID_ARGUMENT);
    gd_dataset_build_result_clear(&result);
    (void)remove(train_path);
    (void)remove(bad_path);
    (void)remove(tokenizer_path);
    return 0;
}

int main(void)
{
    if (test_formatted_dataset_split_and_header() != 0) {
        return 1;
    }
    if (test_plain_text_wrap_and_special_boundary() != 0) {
        return 1;
    }
    if (test_invalid_formatted_record_rejected() != 0) {
        return 1;
    }
    return 0;
}
