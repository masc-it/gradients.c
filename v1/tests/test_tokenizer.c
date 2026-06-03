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

#define CHECK_TRUE(expr)                                                         \
    do {                                                                         \
        if (!(expr)) {                                                           \
            fprintf(stderr, "%s failed at %s:%d\n", #expr, __FILE__, __LINE__); \
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

static int ids_equal(const int32_t *a, int na, const int32_t *b, int nb)
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

static int has_digit_run_123(const int32_t *ids, int n_ids)
{
    int i;
    for (i = 0; i + 2 < n_ids; ++i) {
        if (ids[i] == (int32_t)'1' && ids[i + 1] == (int32_t)'2' &&
            ids[i + 2] == (int32_t)'3') {
            return 1;
        }
    }
    return 0;
}

static int train_fixture(const char *path, gd_tokenizer **tok_out)
{
    const char *inputs[1];
    const char *specials[3] = {"<|pad|>", "<|im_start|>", "<|im_end|>"};
    gd_bpe_train_config cfg;

    inputs[0] = path;
    cfg.vocab_size = 320;
    cfg.min_frequency = 2;
    cfg.split_digits = 1;
    cfg.n_special_tokens = 3;
    cfg.special_tokens = specials;
    cfg.seed = 1234U;
    CHECK_OK(gd_bpe_tokenizer_train(inputs, 1, &cfg, tok_out));
    return 0;
}

static int test_train_encode_decode_save_load(void)
{
    const char *corpus_path = "/tmp/gd_tokenizer_corpus.txt";
    const char *tok_path = "/tmp/gd_tokenizer.json";
    const char *corpus =
        "<|im_start|>ciao mondo 123 abc123def<|im_end|>\n"
        "<|im_start|>ciao mondo 123 abc123def<|im_end|>\n"
        "hello hello hello world world token token token\n"
        "digits 9876543210 must stay split\n";
    gd_tokenizer *tok_a = NULL;
    gd_tokenizer *tok_b = NULL;
    gd_tokenizer *loaded = NULL;
    int32_t im_start = -1;
    int32_t im_end = -1;
    int32_t pad = -1;
    int32_t *ids = NULL;
    int32_t *ids_loaded = NULL;
    int n_ids = 0;
    int n_ids_loaded = 0;
    char *decoded = NULL;
    gd_tokenizer_config load_cfg;

    CHECK_TRUE(write_file(corpus_path, corpus) == 0);
    CHECK_TRUE(train_fixture(corpus_path, &tok_a) == 0);
    CHECK_TRUE(train_fixture(corpus_path, &tok_b) == 0);
    CHECK_TRUE(gd_tokenizer_vocab_size(tok_a) >= 256 + 3);
    CHECK_TRUE(gd_tokenizer_hash(tok_a) == gd_tokenizer_hash(tok_b));

    CHECK_OK(gd_tokenizer_id(tok_a, "<|pad|>", &pad));
    CHECK_OK(gd_tokenizer_id(tok_a, "<|im_start|>", &im_start));
    CHECK_OK(gd_tokenizer_id(tok_a, "<|im_end|>", &im_end));
    CHECK_TRUE(pad == 256);
    CHECK_TRUE(im_start == 257);
    CHECK_TRUE(im_end == 258);

    CHECK_OK(gd_tokenizer_encode(tok_a,
                                 "<|im_start|>perché abc123def<|im_end|>",
                                 &ids,
                                 &n_ids));
    CHECK_TRUE(n_ids >= 8);
    CHECK_TRUE(ids[0] == im_start);
    CHECK_TRUE(ids[n_ids - 1] == im_end);
    CHECK_TRUE(ids[1] != pad);
    CHECK_TRUE(has_digit_run_123(ids, n_ids));

    CHECK_OK(gd_tokenizer_decode(tok_a, ids, n_ids, &decoded));
    CHECK_TRUE(strcmp(decoded, "<|im_start|>perché abc123def<|im_end|>") == 0);

    CHECK_OK(gd_bpe_tokenizer_save(tok_a, tok_path));
    load_cfg.split_digits = 1;
    load_cfg.allow_special = 1;
    CHECK_OK(gd_bpe_tokenizer_load(tok_path, &load_cfg, &loaded));
    CHECK_TRUE(gd_tokenizer_hash(tok_a) == gd_tokenizer_hash(loaded));
    CHECK_OK(gd_tokenizer_encode(loaded,
                                 "<|im_start|>perché abc123def<|im_end|>",
                                 &ids_loaded,
                                 &n_ids_loaded));
    CHECK_TRUE(ids_equal(ids, n_ids, ids_loaded, n_ids_loaded));

    gd_tokenizer_free(ids);
    gd_tokenizer_free(ids_loaded);
    gd_tokenizer_free(decoded);
    gd_tokenizer_destroy(tok_a);
    gd_tokenizer_destroy(tok_b);
    gd_tokenizer_destroy(loaded);
    (void)remove(corpus_path);
    (void)remove(tok_path);
    return 0;
}

static int test_invalid_special_policy(void)
{
    const char *corpus_path = "/tmp/gd_tokenizer_invalid_special.txt";
    const char *inputs[1];
    const char *specials[2] = {"<|im_start|>", "<|im_start|>"};
    gd_bpe_train_config cfg;
    gd_tokenizer *tok = NULL;

    CHECK_TRUE(write_file(corpus_path, "hello world hello world\n") == 0);
    inputs[0] = corpus_path;
    cfg.vocab_size = 300;
    cfg.min_frequency = 2;
    cfg.split_digits = 1;
    cfg.n_special_tokens = 2;
    cfg.special_tokens = specials;
    cfg.seed = 1234U;
    CHECK_STATUS(gd_bpe_tokenizer_train(inputs, 1, &cfg, &tok), GD_ERR_INVALID_ARGUMENT);
    CHECK_TRUE(strstr(gd_last_error(), "duplicate special token") != NULL);
    gd_tokenizer_destroy(tok);
    (void)remove(corpus_path);
    return 0;
}

static int test_digit_split_blocks_digit_merges(void)
{
    const char *corpus_path = "/tmp/gd_tokenizer_digits.txt";
    const char *corpus =
        "11111111111111111111111111111111111111111111111111\n"
        "22222222222222222222222222222222222222222222222222\n"
        "12312312312312312312312312312312312312312312312312\n";
    gd_tokenizer *tok = NULL;
    int32_t *ids = NULL;
    int n_ids = 0;

    CHECK_TRUE(write_file(corpus_path, corpus) == 0);
    CHECK_TRUE(train_fixture(corpus_path, &tok) == 0);
    CHECK_OK(gd_tokenizer_encode(tok, "123321", &ids, &n_ids));
    CHECK_TRUE(n_ids == 6);
    CHECK_TRUE(ids[0] == (int32_t)'1');
    CHECK_TRUE(ids[1] == (int32_t)'2');
    CHECK_TRUE(ids[2] == (int32_t)'3');
    CHECK_TRUE(ids[3] == (int32_t)'3');
    CHECK_TRUE(ids[4] == (int32_t)'2');
    CHECK_TRUE(ids[5] == (int32_t)'1');

    gd_tokenizer_free(ids);
    gd_tokenizer_destroy(tok);
    (void)remove(corpus_path);
    return 0;
}

int main(void)
{
    if (test_train_encode_decode_save_load() != 0) {
        return 1;
    }
    if (test_invalid_special_policy() != 0) {
        return 1;
    }
    if (test_digit_split_blocks_digit_merges() != 0) {
        return 1;
    }
    return 0;
}
