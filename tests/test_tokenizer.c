#include <gradients/tokenizer.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_tokenizer failed: %s (%s:%d)\n", (msg),     \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr)                                                         \
    do {                                                                       \
        gd_status tokenizer_st__ = (expr);                                     \
        if (tokenizer_st__ != GD_OK) {                                         \
            fprintf(stderr,                                                    \
                    "test_tokenizer failed: %s -> %s (%s:%d)\n",              \
                    #expr,                                                     \
                    gd_status_string(tokenizer_st__),                          \
                    __FILE__,                                                  \
                    __LINE__);                                                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_STATUS(expr, want)                                               \
    do {                                                                       \
        gd_status tokenizer_st__ = (expr);                                     \
        if (tokenizer_st__ != (want)) {                                        \
            fprintf(stderr,                                                    \
                    "test_tokenizer failed: %s -> %s, want %s (%s:%d)\n",     \
                    #expr,                                                     \
                    gd_status_string(tokenizer_st__),                          \
                    gd_status_string((want)),                                  \
                    __FILE__,                                                  \
                    __LINE__);                                                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static void write_text_file(const char *path, const char *text)
{
    FILE *f = fopen(path, "wb");
    size_t len = strlen(text);
    CHECK(f != NULL, "open corpus");
    CHECK(fwrite(text, 1U, len, f) == len, "write corpus");
    CHECK(fclose(f) == 0, "close corpus");
}

static int contains_id(const int32_t *ids, int n_ids, int32_t id)
{
    int i;
    for (i = 0; i < n_ids; ++i) {
        if (ids[i] == id) {
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    const char *corpus_path = "/tmp/gd_tokenizer_test_corpus.txt";
    const char *tokenizer_path = "/tmp/gd_tokenizer_test_tokenizer.json";
    const char *specials[] = {"<eot>"};
    const char *roundtrip = "hello <eot> 123 hello";
    const char *literal_special = "hello <eot>";
    const char corpus_line0[] = "hello hello hello <eot>\n";
    const char corpus_line1[] = "byte-level BPE tokenizer test 123 123 123\n";
    const char corpus_line2[] = "hello tokenizer tokenizer tokenizer\n";
    const char corpus[] =
        "hello hello hello <eot>\n"
        "byte-level BPE tokenizer test 123 123 123\n"
        "hello tokenizer tokenizer tokenizer\n";
    gd_bpe_train_config train_cfg;
    gd_bpe_trainer_stats trainer_stats;
    gd_tokenizer_config load_cfg;
    gd_tokenizer_config no_special_cfg;
    gd_bpe_trainer *trainer = NULL;
    gd_tokenizer *tok = NULL;
    gd_tokenizer *streamed = NULL;
    gd_tokenizer *loaded = NULL;
    gd_tokenizer *no_special = NULL;
    int32_t *ids = NULL;
    int32_t *literal_ids = NULL;
    int n_ids = 0;
    int n_literal_ids = 0;
    int32_t special_id = -1;
    int32_t bad_id = -1;
    char *decoded = NULL;
    uint64_t hash;

    write_text_file(corpus_path, corpus);

    memset(&train_cfg, 0, sizeof(train_cfg));
    train_cfg.vocab_size = 280;
    train_cfg.min_frequency = 1;
    train_cfg.split_digits = 1;
    train_cfg.n_special_tokens = 1;
    train_cfg.special_tokens = specials;
    train_cfg.seed = 17U;
    CHECK_OK(gd_bpe_tokenizer_train(&corpus_path, 1, &train_cfg, &tok));
    CHECK_OK(gd_bpe_trainer_create(&train_cfg, &trainer));
    CHECK_OK(gd_bpe_trainer_add_text(trainer, (const uint8_t *)corpus_line0, strlen(corpus_line0)));
    CHECK_OK(gd_bpe_trainer_add_text(trainer, (const uint8_t *)corpus_line1, strlen(corpus_line1)));
    CHECK_OK(gd_bpe_trainer_add_text(trainer, (const uint8_t *)corpus_line2, strlen(corpus_line2)));
    CHECK_OK(gd_bpe_trainer_get_stats(trainer, &trainer_stats));
    CHECK(trainer_stats.texts == 3U, "streaming trainer text count");
    CHECK(trainer_stats.bytes == strlen(corpus), "streaming trainer byte count");
    CHECK(trainer_stats.pieces > 0U, "streaming trainer piece count");
    CHECK(trainer_stats.unique_pieces > 0U, "streaming trainer unique piece count");
    CHECK_OK(gd_bpe_trainer_finish(trainer, &streamed));
    gd_bpe_trainer_destroy(trainer);
    trainer = NULL;
    CHECK(gd_tokenizer_vocab_size(tok) > 256, "trained vocab includes specials/merges");
    CHECK(gd_tokenizer_vocab_size(streamed) == gd_tokenizer_vocab_size(tok), "streamed vocab size");
    CHECK(gd_tokenizer_hash(streamed) == gd_tokenizer_hash(tok), "streamed tokenizer hash");
    CHECK_OK(gd_tokenizer_id(tok, "<eot>", &special_id));
    CHECK(special_id >= 256, "special token id follows byte vocabulary");
    hash = gd_tokenizer_hash(tok);
    CHECK(hash != 0U, "tokenizer hash computed");

    CHECK_OK(gd_tokenizer_encode(tok, roundtrip, &ids, &n_ids));
    CHECK(n_ids > 0, "encoded tokens produced");
    CHECK(contains_id(ids, n_ids, special_id), "special token matched during encode");
    CHECK_OK(gd_tokenizer_decode(tok, ids, n_ids, &decoded));
    CHECK(strcmp(decoded, roundtrip) == 0, "decode roundtrip matches input");
    gd_tokenizer_free(decoded);
    decoded = NULL;

    CHECK_OK(gd_bpe_tokenizer_save(tok, tokenizer_path));
    load_cfg.split_digits = 1;
    load_cfg.allow_special = 1;
    CHECK_OK(gd_bpe_tokenizer_load(tokenizer_path, &load_cfg, &loaded));
    CHECK(gd_tokenizer_vocab_size(loaded) == gd_tokenizer_vocab_size(tok), "loaded vocab size");
    CHECK(gd_tokenizer_hash(loaded) == hash, "loaded hash");
    CHECK_OK(gd_tokenizer_encode(loaded, roundtrip, &literal_ids, &n_literal_ids));
    CHECK(contains_id(literal_ids, n_literal_ids, special_id), "loaded special token matched");
    gd_tokenizer_free(literal_ids);
    literal_ids = NULL;
    n_literal_ids = 0;

    no_special_cfg.split_digits = 1;
    no_special_cfg.allow_special = 0;
    CHECK_OK(gd_bpe_tokenizer_load(tokenizer_path, &no_special_cfg, &no_special));
    CHECK_OK(gd_tokenizer_encode(no_special, literal_special, &literal_ids, &n_literal_ids));
    CHECK(!contains_id(literal_ids, n_literal_ids, special_id), "--no-special encodes literal text");
    CHECK_OK(gd_tokenizer_decode(no_special, literal_ids, n_literal_ids, &decoded));
    CHECK(strcmp(decoded, literal_special) == 0, "literal special roundtrip");

    CHECK_STATUS(gd_tokenizer_decode(tok, &bad_id, 1, &decoded), GD_ERR_INVALID_ARGUMENT);

    gd_tokenizer_free(decoded);
    gd_tokenizer_free(literal_ids);
    gd_tokenizer_free(ids);
    gd_bpe_trainer_destroy(trainer);
    gd_tokenizer_destroy(no_special);
    gd_tokenizer_destroy(loaded);
    gd_tokenizer_destroy(streamed);
    gd_tokenizer_destroy(tok);
    (void)remove(tokenizer_path);
    (void)remove(corpus_path);
    printf("test_tokenizer: ok\n");
    return 0;
}
