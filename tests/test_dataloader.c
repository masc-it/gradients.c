#include "gradients/gradients.h"

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

static int build_fixture_dataset(const char *corpus_path,
                                 const char *tokenizer_path,
                                 const char *out_dir,
                                 gd_dataset_build_result *result)
{
    const char *inputs[1];
    gd_dataset_build_config cfg;
    const char *corpus =
        "<|im_start|>alpha alpha alpha alpha one one one one<|im_end|>\n"
        "<|im_start|>beta beta beta beta two two two two<|im_end|>\n"
        "<|im_start|>gamma gamma gamma gamma three three three three<|im_end|>\n"
        "<|im_start|>delta delta delta delta four four four four<|im_end|>\n"
        "<|im_start|>epsilon epsilon epsilon epsilon five five five five<|im_end|>\n"
        "<|im_start|>zeta zeta zeta zeta six six six six<|im_end|>\n";

    memset(&cfg, 0, sizeof(cfg));
    memset(result, 0, sizeof(*result));
    CHECK_TRUE(write_file(corpus_path, corpus) == 0);
    CHECK_TRUE(train_tokenizer(corpus_path, tokenizer_path) == 0);
    inputs[0] = corpus_path;
    cfg.tokenizer_path = tokenizer_path;
    cfg.input_paths = inputs;
    cfg.n_input_paths = 1;
    cfg.output_dir = out_dir;
    cfg.block_len = 4;
    cfg.train_ratio = 1.0;
    cfg.val_ratio = 0.0;
    cfg.seed = 42U;
    cfg.no_shuffle_split = 1;
    cfg.im_start = "<|im_start|>";
    cfg.im_end = "<|im_end|>";
    CHECK_OK(gd_dataset_build(&cfg, result));
    CHECK_TRUE(result->train.n_samples >= 4U);
    return 0;
}

static int read_shard_tokens(const char *path, int32_t *out, int n)
{
    gd_gdtok_header h;
    FILE *f;
    int i;
    CHECK_OK(gd_gdtok_read_header(path, &h));
    f = fopen(path, "rb");
    if (f == NULL) {
        return 1;
    }
    if (fseek(f, (long)h.payload_offset, SEEK_SET) != 0) {
        (void)fclose(f);
        return 1;
    }
    for (i = 0; i < n; ++i) {
        if (h.dtype == GD_GDTOK_DTYPE_U16) {
            unsigned char b[2];
            if (fread(b, 1U, sizeof(b), f) != sizeof(b)) {
                (void)fclose(f);
                return 1;
            }
            out[i] = (int32_t)((uint32_t)b[0] | ((uint32_t)b[1] << 8U));
        } else {
            unsigned char b[4];
            if (fread(b, 1U, sizeof(b), f) != sizeof(b)) {
                (void)fclose(f);
                return 1;
            }
            out[i] = (int32_t)((uint32_t)b[0] | ((uint32_t)b[1] << 8U) |
                               ((uint32_t)b[2] << 16U) | ((uint32_t)b[3] << 24U));
        }
    }
    if (fclose(f) != 0) {
        return 1;
    }
    return 0;
}

static int copy_slot(gd_context *ctx,
                     gd_batch_slot *slot,
                     int32_t *tokens,
                     int32_t *targets,
                     int32_t *positions,
                     int n_elem)
{
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, slot->tokens, tokens,
                                   (size_t)n_elem * sizeof(int32_t)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, slot->targets, targets,
                                   (size_t)n_elem * sizeof(int32_t)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, slot->positions, positions,
                                   (size_t)n_elem * sizeof(int32_t)));
    return 0;
}

static int arrays_equal(const int32_t *a, const int32_t *b, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}

static int make_loader(gd_context *ctx,
                       gd_token_dataset *ds,
                       gd_dataloader_mode mode,
                       uint64_t seed,
                       gd_dataloader **out)
{
    gd_dataloader_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.batch_size = 2;
    cfg.block_len = 4;
    cfg.seed = seed;
    cfg.shuffle = mode == GD_DATALOADER_RANDOM ? 1 : 0;
    cfg.double_buffer = 1;
    cfg.device = (gd_device){GD_DEVICE_CPU, 0};
    cfg.mode = mode;
    cfg.expected_tokenizer_hash = gd_token_dataset_tokenizer_hash(ds);
    CHECK_OK(gd_dataloader_create(ctx, ds, &cfg, out));
    return 0;
}

static int test_sequential_loader_matches_shard_payload(void)
{
    const char *corpus_path = "/tmp/gd_loader_seq.txt";
    const char *tokenizer_path = "/tmp/gd_loader_seq_tok.json";
    const char *paths[1];
    gd_dataset_build_result build;
    gd_token_dataset *ds = NULL;
    gd_context *ctx = NULL;
    gd_dataloader *dl = NULL;
    gd_batch_slot *slot = NULL;
    int32_t payload[9];
    int32_t tokens[8];
    int32_t targets[8];
    int32_t positions[8];
    int i;

    CHECK_TRUE(build_fixture_dataset(corpus_path, tokenizer_path, "/tmp/gd_loader_seq_ds", &build) == 0);
    paths[0] = build.train.shard_path;
    CHECK_OK(gd_token_dataset_open(paths, 1, &ds));
    CHECK_TRUE(gd_token_dataset_num_samples(ds) == build.train.n_samples);
    CHECK_TRUE(gd_token_dataset_block_len(ds) == 4U);
    CHECK_OK(gd_context_create(&ctx));
    CHECK_TRUE(make_loader(ctx, ds, GD_DATALOADER_SEQUENTIAL, 123U, &dl) == 0);
    CHECK_OK(gd_dataloader_prefetch(dl));
    CHECK_OK(gd_dataloader_next(dl, &slot));
    CHECK_TRUE(slot->state == GD_BATCH_SLOT_IN_USE);
    CHECK_TRUE(copy_slot(ctx, slot, tokens, targets, positions, 8) == 0);
    CHECK_TRUE(read_shard_tokens(build.train.shard_path, payload, 9) == 0);
    for (i = 0; i < 8; ++i) {
        CHECK_TRUE(tokens[i] == payload[i]);
        CHECK_TRUE(targets[i] == payload[i + 1]);
        CHECK_TRUE(positions[i] == i % 4);
    }
    CHECK_OK(gd_dataloader_release_slot(dl, slot));

    gd_dataloader_destroy(dl);
    gd_context_destroy(ctx);
    gd_token_dataset_close(ds);
    gd_dataset_build_result_clear(&build);
    (void)remove(corpus_path);
    (void)remove(tokenizer_path);
    return 0;
}

static int test_double_buffer_slot_states(void)
{
    const char *corpus_path = "/tmp/gd_loader_slots.txt";
    const char *tokenizer_path = "/tmp/gd_loader_slots_tok.json";
    const char *paths[1];
    gd_dataset_build_result build;
    gd_token_dataset *ds = NULL;
    gd_context *ctx = NULL;
    gd_dataloader *dl = NULL;
    gd_batch_slot *a = NULL;
    gd_batch_slot *b = NULL;

    CHECK_TRUE(build_fixture_dataset(corpus_path, tokenizer_path, "/tmp/gd_loader_slots_ds", &build) == 0);
    paths[0] = build.train.shard_path;
    CHECK_OK(gd_token_dataset_open(paths, 1, &ds));
    CHECK_OK(gd_context_create(&ctx));
    CHECK_TRUE(make_loader(ctx, ds, GD_DATALOADER_SEQUENTIAL, 7U, &dl) == 0);

    CHECK_OK(gd_dataloader_prefetch(dl));
    CHECK_OK(gd_dataloader_prefetch(dl));
    CHECK_OK(gd_dataloader_next(dl, &a));
    CHECK_OK(gd_dataloader_next(dl, &b));
    CHECK_TRUE(a != b);
    CHECK_TRUE(a->state == GD_BATCH_SLOT_IN_USE);
    CHECK_TRUE(b->state == GD_BATCH_SLOT_IN_USE);
    CHECK_STATUS(gd_dataloader_prefetch(dl), GD_ERR_INVALID_STATE);
    CHECK_OK(gd_dataloader_release_slot(dl, a));
    CHECK_OK(gd_dataloader_prefetch(dl));
    CHECK_OK(gd_dataloader_release_slot(dl, b));

    gd_dataloader_destroy(dl);
    gd_context_destroy(ctx);
    gd_token_dataset_close(ds);
    gd_dataset_build_result_clear(&build);
    (void)remove(corpus_path);
    (void)remove(tokenizer_path);
    return 0;
}

static int test_random_determinism_state_and_metrics(void)
{
    const char *corpus_path = "/tmp/gd_loader_rand.txt";
    const char *tokenizer_path = "/tmp/gd_loader_rand_tok.json";
    const char *state_path = "/tmp/gd_loader_state.json";
    const char *paths[1];
    gd_dataset_build_result build;
    gd_token_dataset *ds = NULL;
    gd_context *ctx = NULL;
    gd_dataloader *a = NULL;
    gd_dataloader *b = NULL;
    gd_dataloader *c = NULL;
    gd_batch_slot *slot_a = NULL;
    gd_batch_slot *slot_b = NULL;
    gd_batch_slot *slot_c = NULL;
    int32_t a_tokens[8];
    int32_t b_tokens[8];
    int32_t tmp[8];
    gd_dataloader_metrics metrics;
    gd_dataloader_config bad_cfg;

    CHECK_TRUE(build_fixture_dataset(corpus_path, tokenizer_path, "/tmp/gd_loader_rand_ds", &build) == 0);
    paths[0] = build.train.shard_path;
    CHECK_OK(gd_token_dataset_open(paths, 1, &ds));
    CHECK_OK(gd_context_create(&ctx));
    CHECK_TRUE(make_loader(ctx, ds, GD_DATALOADER_RANDOM, 999U, &a) == 0);
    CHECK_TRUE(make_loader(ctx, ds, GD_DATALOADER_RANDOM, 999U, &b) == 0);

    CHECK_OK(gd_dataloader_next(a, &slot_a));
    CHECK_OK(gd_dataloader_next(b, &slot_b));
    CHECK_TRUE(copy_slot(ctx, slot_a, a_tokens, tmp, tmp, 8) == 0);
    CHECK_TRUE(copy_slot(ctx, slot_b, b_tokens, tmp, tmp, 8) == 0);
    CHECK_TRUE(arrays_equal(a_tokens, b_tokens, 8));
    CHECK_OK(gd_dataloader_release_slot(a, slot_a));
    CHECK_OK(gd_dataloader_release_slot(b, slot_b));

    CHECK_OK(gd_dataloader_state_save(a, state_path));
    CHECK_OK(gd_dataloader_next(a, &slot_a));
    CHECK_TRUE(copy_slot(ctx, slot_a, a_tokens, tmp, tmp, 8) == 0);
    CHECK_OK(gd_dataloader_release_slot(a, slot_a));

    CHECK_TRUE(make_loader(ctx, ds, GD_DATALOADER_RANDOM, 1U, &c) == 0);
    CHECK_OK(gd_dataloader_state_load(c, state_path));
    CHECK_OK(gd_dataloader_next(c, &slot_c));
    CHECK_TRUE(copy_slot(ctx, slot_c, b_tokens, tmp, tmp, 8) == 0);
    CHECK_TRUE(arrays_equal(a_tokens, b_tokens, 8));
    CHECK_OK(gd_dataloader_release_slot(c, slot_c));

    gd_dataloader_metrics_get(a, &metrics);
    CHECK_TRUE(metrics.batches_prepared >= 2U);
    CHECK_TRUE(metrics.batches_returned >= 2U);
    CHECK_TRUE(metrics.samples_prepared >= 4U);

    memset(&bad_cfg, 0, sizeof(bad_cfg));
    bad_cfg.batch_size = 2;
    bad_cfg.block_len = 4;
    bad_cfg.seed = 1U;
    bad_cfg.shuffle = 1;
    bad_cfg.double_buffer = 1;
    bad_cfg.device = (gd_device){GD_DEVICE_CPU, 0};
    bad_cfg.mode = GD_DATALOADER_RANDOM;
    bad_cfg.expected_tokenizer_hash = gd_token_dataset_tokenizer_hash(ds) + 1U;
    CHECK_STATUS(gd_dataloader_create(ctx, ds, &bad_cfg, &c), GD_ERR_INVALID_ARGUMENT);

    gd_dataloader_destroy(a);
    gd_dataloader_destroy(b);
    gd_dataloader_destroy(c);
    gd_context_destroy(ctx);
    gd_token_dataset_close(ds);
    gd_dataset_build_result_clear(&build);
    (void)remove(corpus_path);
    (void)remove(tokenizer_path);
    (void)remove(state_path);
    return 0;
}

int main(void)
{
    if (test_sequential_loader_matches_shard_payload() != 0) {
        return 1;
    }
    if (test_double_buffer_slot_states() != 0) {
        return 1;
    }
    if (test_random_determinism_state_and_metrics() != 0) {
        return 1;
    }
    return 0;
}
