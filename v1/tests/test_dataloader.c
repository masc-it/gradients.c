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

static int copy_lm_batch(gd_context *ctx,
                         gd_batch *batch,
                         int32_t *tokens,
                         int32_t *targets,
                         int32_t *positions,
                         int n_elem)
{
    gd_tensor *tok = gd_batch_tensor(batch, "tokens");
    gd_tensor *tgt = gd_batch_tensor(batch, "targets");
    gd_tensor *pos = gd_batch_tensor(batch, "positions");
    CHECK_TRUE(tok != NULL && tgt != NULL && pos != NULL);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, tok, tokens, (size_t)n_elem * sizeof(int32_t)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, tgt, targets, (size_t)n_elem * sizeof(int32_t)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, pos, positions, (size_t)n_elem * sizeof(int32_t)));
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

static void make_lm_fields(gd_batch_field_desc *fields, int batch_size, int block_len)
{
    memset(fields, 0, 3U * sizeof(fields[0]));
    fields[0].name = "tokens";
    fields[0].dtype = GD_DTYPE_I32;
    fields[0].rank = 2;
    fields[0].sizes[0] = batch_size;
    fields[0].sizes[1] = block_len;
    fields[1].name = "targets";
    fields[1].dtype = GD_DTYPE_I32;
    fields[1].rank = 2;
    fields[1].sizes[0] = batch_size;
    fields[1].sizes[1] = block_len;
    fields[2].name = "positions";
    fields[2].dtype = GD_DTYPE_I32;
    fields[2].rank = 2;
    fields[2].sizes[0] = batch_size;
    fields[2].sizes[1] = block_len;
}

static int make_loader_ex(gd_context *ctx,
                          gd_dataset *ds,
                          gd_sampler_mode sampler,
                          uint64_t seed,
                          int batch_size,
                          int num_workers,
                          int prefetch_factor,
                          gd_dataloader **out)
{
    gd_dataloader_config cfg;
    gd_batch_field_desc fields[3];
    memset(&cfg, 0, sizeof(cfg));
    make_lm_fields(fields, batch_size, 4);
    cfg.batch_size = batch_size;
    cfg.seed = seed;
    cfg.device = (gd_device){GD_DEVICE_CPU, 0};
    cfg.sampler = sampler;
    cfg.expected_dataset_fingerprint = gd_dataset_fingerprint(ds);
    cfg.num_workers = num_workers;
    cfg.prefetch_factor = prefetch_factor;
    CHECK_OK(gd_dataloader_create(ctx, ds, &cfg, fields, 3,
                                  gd_collate_gdtok_lm, NULL, out));
    return 0;
}

static int make_loader(gd_context *ctx,
                       gd_dataset *ds,
                       gd_sampler_mode sampler,
                       uint64_t seed,
                       gd_dataloader **out)
{
    return make_loader_ex(ctx, ds, sampler, seed, 2, 0, 0, out);
}

static int test_sequential_loader_matches_shard_payload(void)
{
    const char *corpus_path = "/tmp/gd_loader_seq.txt";
    const char *tokenizer_path = "/tmp/gd_loader_seq_tok.json";
    const char *paths[1];
    gd_dataset_build_result build;
    gd_dataset *ds = NULL;
    gd_context *ctx = NULL;
    gd_dataloader *dl = NULL;
    gd_batch *batch = NULL;
    int32_t payload[9];
    int32_t tokens[8];
    int32_t targets[8];
    int32_t positions[8];
    uint64_t block_len = 0U;
    int i;

    CHECK_TRUE(build_fixture_dataset(corpus_path, tokenizer_path, "/tmp/gd_loader_seq_ds", &build) == 0);
    paths[0] = build.train.shard_path;
    CHECK_OK(gd_dataset_open_gdtok(paths, 1, &ds));
    CHECK_TRUE(gd_dataset_num_samples(ds) == build.train.n_samples);
    CHECK_OK(gd_dataset_get_u64(ds, "block_len", &block_len));
    CHECK_TRUE(block_len == 4U);
    CHECK_OK(gd_context_create(&ctx));
    CHECK_TRUE(make_loader(ctx, ds, GD_SAMPLER_SEQUENTIAL, 123U, &dl) == 0);
    CHECK_OK(gd_dataloader_prefetch(dl));
    CHECK_OK(gd_dataloader_next(dl, &batch));
    CHECK_TRUE(gd_batch_get_state(batch) == GD_BATCH_IN_USE);
    CHECK_TRUE(copy_lm_batch(ctx, batch, tokens, targets, positions, 8) == 0);
    CHECK_TRUE(read_shard_tokens(build.train.shard_path, payload, 9) == 0);
    for (i = 0; i < 8; ++i) {
        CHECK_TRUE(tokens[i] == payload[i]);
        CHECK_TRUE(targets[i] == payload[i + 1]);
        CHECK_TRUE(positions[i] == i % 4);
    }
    CHECK_OK(gd_dataloader_release(dl, batch));

    gd_dataloader_destroy(dl);
    gd_context_destroy(ctx);
    gd_dataset_destroy(ds);
    gd_dataset_build_result_clear(&build);
    (void)remove(corpus_path);
    (void)remove(tokenizer_path);
    return 0;
}

static int test_slot_lifecycle(void)
{
    const char *corpus_path = "/tmp/gd_loader_slots.txt";
    const char *tokenizer_path = "/tmp/gd_loader_slots_tok.json";
    const char *paths[1];
    gd_dataset_build_result build;
    gd_dataset *ds = NULL;
    gd_context *ctx = NULL;
    gd_dataloader *dl = NULL;
    gd_batch *a = NULL;
    gd_batch *b = NULL;

    CHECK_TRUE(build_fixture_dataset(corpus_path, tokenizer_path, "/tmp/gd_loader_slots_ds", &build) == 0);
    paths[0] = build.train.shard_path;
    CHECK_OK(gd_dataset_open_gdtok(paths, 1, &ds));
    CHECK_OK(gd_context_create(&ctx));
    CHECK_TRUE(make_loader(ctx, ds, GD_SAMPLER_SEQUENTIAL, 7U, &dl) == 0);
    CHECK_TRUE(gd_dataloader_slot_count(dl) == 2);

    CHECK_OK(gd_dataloader_prefetch(dl));
    CHECK_OK(gd_dataloader_prefetch(dl));
    CHECK_OK(gd_dataloader_next(dl, &a));
    CHECK_OK(gd_dataloader_next(dl, &b));
    CHECK_TRUE(gd_batch_get_state(a) == GD_BATCH_IN_USE);
    CHECK_TRUE(gd_batch_get_state(b) == GD_BATCH_IN_USE);
    CHECK_STATUS(gd_dataloader_prefetch(dl), GD_ERR_INVALID_STATE);
    CHECK_OK(gd_dataloader_release(dl, a));
    CHECK_OK(gd_dataloader_prefetch(dl));
    CHECK_OK(gd_dataloader_release(dl, b));

    gd_dataloader_destroy(dl);
    gd_context_destroy(ctx);
    gd_dataset_destroy(ds);
    gd_dataset_build_result_clear(&build);
    (void)remove(corpus_path);
    (void)remove(tokenizer_path);
    return 0;
}

static int test_worker_prefetch_factor(void)
{
    const char *corpus_path = "/tmp/gd_loader_workers.txt";
    const char *tokenizer_path = "/tmp/gd_loader_workers_tok.json";
    const char *paths[1];
    gd_dataset_build_result build;
    gd_dataset *ds = NULL;
    gd_context *ctx = NULL;
    gd_dataloader *dl = NULL;
    gd_batch *batch = NULL;
    int i;

    CHECK_TRUE(build_fixture_dataset(corpus_path, tokenizer_path, "/tmp/gd_loader_workers_ds", &build) == 0);
    paths[0] = build.train.shard_path;
    CHECK_OK(gd_dataset_open_gdtok(paths, 1, &ds));
    CHECK_OK(gd_context_create(&ctx));
    CHECK_TRUE(make_loader_ex(ctx, ds, GD_SAMPLER_SEQUENTIAL, 11U, 1, 3, 2, &dl) == 0);
    CHECK_TRUE(gd_dataloader_slot_count(dl) == 6);

    for (i = 0; i < 6; ++i) {
        CHECK_OK(gd_dataloader_prefetch(dl));
    }
    for (i = 0; i < 6; ++i) {
        CHECK_OK(gd_dataloader_next(dl, &batch));
        CHECK_TRUE(gd_batch_get_state(batch) == GD_BATCH_IN_USE);
        CHECK_OK(gd_dataloader_release(dl, batch));
    }

    gd_dataloader_destroy(dl);
    gd_context_destroy(ctx);
    gd_dataset_destroy(ds);
    gd_dataset_build_result_clear(&build);
    (void)remove(corpus_path);
    (void)remove(tokenizer_path);
    return 0;
}

static int test_random_repro_state_metrics_and_fingerprint(void)
{
    const char *corpus_path = "/tmp/gd_loader_rand.txt";
    const char *tokenizer_path = "/tmp/gd_loader_rand_tok.json";
    const char *state_path = "/tmp/gd_loader_state.bin";
    const char *paths[1];
    gd_dataset_build_result build;
    gd_dataset *ds = NULL;
    gd_context *ctx = NULL;
    gd_dataloader *a = NULL;
    gd_dataloader *b = NULL;
    gd_dataloader *c = NULL;
    gd_batch *batch_a = NULL;
    gd_batch *batch_b = NULL;
    gd_batch *batch_c = NULL;
    int32_t ta[8];
    int32_t tb[8];
    int32_t tc[8];
    int32_t dummy[8];
    gd_dataloader_metrics metrics;
    gd_dataloader_config bad_cfg;
    gd_batch_field_desc fields[3];

    CHECK_TRUE(build_fixture_dataset(corpus_path, tokenizer_path, "/tmp/gd_loader_rand_ds", &build) == 0);
    paths[0] = build.train.shard_path;
    CHECK_OK(gd_dataset_open_gdtok(paths, 1, &ds));
    CHECK_OK(gd_context_create(&ctx));
    CHECK_TRUE(make_loader(ctx, ds, GD_SAMPLER_RANDOM_REPLACEMENT, 999U, &a) == 0);
    CHECK_TRUE(make_loader(ctx, ds, GD_SAMPLER_RANDOM_REPLACEMENT, 999U, &b) == 0);
    CHECK_OK(gd_dataloader_next(a, &batch_a));
    CHECK_OK(gd_dataloader_next(b, &batch_b));
    CHECK_TRUE(copy_lm_batch(ctx, batch_a, ta, dummy, dummy, 8) == 0);
    CHECK_TRUE(copy_lm_batch(ctx, batch_b, tb, dummy, dummy, 8) == 0);
    CHECK_TRUE(arrays_equal(ta, tb, 8));
    CHECK_OK(gd_dataloader_release(a, batch_a));
    CHECK_OK(gd_dataloader_release(b, batch_b));

    CHECK_OK(gd_dataloader_state_save(a, state_path));
    CHECK_OK(gd_dataloader_next(a, &batch_a));
    CHECK_TRUE(copy_lm_batch(ctx, batch_a, ta, dummy, dummy, 8) == 0);
    CHECK_OK(gd_dataloader_release(a, batch_a));

    CHECK_TRUE(make_loader(ctx, ds, GD_SAMPLER_RANDOM_REPLACEMENT, 1U, &c) == 0);
    CHECK_OK(gd_dataloader_state_load(c, state_path));
    CHECK_OK(gd_dataloader_next(c, &batch_c));
    CHECK_TRUE(copy_lm_batch(ctx, batch_c, tc, dummy, dummy, 8) == 0);
    CHECK_TRUE(arrays_equal(ta, tc, 8));
    CHECK_OK(gd_dataloader_release(c, batch_c));

    gd_dataloader_metrics_get(a, &metrics);
    CHECK_TRUE(metrics.batches_returned >= 2U);
    CHECK_TRUE(metrics.batches_prepared >= 2U);
    CHECK_TRUE(metrics.samples_prepared >= 4U);

    memset(&bad_cfg, 0, sizeof(bad_cfg));
    make_lm_fields(fields, 2, 4);
    bad_cfg.batch_size = 2;
    bad_cfg.device = (gd_device){GD_DEVICE_CPU, 0};
    bad_cfg.sampler = GD_SAMPLER_RANDOM_REPLACEMENT;
    bad_cfg.expected_dataset_fingerprint = gd_dataset_fingerprint(ds) ^ 1U;
    CHECK_STATUS(gd_dataloader_create(ctx, ds, &bad_cfg, fields, 3,
                                      gd_collate_gdtok_lm, NULL, &c),
                 GD_ERR_INVALID_ARGUMENT);

    gd_dataloader_destroy(a);
    gd_dataloader_destroy(b);
    gd_dataloader_destroy(c);
    gd_context_destroy(ctx);
    gd_dataset_destroy(ds);
    gd_dataset_build_result_clear(&build);
    (void)remove(corpus_path);
    (void)remove(tokenizer_path);
    (void)remove(state_path);
    return 0;
}

typedef struct toy_dataset {
    float features[4][3];
    int32_t labels[4];
} toy_dataset;

static uint64_t toy_num_samples(const void *impl)
{
    (void)impl;
    return 4U;
}

static uint64_t toy_fingerprint(const void *impl)
{
    (void)impl;
    return UINT64_C(0x12345678abcdef00);
}

static gd_status toy_collate(gd_dataset *dataset,
                             const uint64_t *sample_ids,
                             int batch_size,
                             gd_batch *batch,
                             void *user_data)
{
    toy_dataset *toy = (toy_dataset *)gd_dataset_data(dataset);
    int f_idx = gd_batch_field_index(batch, "features");
    int y_idx = gd_batch_field_index(batch, "labels");
    float *features = (float *)gd_batch_host_data(batch, f_idx);
    int32_t *labels = (int32_t *)gd_batch_host_data(batch, y_idx);
    int b;
    (void)user_data;
    if (toy == NULL || features == NULL || labels == NULL || batch_size != 2) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (b = 0; b < batch_size; ++b) {
        uint64_t id = sample_ids[b];
        int d;
        if (id >= 4U) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        for (d = 0; d < 3; ++d) {
            features[(size_t)b * 3U + (size_t)d] = toy->features[id][d];
        }
        labels[b] = toy->labels[id];
    }
    return GD_OK;
}

static int test_custom_classification_dataset(void)
{
    toy_dataset toy = {
        {{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F},
         {7.0F, 8.0F, 9.0F}, {10.0F, 11.0F, 12.0F}},
        {0, 1, 1, 0}
    };
    gd_dataset_ops ops = {"toy-classification", toy_num_samples, toy_fingerprint, NULL, NULL};
    gd_dataset *ds = NULL;
    gd_context *ctx = NULL;
    gd_dataloader *dl = NULL;
    gd_batch *batch = NULL;
    gd_dataloader_config cfg;
    gd_batch_field_desc fields[2];
    float features[6];
    int32_t labels[2];

    CHECK_OK(gd_dataset_create(&ops, &toy, &ds));
    CHECK_OK(gd_context_create(&ctx));
    memset(fields, 0, sizeof(fields));
    fields[0].name = "features";
    fields[0].dtype = GD_DTYPE_F32;
    fields[0].rank = 2;
    fields[0].sizes[0] = 2;
    fields[0].sizes[1] = 3;
    fields[1].name = "labels";
    fields[1].dtype = GD_DTYPE_I32;
    fields[1].rank = 1;
    fields[1].sizes[0] = 2;
    memset(&cfg, 0, sizeof(cfg));
    cfg.batch_size = 2;
    cfg.device = (gd_device){GD_DEVICE_CPU, 0};
    cfg.sampler = GD_SAMPLER_SEQUENTIAL;
    cfg.expected_dataset_fingerprint = gd_dataset_fingerprint(ds);
    CHECK_OK(gd_dataloader_create(ctx, ds, &cfg, fields, 2, toy_collate, NULL, &dl));
    CHECK_OK(gd_dataloader_next(dl, &batch));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gd_batch_tensor(batch, "features"),
                                   features, sizeof(features)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gd_batch_tensor(batch, "labels"),
                                   labels, sizeof(labels)));
    CHECK_TRUE(features[0] == 1.0F && features[5] == 6.0F);
    CHECK_TRUE(labels[0] == 0 && labels[1] == 1);
    CHECK_OK(gd_dataloader_release(dl, batch));

    gd_dataloader_destroy(dl);
    gd_context_destroy(ctx);
    gd_dataset_destroy(ds);
    return 0;
}

int main(void)
{
    CHECK_TRUE(test_sequential_loader_matches_shard_payload() == 0);
    CHECK_TRUE(test_slot_lifecycle() == 0);
    CHECK_TRUE(test_worker_prefetch_factor() == 0);
    CHECK_TRUE(test_random_repro_state_metrics_and_fingerprint() == 0);
    CHECK_TRUE(test_custom_classification_dataset() == 0);
    return 0;
}
