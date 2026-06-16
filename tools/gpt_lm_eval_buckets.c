/* Temporary GPT-LM validation-loss bucket probe.
 *
 * Evaluates a checkpoint with target_ids masked to selected input-position
 * buckets.  This is useful for checking whether the 256-token sliding attention
 * window causes a loss cliff for tokens at positions >= 256.
 */

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../examples/gpt_lm/gpt_lm_shared.h"

#define static
#include "../examples/gpt_lm/gpt_lm_shared.c"
#undef static

#define DEFAULT_CHECKPOINT "examples/gpt_lm/checkpoints/gpt_lm_latest.gdckpt"
#define DEFAULT_DATA_DIR "examples/gpt_lm/data"

typedef struct bucket_cfg {
    int lo;
    int hi;
} bucket_cfg;

static const gd_dataset_field_spec GPT_BUCKET_FIELDS[] = {
    {
        .name = "input_ids",
        .dtype = GD_DTYPE_I32,
        .rank = 1,
        .shape = {-1},
        .collate = GD_GDDS_COLLATE_PACKED_SEQUENCE,
        .ragged_dim = 0,
        .source_field = -1,
    },
    {
        .name = "positions",
        .dtype = GD_DTYPE_I32,
        .rank = 1,
        .shape = {-1},
        .collate = GD_GDDS_COLLATE_PACKED_SEQUENCE,
        .ragged_dim = 0,
        .source_field = -1,
    },
    {
        .name = "target_ids",
        .dtype = GD_DTYPE_I32,
        .rank = 1,
        .shape = {-1},
        .collate = GD_GDDS_COLLATE_PACKED_SEQUENCE,
        .ragged_dim = 0,
        .source_field = -1,
    },
    {
        .name = "segment_lengths",
        .dtype = GD_DTYPE_I32,
        .rank = 1,
        .shape = {-1},
        .collate = GD_GDDS_COLLATE_PACKED_SEQUENCE,
        .ragged_dim = 0,
        .source_field = -1,
    },
    {
        .name = "cu_seqlens",
        .dtype = GD_DTYPE_I32,
        .rank = 1,
        .shape = {-1},
        .collate = GD_GDDS_COLLATE_GENERATED,
        .generated = GD_GDDS_GENERATED_CU_SEQLENS_FROM_LENGTHS,
        .source_field = 3,
    },
};

static gd_status bucket_transform(const gd_sample *src, gd_sample *dst, void *user_data)
{
    const bucket_cfg *cfg = (const bucket_cfg *)user_data;
    int input_idx;
    int pos_idx;
    int target_idx;
    int segment_idx;
    const int32_t *src_pos;
    const int32_t *src_target;
    int32_t *dst_target;
    int64_t shape[1];
    int64_t n;
    int64_t i;
    gd_status st;
    if (src == NULL || dst == NULL || cfg == NULL) { return GD_ERR_INVALID_ARGUMENT; }
    input_idx = gd_sample_field_index(src, "input_ids");
    pos_idx = gd_sample_field_index(src, "positions");
    target_idx = gd_sample_field_index(src, "target_ids");
    segment_idx = gd_sample_field_index(src, "segment_lengths");
    if (input_idx < 0 || pos_idx < 0 || target_idx < 0 || segment_idx < 0) { return GD_ERR_INVALID_ARGUMENT; }
    st = gd_sample_copy_field(dst, 0, src, input_idx);
    if (st != GD_OK) { return st; }
    st = gd_sample_copy_field(dst, 1, src, pos_idx);
    if (st != GD_OK) { return st; }
    st = gd_sample_copy_field(dst, 3, src, segment_idx);
    if (st != GD_OK) { return st; }
    if (gd_sample_field_dtype(src, target_idx) != GD_DTYPE_I32 || gd_sample_field_dtype(src, pos_idx) != GD_DTYPE_I32 ||
        gd_sample_field_rank(src, target_idx) != 1 || gd_sample_field_rank(src, pos_idx) != 1) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n = gd_sample_field_dim(src, target_idx, 0);
    if (n <= 0 || gd_sample_field_dim(src, pos_idx, 0) != n) { return GD_ERR_INVALID_ARGUMENT; }
    shape[0] = n;
    st = gd_sample_resize_field(dst, 2, GD_DTYPE_I32, 1, shape);
    if (st != GD_OK) { return st; }
    src_pos = (const int32_t *)gd_sample_field_data(src, pos_idx);
    src_target = (const int32_t *)gd_sample_field_data(src, target_idx);
    dst_target = (int32_t *)gd_sample_mutable_field_data(dst, 2);
    if (src_pos == NULL || src_target == NULL || dst_target == NULL) { return GD_ERR_INVALID_ARGUMENT; }
    for (i = 0; i < n; ++i) {
        const int p = (int)src_pos[i];
        const int t = (int)src_target[i];
        dst_target[i] = (t >= 0 && p >= cfg->lo && p <= cfg->hi) ? (int32_t)t : (int32_t)-1;
    }
    return GD_OK;
}

static int parse_i64_arg(const char *text, int64_t min_value, int64_t max_value, int64_t *out)
{
    char *end = NULL;
    long long parsed;
    if (text == NULL || text[0] == '\0' || out == NULL) { return 0; }
    errno = 0;
    parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) { return 0; }
    *out = (int64_t)parsed;
    return 1;
}

static int parse_float_arg(const char *text, float min_value, float max_value, float *out)
{
    char *end = NULL;
    float parsed;
    if (text == NULL || text[0] == '\0' || out == NULL) { return 0; }
    errno = 0;
    parsed = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed) || parsed < min_value || parsed > max_value) { return 0; }
    *out = parsed;
    return 1;
}

static const char *arg_value(int argc, char **argv, int *index, const char *name)
{
    const size_t name_len = strlen(name);
    const char *arg = argv[*index];
    if (strncmp(arg, name, name_len) == 0 && arg[name_len] == '=') { return arg + name_len + 1U; }
    if (strcmp(arg, name) == 0) {
        if (*index + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(2); }
        *index += 1;
        return argv[*index];
    }
    return NULL;
}

static bool metadata_value(const char *metadata, size_t metadata_len, const char *key, char *out, size_t out_size)
{
    const size_t key_len = strlen(key);
    size_t offset = 0U;
    if (metadata == NULL || key == NULL || out == NULL || out_size == 0U) { return false; }
    while (offset < metadata_len) {
        const size_t line_start = offset;
        size_t line_end = line_start;
        while (line_end < metadata_len && metadata[line_end] != '\n') { line_end += 1U; }
        if (line_end > line_start + key_len && strncmp(metadata + line_start, key, key_len) == 0 && metadata[line_start + key_len] == '=') {
            const size_t value_start = line_start + key_len + 1U;
            const size_t value_len = line_end - value_start;
            if (value_len >= out_size) { return false; }
            memcpy(out, metadata + value_start, value_len);
            out[value_len] = '\0';
            return true;
        }
        offset = line_end < metadata_len ? line_end + 1U : line_end;
    }
    return false;
}

static gpt_config default_config(void)
{
    gpt_config c;
    memset(&c, 0, sizeof(c));
    c.data_dir = DEFAULT_DATA_DIR;
    c.tokenizer_path = NULL;
    c.generate_prompt = NULL;
    c.checkpoint_path = DEFAULT_CHECKPOINT;
    c.val_split = "val";
    c.epochs = 0;
    c.batch_size = GPT_DEFAULT_BATCH_SIZE;
    c.n_layers = GPT_DEFAULT_LAYERS;
    c.lr_warmup_steps = -1;
    c.max_new_tokens = 1;
    c.epochs_set = true;
    c.save_best = false;
    c.save_latest = false;
    c.seed = GPT_DEFAULT_SEED;
    c.dropout_p = GPT_DEFAULT_DROPOUT_P;
    c.lr_max = GPT_DEFAULT_LR_MAX;
    c.lr_min = GPT_DEFAULT_LR_MIN;
    c.weight_decay = GPT_DEFAULT_WEIGHT_DECAY;
    c.grad_clip_norm = GPT_DEFAULT_GRAD_CLIP_NORM;
    c.logits_softcap = 0.0f;
    return c;
}

static uint64_t count_valid_targets(gd_batch *batch)
{
    const int idx = gd_batch_field_index(batch, "target_ids");
    const int32_t *target;
    size_t n;
    size_t i;
    uint64_t count = 0U;
    if (idx < 0) { return 0U; }
    target = (const int32_t *)gd_batch_host_data(batch, idx);
    n = gd_batch_field_nbytes(batch, idx) / sizeof(target[0]);
    if (target == NULL) { return 0U; }
    for (i = 0U; i < n; ++i) { if (target[i] >= 0) { count += 1U; } }
    return count;
}

static gd_status forward_batch(gd_context *ctx, gpt_lm *model, gd_batch *batch, gd_tensor *loss_out)
{
    gd_tensor *input_ids = gd_batch_tensor(batch, "input_ids");
    gd_tensor *target_ids = gd_batch_tensor(batch, "target_ids");
    gd_tensor *positions = gd_batch_tensor(batch, "positions");
    gd_tensor *cu_seqlens = gd_batch_tensor(batch, "cu_seqlens");
    if (input_ids == NULL || target_ids == NULL || positions == NULL || cu_seqlens == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gpt_lm_forward(ctx, model, input_ids, target_ids, positions, cu_seqlens, 0U, loss_out);
}

static double eval_bucket(gd_context *ctx,
                          gpt_lm *model,
                          const gpt_config *config,
                          const char *split,
                          int lo,
                          int hi,
                          uint64_t *valid_out,
                          uint64_t *batches_out)
{
    bucket_cfg bucket;
    gd_dataset_transform_config transform;
    gd_dataset *dataset = NULL;
    gd_dataloader *loader = NULL;
    gd_dataloader_config dl_cfg;
    uint64_t steps;
    uint64_t step;
    double weighted_loss = 0.0;
    uint64_t total_valid = 0U;
    uint64_t used_batches = 0U;
    bucket.lo = lo;
    bucket.hi = hi;
    transform.transform = bucket_transform;
    transform.user_data = &bucket;
    transform.output_fields = GPT_BUCKET_FIELDS;
    transform.n_output_fields = (int)(sizeof(GPT_BUCKET_FIELDS) / sizeof(GPT_BUCKET_FIELDS[0]));
    TRY(ctx, gd_dataset_open_gdds_split_with_transform(config->data_dir, split, &transform, &dataset));
    dl_cfg = gd_dataloader_config_default(config->batch_size);
    dl_cfg.num_workers = 1;
    dl_cfg.prefetch_factor = 2;
    TRY(ctx, gd_dataloader_create(ctx, dataset, NULL, &dl_cfg, &loader));
    TRY(ctx, gd_dataloader_prefetch(loader));
    steps = gd_dataloader_steps_per_epoch(loader);
    gd_module_set_training(&model->mod, false);
    for (step = 0U; step < steps; ++step) {
        gd_batch *batch = NULL;
        uint64_t valid;
        gd_tensor loss;
        float loss_value = NAN;
        TRY(ctx, gd_dataloader_next(loader, &batch));
        valid = count_valid_targets(batch);
        if (valid > 0U) {
            TRY(ctx, gd_begin_step(ctx, GD_SCOPE_EVAL, batch));
            TRY(ctx, forward_batch(ctx, model, batch, &loss));
            TRY(ctx, gd_end_step(ctx));
            TRY(ctx, gd_tensor_read_f32(ctx, &loss, &loss_value, 1U));
            if (isfinite(loss_value)) {
                weighted_loss += (double)loss_value * (double)valid;
                total_valid += valid;
                used_batches += 1U;
            }
        }
        TRY(ctx, gd_dataloader_release(loader, batch));
    }
    gd_dataloader_destroy(loader);
    gd_dataset_destroy(dataset);
    if (valid_out != NULL) { *valid_out = total_valid; }
    if (batches_out != NULL) { *batches_out = used_batches; }
    return total_valid > 0U ? weighted_loss / (double)total_valid : NAN;
}

static void usage(const char *argv0)
{
    printf("usage: %s [--checkpoint PATH] [--data-dir PATH] [--split val] [--batch-size N]\n", argv0);
}

int main(int argc, char **argv)
{
    gpt_config config = default_config();
    const char *split = "val";
    bool layers_set = false;
    bool softcap_set = false;
    int i;
    char *metadata = NULL;
    size_t metadata_len = 0U;
    gd_memory_config mem;
    gd_context *ctx = NULL;
    gpt_lm model;
    gd_module_load_options load_options;
    gd_status st;
    const int buckets[][2] = {
        {0, 511}, {0, 15}, {16, 31}, {32, 63}, {64, 127}, {128, 255}, {256, 511},
    };
    const char *labels[] = {"all", "000-015", "016-031", "032-063", "064-127", "128-255", "256-511"};
    memset(&model, 0, sizeof(model));
    for (i = 1; i < argc; ++i) {
        const char *value;
        int64_t parsed_i64;
        float parsed_f32;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) { usage(argv[0]); return 0; }
        value = arg_value(argc, argv, &i, "--checkpoint");
        if (value != NULL) { config.checkpoint_path = value; continue; }
        value = arg_value(argc, argv, &i, "--data-dir");
        if (value != NULL) { config.data_dir = value; continue; }
        value = arg_value(argc, argv, &i, "--split");
        if (value != NULL) { split = value; continue; }
        value = arg_value(argc, argv, &i, "--batch-size");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, 1024, &parsed_i64)) { fprintf(stderr, "invalid batch size\n"); return 2; }
            config.batch_size = (int)parsed_i64;
            continue;
        }
        value = arg_value(argc, argv, &i, "--layers");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, 96, &parsed_i64)) { fprintf(stderr, "invalid layers\n"); return 2; }
            config.n_layers = (int)parsed_i64;
            layers_set = true;
            continue;
        }
        value = arg_value(argc, argv, &i, "--logits-softcap");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 1000000.0f, &parsed_f32)) { fprintf(stderr, "invalid softcap\n"); return 2; }
            config.logits_softcap = parsed_f32;
            softcap_set = true;
            continue;
        }
        fprintf(stderr, "unknown argument: %s\n", argv[i]);
        return 2;
    }
    st = gd_checkpoint_read_metadata(config.checkpoint_path, &metadata, &metadata_len);
    if (st == GD_OK) {
        char value[128];
        int64_t parsed_i64;
        float parsed_f32;
        if (!layers_set && metadata_value(metadata, metadata_len, "n_layers", value, sizeof(value)) && parse_i64_arg(value, 1, 96, &parsed_i64)) {
            config.n_layers = (int)parsed_i64;
        }
        if (!softcap_set && metadata_value(metadata, metadata_len, "logits_softcap", value, sizeof(value)) && parse_float_arg(value, 0.0f, 1000000.0f, &parsed_f32)) {
            config.logits_softcap = parsed_f32;
        }
    }
    mem = gpt_memory_config(&config);
    st = gd_context_create(&mem, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("gpt_lm_eval_buckets: skipped (no supported gradients.c backend)\n");
        free(metadata);
        return 0;
    }
    if (st != GD_OK) { gpt_fail_status(ctx, st, "gd_context_create", __LINE__); }
    gpt_lm_init(ctx, &model, &config);
    load_options.strict = true;
    load_options.load_buffers = true;
    TRY(ctx, gd_module_load_state(ctx, &model.mod, config.checkpoint_path, &load_options));
    TRY(ctx, gd_context_seal_params(ctx));
    printf("eval_buckets: checkpoint=%s data_dir=%s split=%s batch=%d layers=%d logits_softcap=%.3f\n",
           config.checkpoint_path,
           config.data_dir,
           split,
           config.batch_size,
           config.n_layers,
           (double)config.logits_softcap);
    if (metadata != NULL) {
        char value[128];
        if (metadata_value(metadata, metadata_len, "epoch", value, sizeof(value))) { printf("metadata: epoch=%s", value); }
        if (metadata_value(metadata, metadata_len, "val_loss", value, sizeof(value))) { printf(" val_loss=%s", value); }
        if (metadata_value(metadata, metadata_len, "best_val_loss", value, sizeof(value))) { printf(" best_val_loss=%s", value); }
        printf("\n");
    }
    printf("bucket valid_targets batches loss_nats bits_per_token ppl\n");
    for (i = 0; i < (int)(sizeof(buckets) / sizeof(buckets[0])); ++i) {
        uint64_t valid = 0U;
        uint64_t batches = 0U;
        double loss = eval_bucket(ctx, &model, &config, split, buckets[i][0], buckets[i][1], &valid, &batches);
        printf("%-7s %12llu %7llu %.6f %.4f %.3f\n",
               labels[i],
               (unsigned long long)valid,
               (unsigned long long)batches,
               loss,
               loss / log(2.0),
               exp(loss));
    }
    gpt_lm_deinit(&model);
    gd_context_destroy(ctx);
    free(metadata);
    return 0;
}
