#include "gd_example_config.h"

#include <gradients/gradients.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define MNIST_INPUT_DIM 784
#define MNIST_HIDDEN_DIM 128
#define MNIST_CLASSES 10
#define MNIST_MAX_BATCH_SIZE 100000
#define MNIST_MAX_DATALOADER_WORKERS 64
#define MNIST_MAX_DATALOADER_PREFETCH_FACTOR 16

typedef struct mnist_mlp {
    gd_module mod;
    gd_linear_layer fc1;
    gd_linear_layer fc2;
    float dropout_p;
} mnist_mlp;

typedef struct mnist_config {
    const char *config_path;
    char *data_dir;
    int train_epochs;
    int report_every;
    int train_batch;
    int eval_batch;
    int dataloader_workers;
    int dataloader_prefetch_factor;
    uint64_t train_seed;
    uint64_t dropout_seed;
    float dropout_p;
    float lr_max;
    float lr_min;
    int lr_warmup_steps; /* -1 means auto after total_steps is known. */
    float weight_decay;
    float amp_init_scale;
    uint32_t amp_growth_interval;
    float amp_max_scale;

    /* Derived once after validation; downstream code assumes these are valid. */
    size_t data_slot_bytes;
    uint32_t data_slots;
    size_t eval_logits_count;
    size_t eval_target_bytes;
} mnist_config;

typedef struct mnist_transform_state {
    uint16_t u8_to_f16[256];
} mnist_transform_state;

static const gd_dataset_field_spec MNIST_TRANSFORM_FIELDS[] = {
    {
        .name = "image",
        .dtype = GD_DTYPE_F16,
        .rank = 1,
        .shape = {MNIST_INPUT_DIM},
        .collate = GD_GDDS_COLLATE_STACK,
    },
    {
        .name = "target",
        .dtype = GD_DTYPE_I32,
        .rank = 0,
        .shape = {0},
        .collate = GD_GDDS_COLLATE_STACK,
    },
};

static void fail_status(gd_context *ctx, gd_status st, const char *expr, int line)
{
    fprintf(stderr,
            "mlp_mnist failed at line %d: %s -> %s (%s)\n",
            line,
            expr,
            gd_status_string(st),
            ctx != NULL ? gd_context_error(ctx) : "no context");
    exit(1);
}

#define TRY(ctx, expr)                                                        \
    do {                                                                      \
        gd_status _st = (expr);                                               \
        if (_st != GD_OK) {                                                   \
            fail_status((ctx), _st, #expr, __LINE__);                         \
        }                                                                     \
    } while (0)

static void print_usage(const char *argv0)
{
    printf("usage: %s --config PATH\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  --config, -c PATH   YAML configuration file\n");
    printf("  --help, -h          show this help\n");
}

static const char *arg_value(int argc, char **argv, int *index, const char *name)
{
    const size_t name_len = strlen(name);
    const char *arg = argv[*index];
    if (strncmp(arg, name, name_len) == 0 && arg[name_len] == '=') {
        return arg + name_len + 1U;
    }
    if (strcmp(arg, name) == 0) {
        if (*index + 1 >= argc) {
            fprintf(stderr, "mlp_mnist: missing value for %s\n", name);
            exit(2);
        }
        *index += 1;
        return argv[*index];
    }
    return NULL;
}

static const char *parse_config_path(int argc, char **argv)
{
    const char *config_path = NULL;
    int i;
    for (i = 1; i < argc; ++i) {
        const char *value;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        value = arg_value(argc, argv, &i, "--config");
        if (value == NULL && strcmp(argv[i], "-c") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "mlp_mnist: missing value for -c\n");
                exit(2);
            }
            ++i;
            value = argv[i];
        }
        if (value == NULL && argv[i][0] != '-') {
            value = argv[i];
        }
        if (value != NULL) {
            if (config_path != NULL) {
                fprintf(stderr, "mlp_mnist: --config specified more than once\n");
                exit(2);
            }
            config_path = value;
            continue;
        }
        fprintf(stderr, "mlp_mnist: unknown argument %s\n", argv[i]);
        print_usage(argv[0]);
        exit(2);
    }
    if (config_path == NULL || config_path[0] == '\0') {
        fprintf(stderr, "mlp_mnist: --config PATH is required\n");
        print_usage(argv[0]);
        exit(2);
    }
    return config_path;
}

static char *mnist_strdup(const char *text)
{
    const size_t len = text != NULL ? strlen(text) : 0U;
    char *out;
    if (text == NULL || len > SIZE_MAX - 1U) {
        return NULL;
    }
    out = (char *)malloc(len + 1U);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, text, len + 1U);
    return out;
}

static void mnist_config_deinit(mnist_config *config)
{
    if (config == NULL) {
        return;
    }
    free(config->data_dir);
    config->data_dir = NULL;
}

static void mnist_config_die(const char *path, const gd_example_config_error *error)
{
    if (error != NULL && error->line > 0U) {
        fprintf(stderr,
                "mlp_mnist: invalid config %s:%u: %s\n",
                path != NULL ? path : "(null)",
                error->line,
                gd_example_config_error_message(error));
    } else {
        fprintf(stderr,
                "mlp_mnist: invalid config %s: %s\n",
                path != NULL ? path : "(null)",
                gd_example_config_error_message(error));
    }
    exit(2);
}

static size_t mnist_max_size(size_t a, size_t b)
{
    return a > b ? a : b;
}

static size_t mnist_align_up_size(size_t value, size_t alignment)
{
    return ((value + alignment - 1U) / alignment) * alignment;
}

static void mnist_config_invalid(mnist_config *config, const char *message)
{
    fprintf(stderr,
            "mlp_mnist: invalid config %s: %s\n",
            config->config_path != NULL ? config->config_path : "(null)",
            message != NULL ? message : "invalid value");
    mnist_config_deinit(config);
    exit(2);
}

static void mnist_config_finalize(mnist_config *config)
{
    const size_t sample_bytes = (size_t)MNIST_INPUT_DIM * sizeof(uint16_t) + sizeof(int32_t);
    const size_t max_batch = mnist_max_size((size_t)config->train_batch,
                                            (size_t)config->eval_batch);
    const uint64_t requested_slots = (uint64_t)config->dataloader_workers *
                                         (uint64_t)config->dataloader_prefetch_factor +
                                     1U;
    const size_t slot_padding = 64U * 1024U;
    size_t batch_bytes;
    size_t slot_payload_bytes;

    if (config->data_dir == NULL || config->data_dir[0] == '\0') {
        mnist_config_invalid(config, "data_dir must not be empty");
    }
    if (config->lr_min > config->lr_max) {
        mnist_config_invalid(config, "lr_min must be <= lr_max");
    }
    if (config->amp_growth_interval == 0U) {
        mnist_config_invalid(config, "amp_growth_interval must be > 0");
    }
    if (config->amp_max_scale < config->amp_init_scale) {
        mnist_config_invalid(config, "amp_max_scale must be >= amp_init_scale");
    }
    if (sample_bytes > SIZE_MAX / max_batch) {
        mnist_config_invalid(config, "train_batch/eval_batch is too large for batch buffers");
    }
    batch_bytes = sample_bytes * max_batch;
    if (batch_bytes > SIZE_MAX - slot_padding) {
        mnist_config_invalid(config, "train_batch/eval_batch is too large for data slots");
    }
    slot_payload_bytes = batch_bytes + slot_padding;
    if (mnist_max_size(1U * 1024U * 1024U, slot_payload_bytes) > SIZE_MAX - 255U) {
        mnist_config_invalid(config, "data slot size is too large");
    }
    if (requested_slots > (uint64_t)UINT32_MAX) {
        mnist_config_invalid(config, "dataloader_workers * dataloader_prefetch_factor is too large");
    }
    if ((size_t)config->eval_batch > SIZE_MAX / (size_t)MNIST_CLASSES) {
        mnist_config_invalid(config, "eval_batch is too large for logits buffer");
    }

    config->data_slot_bytes = mnist_align_up_size(mnist_max_size(1U * 1024U * 1024U,
                                                                 slot_payload_bytes),
                                                  256U);
    config->data_slots = (uint32_t)requested_slots;
    if (config->data_slots < 4U) {
        config->data_slots = 4U;
    }
    config->eval_logits_count = (size_t)config->eval_batch * (size_t)MNIST_CLASSES;
    if (config->eval_logits_count > SIZE_MAX / sizeof(float) ||
        (size_t)config->eval_batch > SIZE_MAX / sizeof(int32_t)) {
        mnist_config_invalid(config, "eval_batch is too large for evaluation buffers");
    }
    config->eval_target_bytes = (size_t)config->eval_batch * sizeof(int32_t);
}

static mnist_config mnist_config_from_yaml(const char *path)
{
    static const char *const known_keys[] = {
        "data_dir",
        "train_epochs",
        "report_every",
        "train_batch",
        "eval_batch",
        "dataloader_workers",
        "dataloader_prefetch_factor",
        "train_seed",
        "dropout_seed",
        "dropout_p",
        "lr_max",
        "lr_min",
        "lr_warmup_steps",
        "weight_decay",
        "amp_init_scale",
        "amp_growth_interval",
        "amp_max_scale",
    };
    gd_example_config_doc doc;
    gd_example_config_error error;
    mnist_config config;
    const char *data_dir = NULL;
    uint64_t parsed_u64 = 0U;
    memset(&config, 0, sizeof(config));
    config.config_path = path;
    if (!gd_example_config_load_yaml_file(path, &doc, &error) ||
        !gd_example_config_validate_keys(&doc, known_keys, GD_ARRAY_LEN(known_keys), &error) ||
        !gd_example_config_require_string(&doc, "data_dir", &data_dir, &error) ||
        !gd_example_config_require_int(&doc, "train_epochs", 1, 1000000, &config.train_epochs, &error) ||
        !gd_example_config_require_int(&doc, "report_every", 0, 1000000, &config.report_every, &error) ||
        !gd_example_config_require_int(&doc, "train_batch", 1, MNIST_MAX_BATCH_SIZE, &config.train_batch, &error) ||
        !gd_example_config_require_int(&doc, "eval_batch", 1, MNIST_MAX_BATCH_SIZE, &config.eval_batch, &error) ||
        !gd_example_config_require_int(&doc,
                                       "dataloader_workers",
                                       1,
                                       MNIST_MAX_DATALOADER_WORKERS,
                                       &config.dataloader_workers,
                                       &error) ||
        !gd_example_config_require_int(&doc,
                                       "dataloader_prefetch_factor",
                                       1,
                                       MNIST_MAX_DATALOADER_PREFETCH_FACTOR,
                                       &config.dataloader_prefetch_factor,
                                       &error) ||
        !gd_example_config_require_u64(&doc, "train_seed", UINT64_MAX, &config.train_seed, &error) ||
        !gd_example_config_require_u64(&doc, "dropout_seed", UINT64_MAX, &config.dropout_seed, &error) ||
        !gd_example_config_require_f32(&doc, "dropout_p", 0.0f, 0.95f, &config.dropout_p, &error) ||
        !gd_example_config_require_f32(&doc, "lr_max", 0.0f, 100.0f, &config.lr_max, &error) ||
        !gd_example_config_require_f32(&doc, "lr_min", 0.0f, 100.0f, &config.lr_min, &error) ||
        !gd_example_config_require_int(&doc, "lr_warmup_steps", -1, 1000000000, &config.lr_warmup_steps, &error) ||
        !gd_example_config_require_f32(&doc, "weight_decay", 0.0f, 100.0f, &config.weight_decay, &error) ||
        !gd_example_config_require_f32(&doc, "amp_init_scale", 1.0f, 1000000000.0f, &config.amp_init_scale, &error) ||
        !gd_example_config_require_u64(&doc,
                                       "amp_growth_interval",
                                       (uint64_t)UINT32_MAX,
                                       &parsed_u64,
                                       &error) ||
        !gd_example_config_require_f32(&doc, "amp_max_scale", 1.0f, 1000000000.0f, &config.amp_max_scale, &error)) {
        gd_example_config_doc_free(&doc);
        mnist_config_die(path, &error);
    }
    config.amp_growth_interval = (uint32_t)parsed_u64;
    config.data_dir = mnist_strdup(data_dir);
    gd_example_config_doc_free(&doc);
    if (config.data_dir == NULL) {
        fprintf(stderr, "mlp_mnist: out of memory while storing data_dir from %s\n", path);
        exit(2);
    }
    mnist_config_finalize(&config);
    return config;
}

static uint16_t mnist_f32_to_f16_bits(float value)
{
    union {
        float f;
        uint32_t u;
    } v;
    uint32_t sign;
    int32_t exp;
    uint32_t mant;
    uint32_t out_exp;
    uint32_t out_mant;
    v.f = value;
    sign = (v.u >> 16) & 0x8000U;
    exp = (int32_t)((v.u >> 23) & 0xffU) - 127;
    mant = v.u & 0x7fffffU;
    if (((v.u >> 23) & 0xffU) == 0xffU) {
        return (uint16_t)(sign | (mant == 0U ? 0x7c00U : 0x7e00U));
    }
    if (exp > 15) {
        return (uint16_t)(sign | 0x7c00U);
    }
    if (exp < -14) {
        uint32_t shifted;
        uint32_t remainder;
        uint32_t halfway;
        int32_t shift = -14 - exp;
        if (shift > 24) {
            return (uint16_t)sign;
        }
        mant |= 0x800000U;
        shifted = mant >> (uint32_t)(shift + 13);
        remainder = mant & ((1U << (uint32_t)(shift + 13)) - 1U);
        halfway = 1U << (uint32_t)(shift + 12);
        if (remainder > halfway || (remainder == halfway && (shifted & 1U) != 0U)) {
            shifted += 1U;
        }
        return (uint16_t)(sign | shifted);
    }
    out_exp = (uint32_t)(exp + 15);
    out_mant = mant >> 13;
    {
        uint32_t remainder = mant & 0x1fffU;
        if (remainder > 0x1000U || (remainder == 0x1000U && (out_mant & 1U) != 0U)) {
            out_mant += 1U;
            if (out_mant == 0x400U) {
                out_mant = 0U;
                out_exp += 1U;
                if (out_exp >= 31U) {
                    return (uint16_t)(sign | 0x7c00U);
                }
            }
        }
    }
    return (uint16_t)(sign | (out_exp << 10) | out_mant);
}

static void mnist_transform_state_init(mnist_transform_state *state)
{
    int i;
    if (state == NULL) {
        return;
    }
    for (i = 0; i < 256; ++i) {
        state->u8_to_f16[i] = mnist_f32_to_f16_bits((float)i * (1.0f / 255.0f));
    }
}

static gd_status mnist_u8_normalize_transform(const gd_sample *src,
                                              gd_sample *dst,
                                              void *user_data)
{
    const mnist_transform_state *state = (const mnist_transform_state *)user_data;
    const uint8_t *restrict image_u8;
    uint16_t *restrict image_f16;
    size_t i;
    if (state == NULL || src == NULL || dst == NULL || gd_sample_field_count(src) < 2 ||
        gd_sample_field_count(dst) < 2) {
        return GD_ERR_INVALID_ARGUMENT;
    }

    image_u8 = (const uint8_t *)gd_sample_field_data(src, 0);
    image_f16 = (uint16_t *)gd_sample_mutable_field_data(dst, 0);
    if (image_u8 == NULL || image_f16 == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i + 8U <= (size_t)MNIST_INPUT_DIM; i += 8U) {
        image_f16[i + 0U] = state->u8_to_f16[image_u8[i + 0U]];
        image_f16[i + 1U] = state->u8_to_f16[image_u8[i + 1U]];
        image_f16[i + 2U] = state->u8_to_f16[image_u8[i + 2U]];
        image_f16[i + 3U] = state->u8_to_f16[image_u8[i + 3U]];
        image_f16[i + 4U] = state->u8_to_f16[image_u8[i + 4U]];
        image_f16[i + 5U] = state->u8_to_f16[image_u8[i + 5U]];
        image_f16[i + 6U] = state->u8_to_f16[image_u8[i + 6U]];
        image_f16[i + 7U] = state->u8_to_f16[image_u8[i + 7U]];
    }
    for (; i < (size_t)MNIST_INPUT_DIM; ++i) {
        image_f16[i] = state->u8_to_f16[image_u8[i]];
    }
    return gd_sample_copy_field(dst, 1, src, 1);
}

static double wall_seconds(void)
{
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0) {
        return 0.0;
    }
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

static gd_memory_config mnist_memory_config(const mnist_config *config)
{
    return (gd_memory_config){
        .params_bytes = 4U * 1024U * 1024U,
        .state_bytes = 16U * 1024U * 1024U,
        .scratch_slot_bytes = 32U * 1024U * 1024U,
        .data_slot_bytes = config->data_slot_bytes,
        .scratch_slots = 3U,
        .data_slots = config->data_slots,
        .default_alignment = 256U,
    };
}

static gd_adamw_config mnist_adamw_config(const mnist_config *config, float lr)
{
    gd_adamw_config cfg = gd_adamw_config_default();
    cfg.lr = lr;
    cfg.weight_decay = config->weight_decay;
    return cfg;
}

static gd_amp_config mnist_amp_config(const mnist_config *config)
{
    gd_amp_config cfg = gd_amp_config_default();
    cfg.init_scale = config->amp_init_scale;
    cfg.growth_interval = config->amp_growth_interval;
    cfg.max_scale = config->amp_max_scale;
    return cfg;
}

static gd_linear_layer_config mnist_linear_config(int64_t in_features,
                                                  int64_t out_features,
                                                  uint64_t seed,
                                                  float limit)
{
    gd_linear_layer_config cfg = gd_linear_layer_config_make(in_features,
                                                             out_features,
                                                             GD_DTYPE_F16,
                                                             seed);
    cfg.weight_low = -limit;
    cfg.weight_high = limit;
    return cfg;
}

static void mnist_mlp_init(gd_context *ctx, mnist_mlp *model, float dropout_p)
{
    const gd_linear_layer_config fc1_cfg = mnist_linear_config(MNIST_INPUT_DIM,
                                                               MNIST_HIDDEN_DIM,
                                                               101U,
                                                               0.088f);
    const gd_linear_layer_config fc2_cfg = mnist_linear_config(MNIST_HIDDEN_DIM,
                                                               MNIST_CLASSES,
                                                               202U,
                                                               0.217f);
    memset(model, 0, sizeof(*model));
    model->dropout_p = dropout_p;
    TRY(ctx, gd_module_init(ctx, &model->mod, "mnist_mlp"));
    TRY(ctx, gd_linear_layer_init_child(ctx, &model->mod, "fc1", &model->fc1, &fc1_cfg));
    TRY(ctx, gd_linear_layer_init_child(ctx, &model->mod, "fc2", &model->fc2, &fc2_cfg));
}

static void mnist_mlp_deinit(mnist_mlp *model)
{
    gd_linear_layer_deinit(&model->fc2);
    gd_linear_layer_deinit(&model->fc1);
    gd_module_deinit(&model->mod);
}

static gd_status mnist_mlp_forward(gd_context *ctx,
                                   mnist_mlp *model,
                                   const gd_tensor *x,
                                   uint64_t dropout_seed,
                                   gd_tensor *out)
{
    gd_tensor hidden;
    gd_tensor activated;
    gd_tensor dropped;
    gd_status st;
    st = gd_linear_layer_forward(ctx, &model->fc1, x, &hidden);
    if (st != GD_OK) {
        return st;
    }
    st = gd_relu(ctx, &hidden, &activated);
    if (st != GD_OK) {
        return st;
    }
    st = gd_dropout(ctx,
                    &activated,
                    model->dropout_p,
                    model->mod.training,
                    dropout_seed,
                    &dropped);
    if (st != GD_OK) {
        return st;
    }
    return gd_linear_layer_forward(ctx, &model->fc2, &dropped, out);
}

static void print_param_set(const gd_param_set *params)
{
    uint32_t i;
    printf("parameters:\n");
    for (i = 0U; i < params->count; ++i) {
        printf("  %-24s lr_mult=%.2f weight_decay=%.4f trainable=%s\n",
               params->items[i].path,
               (double)params->items[i].lr_mult,
               (double)params->items[i].weight_decay,
               params->items[i].trainable ? "yes" : "no");
    }
}

static gd_status create_gdds_loader(gd_context *ctx,
                                    gd_dataset *dataset,
                                    gd_sampler *sampler,
                                    int batch_size,
                                    const mnist_config *config,
                                    gd_dataloader **out)
{
    gd_dataloader_config cfg;
    gd_status st;
    *out = NULL;
    cfg = gd_dataloader_config_default(batch_size);
    cfg.num_workers = config->dataloader_workers;
    cfg.prefetch_factor = config->dataloader_prefetch_factor;
    st = gd_dataloader_create(ctx, dataset, sampler, &cfg, out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_dataloader_prefetch(*out);
    if (st != GD_OK) {
        gd_dataloader_destroy(*out);
        *out = NULL;
        return st;
    }
    return GD_OK;
}

static gd_tensor *required_batch_tensor(gd_context *ctx, gd_batch *batch, const char *name, int line)
{
    gd_tensor *tensor = gd_batch_tensor(batch, name);
    if (tensor == NULL) {
        fail_status(ctx, GD_ERR_INVALID_ARGUMENT, name, line);
    }
    return tensor;
}

static int argmax10(const float *values)
{
    int best = 0;
    int i;
    for (i = 1; i < MNIST_CLASSES; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

static void evaluate_accuracy(gd_context *ctx,
                              mnist_mlp *model,
                              gd_dataloader *loader,
                              const mnist_config *config,
                              size_t n_samples,
                              int *correct_out,
                              size_t *total_out)
{
    const size_t eval_batch = (size_t)config->eval_batch;
    const size_t logits_count = config->eval_logits_count;
    float *logits = NULL;
    int32_t *labels = NULL;
    size_t seen = 0U;
    int correct = 0;
    *correct_out = 0;
    *total_out = 0U;
    if (n_samples == 0U) {
        return;
    }
    logits = (float *)malloc(logits_count * sizeof(logits[0]));
    labels = (int32_t *)malloc(config->eval_target_bytes);
    if (logits == NULL || labels == NULL) {
        free(logits);
        free(labels);
        fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "evaluate_accuracy malloc", __LINE__);
    }
    while (seen < n_samples) {
        gd_batch *batch = NULL;
        gd_tensor *image;
        gd_tensor *target;
        gd_tensor pred;
        size_t remaining = n_samples - seen;
        size_t rows = remaining < eval_batch ? remaining : eval_batch;
        size_t row;
        TRY(ctx, gd_dataloader_next(loader, &batch));
        TRY(ctx, gd_begin_step(ctx, GD_SCOPE_EVAL, batch));
        image = required_batch_tensor(ctx, batch, "image", __LINE__);
        target = required_batch_tensor(ctx, batch, "target", __LINE__);
        TRY(ctx, mnist_mlp_forward(ctx, model, image, 0U, &pred));
        TRY(ctx, gd_end_step(ctx));
        TRY(ctx, gd_tensor_read_f32(ctx, &pred, logits, logits_count));
        TRY(ctx, gd_tensor_read(ctx, target, labels, config->eval_target_bytes));
        TRY(ctx, gd_dataloader_release(loader, batch));
        TRY(ctx, gd_dataloader_prefetch(loader));
        for (row = 0U; row < rows; ++row) {
            int predicted = argmax10(&logits[row * (size_t)MNIST_CLASSES]);
            if (predicted == labels[row]) {
                correct += 1;
            }
        }
        seen += rows;
    }
    free(labels);
    free(logits);
    *correct_out = correct;
    *total_out = n_samples;
}

static void train_mnist(gd_context *ctx,
                        mnist_mlp *model,
                        gd_dataloader *loader,
                        gd_optimizer *optimizer,
                        gd_amp_scaler *scaler,
                        const gd_lr_scheduler_config *lr_config,
                        const mnist_config *config,
                        size_t steps_per_epoch,
                        size_t train_steps)
{
    double last_report_time;
    size_t last_report_step = 0U;
    const size_t report_every = (size_t)config->report_every;
    gd_module_set_training(&model->mod, true);
    last_report_time = wall_seconds();
    for (size_t step = 0U; step < train_steps; ++step) {
        gd_batch *batch = NULL;
        gd_tensor *image;
        gd_tensor *target;
        gd_tensor logits;
        gd_tensor loss;
        float lr = 0.0f;
        const size_t current_step = step + 1U;
        const size_t epoch = step / steps_per_epoch + 1U;
        const size_t epoch_step = step % steps_per_epoch + 1U;
        const int report = report_every > 0U &&
                           (step == 0U || current_step % report_every == 0U ||
                            current_step == train_steps);
        TRY(ctx, gd_lr_scheduler_value(lr_config, (uint64_t)step, &lr));
        TRY(ctx, gd_dataloader_next(loader, &batch));
        TRY(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, batch));
        image = required_batch_tensor(ctx, batch, "image", __LINE__);
        target = required_batch_tensor(ctx, batch, "target", __LINE__);
        TRY(ctx, mnist_mlp_forward(ctx,
                                    model,
                                    image,
                                    config->dropout_seed ^ (uint64_t)current_step,
                                    &logits));
        TRY(ctx, gd_cross_entropy(ctx, &logits, target, &loss));
        TRY(ctx, gd_backward_amp(ctx, &loss, NULL, scaler));
        TRY(ctx, gd_optimizer_step_amp_lr(ctx, optimizer, scaler, lr));
        TRY(ctx, gd_end_step(ctx));
        TRY(ctx, gd_dataloader_release(loader, batch));
        TRY(ctx, gd_dataloader_prefetch(loader));

        if (report) {
            const double now = wall_seconds();
            const double elapsed = now - last_report_time;
            const size_t batches = current_step - last_report_step;
            const double batches_per_sec = elapsed > 0.0 ? (double)batches / elapsed : 0.0;
            float loss_value = 0.0f;
            TRY(ctx, gd_tensor_item(ctx, &loss, &loss_value));
            printf("epoch=%zu/%d batch=%zu/%zu step=%zu loss=%.6f lr=%.6g batch/s=%.2f\n",
                   epoch,
                   config->train_epochs,
                   epoch_step,
                   steps_per_epoch,
                   current_step,
                   (double)loss_value,
                   (double)lr,
                   batches_per_sec);
            last_report_time = now;
            last_report_step = current_step;
        }
    }
}

int main(int argc, char **argv)
{
    mnist_config config = mnist_config_from_yaml(parse_config_path(argc, argv));
    const gd_memory_config mem = mnist_memory_config(&config);
    gd_context *ctx = NULL;
    gd_status st = gd_context_create(&mem, &ctx);
    mnist_transform_state transform_state;
    gd_dataset_transform_config transform_cfg;
    gd_dataset *train_dataset = NULL;
    gd_dataset *test_dataset = NULL;
    gd_sampler *train_sampler = NULL;
    gd_dataloader *train_loader = NULL;
    gd_dataloader *test_loader = NULL;
    mnist_mlp model = {0};
    gd_param_set params = {0};
    gd_lr_scheduler_config lr_config = {0};
    gd_optimizer *optimizer = NULL;
    gd_amp_scaler *scaler = NULL;
    int correct = 0;
    int exit_code = 1;
    size_t train_samples = 0U;
    size_t test_samples = 0U;
    size_t total = 0U;
    size_t steps_per_epoch = 0U;
    size_t samples_per_epoch = 0U;
    size_t train_steps = 0U;
    size_t eval_samples = 0U;
    size_t optimizer_steps = 0U;
    float accuracy = 0.0f;

    if (st == GD_ERR_UNSUPPORTED) {
        printf("mlp_mnist: skipped (no supported gradients.c backend)\n");
        exit_code = 0;
        goto cleanup;
    }
    if (st != GD_OK) {
        fail_status(ctx, st, "gd_context_create", __LINE__);
    }

    mnist_transform_state_init(&transform_state);
    transform_cfg = (gd_dataset_transform_config){
        .transform = mnist_u8_normalize_transform,
        .user_data = &transform_state,
        .output_fields = MNIST_TRANSFORM_FIELDS,
        .n_output_fields = (int)GD_ARRAY_LEN(MNIST_TRANSFORM_FIELDS),
    };
    TRY(ctx, gd_dataset_open_gdds_split_with_transform(config.data_dir,
                                                       "train",
                                                       &transform_cfg,
                                                       &train_dataset));
    TRY(ctx, gd_dataset_open_gdds_split_with_transform(config.data_dir,
                                                       "test",
                                                       &transform_cfg,
                                                       &test_dataset));
    train_samples = (size_t)gd_dataset_num_samples(train_dataset);
    test_samples = (size_t)gd_dataset_num_samples(test_dataset);

    mnist_mlp_init(ctx, &model, config.dropout_p);
    {
        const gd_param_group groups[] = {
            {
                .name = "encoder",
                .match = "mnist_mlp.fc1.*",
                .lr_mult = 1.0f,
                .weight_decay = config.weight_decay,
                .trainable = true,
            },
            {
                .name = "classifier",
                .match = "mnist_mlp.fc2.*",
                .lr_mult = 1.0f,
                .weight_decay = config.weight_decay,
                .trainable = true,
            },
        };
        TRY(ctx, gd_module_collect_params(ctx, &model.mod, groups, GD_ARRAY_LEN(groups), &params));
    }
    print_param_set(&params);

    {
        const gd_adamw_config adam = mnist_adamw_config(&config, config.lr_max);
        const gd_amp_config amp = mnist_amp_config(&config);
        TRY(ctx, gd_adamw_create(ctx, &params, &adam, &optimizer));
        TRY(ctx, gd_amp_scaler_create(ctx, &amp, &scaler));
    }
    TRY(ctx, gd_context_seal_params(ctx));
    TRY(ctx, gd_sampler_create_random(train_dataset, config.train_seed, &train_sampler));
    TRY(ctx, create_gdds_loader(ctx,
                                train_dataset,
                                train_sampler,
                                config.train_batch,
                                &config,
                                &train_loader));
    steps_per_epoch = (size_t)gd_dataloader_steps_per_epoch(train_loader);
    if (steps_per_epoch == 0U || (size_t)config.train_epochs > SIZE_MAX / steps_per_epoch) {
        fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "train_epochs", __LINE__);
    }
    samples_per_epoch = (size_t)gd_dataloader_samples_per_epoch(train_loader);
    train_steps = (size_t)config.train_epochs * steps_per_epoch;
    lr_config = gd_lr_scheduler_config_default();
    lr_config.max_lr = config.lr_max;
    lr_config.min_lr = config.lr_min;
    lr_config.total_steps = (uint64_t)train_steps;
    if (config.lr_warmup_steps >= 0) {
        lr_config.warmup_steps = (uint64_t)config.lr_warmup_steps;
    } else {
        const size_t auto_warmup = train_steps < 100U ? train_steps / 10U : 100U;
        lr_config.warmup_steps = (uint64_t)auto_warmup;
    }
    {
        float initial_lr = 0.0f;
        TRY(ctx, gd_lr_scheduler_value(&lr_config, 0U, &initial_lr));
        (void)initial_lr;
    }
    printf("config: path=%s\n", config.config_path);
    printf("dataset: dir=%s train=%zu test=%zu storage=u8 transform=f16_normalize train_batch=%d eval_batch=%d epochs=%d dropout_p=%.3f steps_per_epoch=%zu samples_per_epoch=%zu total_steps=%zu\n",
           config.data_dir,
           train_samples,
           test_samples,
           config.train_batch,
           config.eval_batch,
           config.train_epochs,
           (double)config.dropout_p,
           steps_per_epoch,
           samples_per_epoch,
           train_steps);
    printf("dataloader: workers=%d prefetch_factor=%d slots=%d\n",
           config.dataloader_workers,
           config.dataloader_prefetch_factor,
           gd_dataloader_slot_count(train_loader));
    printf("optim: cosine_lr max=%.6g min=%.6g warmup=%llu total=%llu weight_decay=%.4g amp_scale=%.4g\n",
           (double)lr_config.max_lr,
           (double)lr_config.min_lr,
           (unsigned long long)lr_config.warmup_steps,
           (unsigned long long)lr_config.total_steps,
           (double)config.weight_decay,
           (double)gd_amp_scaler_scale(scaler));

    train_mnist(ctx,
                &model,
                train_loader,
                optimizer,
                scaler,
                &lr_config,
                &config,
                steps_per_epoch,
                train_steps);

    gd_dataloader_destroy(train_loader);
    train_loader = NULL;

    gd_module_set_training(&model.mod, false);
    TRY(ctx, create_gdds_loader(ctx,
                                test_dataset,
                                NULL,
                                config.eval_batch,
                                &config,
                                &test_loader));
    eval_samples = (size_t)gd_dataloader_samples_per_epoch(test_loader);
    evaluate_accuracy(ctx, &model, test_loader, &config, eval_samples, &correct, &total);
    accuracy = total > 0U ? (float)correct / (float)total : 0.0f;
    optimizer_steps = (size_t)gd_optimizer_step_count(optimizer);
    printf("test_accuracy=%.4f correct=%d/%zu optimizer_steps=%zu\n",
           (double)accuracy,
           correct,
           total,
           optimizer_steps);

    printf("mlp_mnist: ok\n");
    exit_code = 0;

cleanup:
    gd_dataloader_destroy(test_loader);
    gd_dataloader_destroy(train_loader);
    gd_sampler_destroy(train_sampler);
    gd_dataset_destroy(test_dataset);
    gd_dataset_destroy(train_dataset);
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(optimizer);
    gd_param_set_free(&params);
    mnist_mlp_deinit(&model);
    gd_context_destroy(ctx);
    mnist_config_deinit(&config);
    return exit_code;
}
