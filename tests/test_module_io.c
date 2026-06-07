#include <gradients/gradients.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_module_io failed: %s (%s:%d)\n",           \
                    (msg), __FILE__, __LINE__);                                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static gd_memory_config module_io_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 1U << 20;
    cfg.state_bytes = 1U << 20;
    cfg.scratch_slot_bytes = 1U << 20;
    cfg.data_slot_bytes = 1U << 16;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static void init_toy_module(gd_context *ctx,
                            gd_module *root,
                            gd_module *child,
                            gd_tensor *weight,
                            gd_tensor *running,
                            gd_init_spec weight_init,
                            gd_init_spec buffer_init)
{
    const int64_t weight_shape[2] = {2, 3};
    const int64_t buffer_shape[1] = {2};
    gd_tensor_spec weight_spec;
    gd_tensor_spec buffer_spec;
    weight_spec = gd_tensor_spec_make(GD_DTYPE_F32, gd_shape_make(2U, weight_shape), 256U);
    buffer_spec = gd_tensor_spec_make(GD_DTYPE_F32, gd_shape_make(1U, buffer_shape), 256U);
    CHECK_OK(gd_module_init(ctx, root, "toy"));
    CHECK_OK(gd_module_init_child(ctx, root, "block", child));
    CHECK_OK(gd_module_param(ctx, child, "weight", &weight_spec, &weight_init, weight));
    CHECK_OK(gd_module_buffer(ctx, child, "running", &buffer_spec, &buffer_init, running));
}

static void check_f32_array(const float *got, const float *expect, uint32_t count, const char *message)
{
    uint32_t i;
    for (i = 0U; i < count; ++i) {
        CHECK(abs_f32(got[i] - expect[i]) < 1.0e-6f, message);
    }
}

static void test_module_state_roundtrip(gd_context *ctx)
{
    const char *path = "build/test_module_io.ckpt";
    const char metadata[] = "model=toy\nepoch=3\nval_loss=1.25\n";
    const float weight_values[6] = {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f};
    const float buffer_values[2] = {0.25f, -0.75f};
    float got_weight[6];
    float got_buffer[2];
    char *loaded_metadata = NULL;
    size_t loaded_metadata_len = 0U;
    gd_module root_a;
    gd_module child_a;
    gd_tensor weight_a;
    gd_tensor running_a;
    gd_module root_b;
    gd_module child_b;
    gd_tensor weight_b;
    gd_tensor running_b;
    gd_module_save_options save_options;
    gd_module_load_options load_options;

    (void)remove(path);
    init_toy_module(ctx, &root_a, &child_a, &weight_a, &running_a, gd_init_zero(), gd_init_zero());
    CHECK_OK(gd_tensor_write_f32(ctx, &weight_a, weight_values, GD_ARRAY_LEN(weight_values)));
    CHECK_OK(gd_tensor_write_f32(ctx, &running_a, buffer_values, GD_ARRAY_LEN(buffer_values)));

    save_options.metadata = metadata;
    save_options.metadata_len = strlen(metadata);
    save_options.include_buffers = true;
    CHECK_OK(gd_module_save_state(ctx, &root_a, path, &save_options));

    CHECK_OK(gd_checkpoint_read_metadata(path, &loaded_metadata, &loaded_metadata_len));
    CHECK(loaded_metadata_len == strlen(metadata), "metadata length roundtrip");
    CHECK(strcmp(loaded_metadata, metadata) == 0, "metadata content roundtrip");
    free(loaded_metadata);

    init_toy_module(ctx, &root_b, &child_b, &weight_b, &running_b, gd_init_zero(), gd_init_zero());
    load_options.strict = true;
    load_options.load_buffers = true;
    CHECK_OK(gd_module_load_state(ctx, &root_b, path, &load_options));
    CHECK_OK(gd_tensor_read_f32(ctx, &weight_b, got_weight, GD_ARRAY_LEN(got_weight)));
    CHECK_OK(gd_tensor_read_f32(ctx, &running_b, got_buffer, GD_ARRAY_LEN(got_buffer)));
    check_f32_array(got_weight, weight_values, GD_ARRAY_LEN(got_weight), "loaded weight mismatch");
    check_f32_array(got_buffer, buffer_values, GD_ARRAY_LEN(got_buffer), "loaded buffer mismatch");

    gd_module_deinit(&child_b);
    gd_module_deinit(&root_b);
    gd_module_deinit(&child_a);
    gd_module_deinit(&root_a);
    (void)remove(path);
}

static void test_module_state_strict_mismatch(gd_context *ctx)
{
    const char *path = "build/test_module_io_mismatch.ckpt";
    const int64_t bad_shape[2] = {2, 2};
    gd_tensor_spec bad_spec;
    gd_module root_a;
    gd_module child_a;
    gd_tensor weight_a;
    gd_tensor running_a;
    gd_module root_bad;
    gd_module child_bad;
    gd_tensor bad_weight;
    gd_module_save_options save_options;
    gd_module_load_options load_options;

    (void)remove(path);
    init_toy_module(ctx, &root_a, &child_a, &weight_a, &running_a, gd_init_one(), gd_init_zero());
    save_options.metadata = NULL;
    save_options.metadata_len = 0U;
    save_options.include_buffers = true;
    CHECK_OK(gd_module_save_state(ctx, &root_a, path, &save_options));

    bad_spec = gd_tensor_spec_make(GD_DTYPE_F32, gd_shape_make(2U, bad_shape), 256U);
    CHECK_OK(gd_module_init(ctx, &root_bad, "toy"));
    CHECK_OK(gd_module_init_child(ctx, &root_bad, "block", &child_bad));
    CHECK_OK(gd_module_param(ctx, &child_bad, "weight", &bad_spec, NULL, &bad_weight));

    load_options.strict = true;
    load_options.load_buffers = false;
    CHECK(gd_module_load_state(ctx, &root_bad, path, &load_options) != GD_OK,
          "strict load rejects shape mismatch");

    gd_module_deinit(&child_bad);
    gd_module_deinit(&root_bad);
    gd_module_deinit(&child_a);
    gd_module_deinit(&root_a);
    (void)remove(path);
}

static void test_module_state_non_strict(gd_context *ctx)
{
    const char *path = "build/test_module_io_non_strict.ckpt";
    const float weight_values[6] = {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f};
    float got_weight[6];
    gd_module root_a;
    gd_module child_a;
    gd_tensor weight_a;
    gd_tensor running_a;
    gd_module root_b;
    gd_module child_b;
    gd_tensor weight_b;
    gd_tensor running_b;
    gd_module_save_options save_options;
    gd_module_load_options load_options;

    (void)remove(path);
    init_toy_module(ctx, &root_a, &child_a, &weight_a, &running_a, gd_init_zero(), gd_init_one());
    CHECK_OK(gd_tensor_write_f32(ctx, &weight_a, weight_values, GD_ARRAY_LEN(weight_values)));
    save_options.metadata = NULL;
    save_options.metadata_len = 0U;
    save_options.include_buffers = true;
    CHECK_OK(gd_module_save_state(ctx, &root_a, path, &save_options));

    init_toy_module(ctx, &root_b, &child_b, &weight_b, &running_b, gd_init_zero(), gd_init_zero());
    load_options.strict = false;
    load_options.load_buffers = false;
    CHECK_OK(gd_module_load_state(ctx, &root_b, path, &load_options));
    CHECK_OK(gd_tensor_read_f32(ctx, &weight_b, got_weight, GD_ARRAY_LEN(got_weight)));
    check_f32_array(got_weight, weight_values, GD_ARRAY_LEN(got_weight), "non-strict loaded weight mismatch");

    gd_module_deinit(&child_b);
    gd_module_deinit(&root_b);
    gd_module_deinit(&child_a);
    gd_module_deinit(&root_a);
    (void)remove(path);
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = module_io_config();
    gd_status st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_module_io: skipped (no supported GPU backend)\n");
        return 0;
    }
    CHECK_OK(st);

    test_module_state_roundtrip(ctx);
    test_module_state_strict_mismatch(ctx);
    test_module_state_non_strict(ctx);

    gd_context_destroy(ctx);
    printf("test_module_io: ok\n");
    return 0;
}
