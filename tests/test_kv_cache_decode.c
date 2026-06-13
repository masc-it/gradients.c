#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_kv_cache_decode failed: %s (%s:%d)\n",      \
                    (msg), __FILE__, __LINE__);                                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static gd_memory_config test_config(void)
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

static void test_kv_cache_append_at(gd_context *ctx)
{
    const int64_t cache_shape[4] = {1, 5, 1, 2};
    const int64_t new_shape[4] = {1, 2, 1, 2};
    const float k_new_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float v_new_data[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
    const float expect_k[10] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                               2.0f, 3.0f, 4.0f, 0.0f, 0.0f};
    const float expect_v[10] = {0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
                               -2.0f, -3.0f, -4.0f, 0.0f, 0.0f};
    float got_k[10];
    float got_v[10];
    gd_tensor k_cache;
    gd_tensor v_cache;
    gd_tensor k_new;
    gd_tensor v_new;
    uint32_t i;

    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_STATE, GD_DTYPE_F32,
                             gd_shape_make(4U, cache_shape), 256U, &k_cache));
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_STATE, GD_DTYPE_F32,
                             gd_shape_make(4U, cache_shape), 256U, &v_cache));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(4U, new_shape), 256U, &k_new));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(4U, new_shape), 256U, &v_new));
    CHECK_OK(gd_tensor_write_f32(ctx, &k_new, k_new_data, GD_ARRAY_LEN(k_new_data)));
    CHECK_OK(gd_tensor_write_f32(ctx, &v_new, v_new_data, GD_ARRAY_LEN(v_new_data)));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
    CHECK_OK(gd_kv_cache_append_at(ctx, &k_cache, &v_cache, 2, &k_new, &v_new));
    CHECK_OK(gd_end_step(ctx));

    memset(got_k, 0, sizeof(got_k));
    memset(got_v, 0, sizeof(got_v));
    CHECK_OK(gd_tensor_read_f32(ctx, &k_cache, got_k, GD_ARRAY_LEN(got_k)));
    CHECK_OK(gd_tensor_read_f32(ctx, &v_cache, got_v, GD_ARRAY_LEN(got_v)));
    for (i = 0U; i < GD_ARRAY_LEN(got_k); ++i) {
        CHECK(abs_f32(got_k[i] - expect_k[i]) < 1.0e-6f, "k cache append mismatch");
        CHECK(abs_f32(got_v[i] - expect_v[i]) < 1.0e-6f, "v cache append mismatch");
    }
}

static void test_kv_cache_append_positions(gd_context *ctx)
{
    const int64_t cache_shape[4] = {2, 5, 1, 2};
    const int64_t new_shape[4] = {2, 2, 1, 2};
    const int64_t pos_shape[1] = {2};
    const float k_new_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const float v_new_data[8] = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};
    const int32_t pos_data[2] = {1, 3};
    const float expect_k[20] = {0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const float expect_v[20] = {0.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -5.0f, -6.0f, -7.0f, -8.0f};
    float got_k[20];
    float got_v[20];
    gd_tensor k_cache;
    gd_tensor v_cache;
    gd_tensor k_new;
    gd_tensor v_new;
    gd_tensor pos;
    uint32_t i;

    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_STATE, GD_DTYPE_F32,
                             gd_shape_make(4U, cache_shape), 256U, &k_cache));
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_STATE, GD_DTYPE_F32,
                             gd_shape_make(4U, cache_shape), 256U, &v_cache));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(4U, new_shape), 256U, &k_new));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(4U, new_shape), 256U, &v_new));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32,
                             gd_shape_make(1U, pos_shape), 256U, &pos));
    CHECK_OK(gd_tensor_write_f32(ctx, &k_new, k_new_data, GD_ARRAY_LEN(k_new_data)));
    CHECK_OK(gd_tensor_write_f32(ctx, &v_new, v_new_data, GD_ARRAY_LEN(v_new_data)));
    CHECK_OK(gd_tensor_write(ctx, &pos, pos_data, sizeof(pos_data)));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
    CHECK_OK(gd_kv_cache_append_positions(ctx, &k_cache, &v_cache, &pos, &k_new, &v_new));
    CHECK_OK(gd_end_step(ctx));

    CHECK_OK(gd_tensor_read_f32(ctx, &k_cache, got_k, GD_ARRAY_LEN(got_k)));
    CHECK_OK(gd_tensor_read_f32(ctx, &v_cache, got_v, GD_ARRAY_LEN(got_v)));
    for (i = 0U; i < GD_ARRAY_LEN(got_k); ++i) {
        CHECK(abs_f32(got_k[i] - expect_k[i]) < 1.0e-6f, "k cache append positions mismatch");
        CHECK(abs_f32(got_v[i] - expect_v[i]) < 1.0e-6f, "v cache append positions mismatch");
    }
}

static void test_kv_cache_append_packed(gd_context *ctx)
{
    const int64_t cache_shape[4] = {2, 5, 1, 2};
    const int64_t new_shape[3] = {3, 1, 2};
    const int64_t pos_shape[1] = {2};
    const int64_t cu_shape[1] = {3};
    const float k_new_data[6] = {11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    const float v_new_data[6] = {-11.0f, -12.0f, -13.0f, -14.0f, -15.0f, -16.0f};
    const int32_t pos_data[2] = {1, 0};
    const int32_t cu_data[3] = {0, 2, 3};
    const float expect_k[20] = {0.0f, 0.0f, 11.0f, 12.0f, 13.0f, 14.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                15.0f, 16.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float expect_v[20] = {0.0f, 0.0f, -11.0f, -12.0f, -13.0f, -14.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                -15.0f, -16.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float got_k[20];
    float got_v[20];
    gd_tensor k_cache;
    gd_tensor v_cache;
    gd_tensor k_new;
    gd_tensor v_new;
    gd_tensor pos;
    gd_tensor cu;
    uint32_t i;

    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_STATE, GD_DTYPE_F32,
                             gd_shape_make(4U, cache_shape), 256U, &k_cache));
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_STATE, GD_DTYPE_F32,
                             gd_shape_make(4U, cache_shape), 256U, &v_cache));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(3U, new_shape), 256U, &k_new));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(3U, new_shape), 256U, &v_new));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32,
                             gd_shape_make(1U, pos_shape), 256U, &pos));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32,
                             gd_shape_make(1U, cu_shape), 256U, &cu));
    CHECK_OK(gd_tensor_write_f32(ctx, &k_new, k_new_data, GD_ARRAY_LEN(k_new_data)));
    CHECK_OK(gd_tensor_write_f32(ctx, &v_new, v_new_data, GD_ARRAY_LEN(v_new_data)));
    CHECK_OK(gd_tensor_write(ctx, &pos, pos_data, sizeof(pos_data)));
    CHECK_OK(gd_tensor_write(ctx, &cu, cu_data, sizeof(cu_data)));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
    CHECK_OK(gd_kv_cache_append_packed(ctx, &k_cache, &v_cache, &pos, &cu, &k_new, &v_new));
    CHECK_OK(gd_end_step(ctx));

    CHECK_OK(gd_tensor_read_f32(ctx, &k_cache, got_k, GD_ARRAY_LEN(got_k)));
    CHECK_OK(gd_tensor_read_f32(ctx, &v_cache, got_v, GD_ARRAY_LEN(got_v)));
    for (i = 0U; i < GD_ARRAY_LEN(got_k); ++i) {
        CHECK(abs_f32(got_k[i] - expect_k[i]) < 1.0e-6f, "k cache append packed mismatch");
        CHECK(abs_f32(got_v[i] - expect_v[i]) < 1.0e-6f, "v cache append packed mismatch");
    }
}

static void test_sdpa_decode_positions(gd_context *ctx)
{
    const int64_t q_shape[4] = {2, 1, 2, 4};
    const int64_t kv_shape[4] = {2, 5, 1, 4};
    const int64_t pos_shape[1] = {2};
    float q_data[16];
    float k_data[40];
    float v_data[40];
    float got_batched[16];
    float got_scalar[16];
    int32_t pos_data[2] = {1, 3};
    gd_tensor q;
    gd_tensor k;
    gd_tensor v;
    gd_tensor pos;
    gd_tensor out_batched;
    gd_tensor q_b;
    gd_tensor k_b;
    gd_tensor v_b;
    gd_tensor out_b;
    gd_sdpa_decode_config cfg;
    uint32_t i;
    uint32_t b;

    for (i = 0U; i < GD_ARRAY_LEN(q_data); ++i) {
        q_data[i] = ((float)(int32_t)i - 7.0f) * 0.03f;
    }
    for (i = 0U; i < GD_ARRAY_LEN(k_data); ++i) {
        k_data[i] = ((float)(int32_t)i - 11.0f) * 0.02f;
        v_data[i] = ((float)(int32_t)i + 3.0f) * -0.015f;
    }
    cfg.scale = 0.5f;
    cfg.sliding_window = 0;
    cfg.prefix_len = 0;

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(4U, q_shape), 256U, &q));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(4U, kv_shape), 256U, &k));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(4U, kv_shape), 256U, &v));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32,
                             gd_shape_make(1U, pos_shape), 256U, &pos));
    CHECK_OK(gd_tensor_write_f32(ctx, &q, q_data, GD_ARRAY_LEN(q_data)));
    CHECK_OK(gd_tensor_write_f32(ctx, &k, k_data, GD_ARRAY_LEN(k_data)));
    CHECK_OK(gd_tensor_write_f32(ctx, &v, v_data, GD_ARRAY_LEN(v_data)));
    CHECK_OK(gd_tensor_write(ctx, &pos, pos_data, sizeof(pos_data)));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
    CHECK_OK(gd_sdpa_decode_positions(ctx, &q, &k, &v, &pos, &cfg, &out_batched));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &out_batched, got_batched, GD_ARRAY_LEN(got_batched)));

    for (b = 0U; b < 2U; ++b) {
        CHECK_OK(gd_tensor_slice(ctx, &q, 0U, (int64_t)b, 1, &q_b));
        CHECK_OK(gd_tensor_slice(ctx, &k, 0U, (int64_t)b, 1, &k_b));
        CHECK_OK(gd_tensor_slice(ctx, &v, 0U, (int64_t)b, 1, &v_b));
        CHECK_OK(gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
        CHECK_OK(gd_sdpa_decode_at(ctx, &q_b, &k_b, &v_b, pos_data[b], &cfg, &out_b));
        CHECK_OK(gd_end_step(ctx));
        CHECK_OK(gd_tensor_read_f32(ctx, &out_b, got_scalar + b * 8U, 8U));
    }
    for (i = 0U; i < GD_ARRAY_LEN(got_batched); ++i) {
        CHECK(abs_f32(got_batched[i] - got_scalar[i]) < 1.0e-6f,
              "sdpa_decode_positions differs from scalar decode");
    }
}

static void test_sdpa_decode_at_matches_tensor_pos(gd_context *ctx)
{
    const int64_t q_shape[4] = {1, 2, 2, 4};
    const int64_t kv_shape[4] = {1, 5, 1, 4};
    float q_data[16];
    float k_data[20];
    float v_data[20];
    float got_tensor[16];
    float got_scalar[16];
    int32_t pos_data = 2;
    gd_tensor q;
    gd_tensor k;
    gd_tensor v;
    gd_tensor pos;
    gd_tensor out_tensor;
    gd_tensor out_scalar;
    gd_sdpa_decode_config cfg;
    uint32_t i;

    for (i = 0U; i < GD_ARRAY_LEN(q_data); ++i) {
        q_data[i] = ((float)(int32_t)i - 3.0f) * 0.05f;
    }
    for (i = 0U; i < GD_ARRAY_LEN(k_data); ++i) {
        k_data[i] = ((float)(int32_t)i - 5.0f) * 0.04f;
        v_data[i] = ((float)(int32_t)i + 2.0f) * -0.03f;
    }
    cfg.scale = 0.5f;
    cfg.sliding_window = 0;
    cfg.prefix_len = 0;

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(4U, q_shape), 256U, &q));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(4U, kv_shape), 256U, &k));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(4U, kv_shape), 256U, &v));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32,
                             GD_SCALAR_SHAPE, 256U, &pos));
    CHECK_OK(gd_tensor_write_f32(ctx, &q, q_data, GD_ARRAY_LEN(q_data)));
    CHECK_OK(gd_tensor_write_f32(ctx, &k, k_data, GD_ARRAY_LEN(k_data)));
    CHECK_OK(gd_tensor_write_f32(ctx, &v, v_data, GD_ARRAY_LEN(v_data)));
    CHECK_OK(gd_tensor_write(ctx, &pos, &pos_data, sizeof(pos_data)));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
    CHECK_OK(gd_sdpa_decode(ctx, &q, &k, &v, &pos, &cfg, &out_tensor));
    CHECK_OK(gd_sdpa_decode_at(ctx, &q, &k, &v, pos_data, &cfg, &out_scalar));
    CHECK_OK(gd_end_step(ctx));

    CHECK_OK(gd_tensor_read_f32(ctx, &out_tensor, got_tensor, GD_ARRAY_LEN(got_tensor)));
    CHECK_OK(gd_tensor_read_f32(ctx, &out_scalar, got_scalar, GD_ARRAY_LEN(got_scalar)));
    for (i = 0U; i < GD_ARRAY_LEN(got_tensor); ++i) {
        CHECK(abs_f32(got_tensor[i] - got_scalar[i]) < 1.0e-6f,
              "sdpa_decode_at differs from tensor-position decode");
    }
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = test_config();
    gd_status st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_kv_cache_decode: skipped (no supported GPU backend)\n");
        return 0;
    }
    CHECK_OK(st);
    test_kv_cache_append_at(ctx);
    test_kv_cache_append_positions(ctx);
    test_kv_cache_append_packed(ctx);
    test_sdpa_decode_at_matches_tensor_pos(ctx);
    test_sdpa_decode_positions(ctx);
    gd_context_destroy(ctx);
    printf("test_kv_cache_decode: ok\n");
    return 0;
}
