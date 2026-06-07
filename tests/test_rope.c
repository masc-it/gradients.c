#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "test_rope failed: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)
#define CHECK_STATUS(expr, status) CHECK((expr) == (status), #expr)

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static void check_close(float have, float want, float tol, const char *msg)
{
    if (abs_f32(have - want) > tol) {
        fprintf(stderr,
                "test_rope failed: %s have=%.8f want=%.8f tol=%.8f\n",
                msg,
                (double)have,
                (double)want,
                (double)tol);
        exit(1);
    }
}

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static gd_memory_config rope_config(size_t tensor_bytes, size_t pos_bytes)
{
    gd_memory_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(tensor_bytes * 3U + pos_bytes + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(tensor_bytes * 12U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static size_t shape_count(const int64_t *shape, uint32_t rank)
{
    size_t count = 1U;
    uint32_t i;
    for (i = 0U; i < rank; ++i) {
        count *= (size_t)shape[i];
    }
    return count;
}

static void rope_ref(const float *x,
                     const int32_t *pos,
                     float *out,
                     const int64_t *shape,
                     uint32_t rank,
                     float theta,
                     int32_t n_dims,
                     int interleaved,
                     float sin_sign)
{
    const size_t head_dim = (size_t)shape[rank - 1U];
    const size_t heads = (size_t)shape[rank - 2U];
    const size_t rows = shape_count(shape, rank) / head_dim;
    const size_t half = (size_t)n_dims / 2U;
    size_t r;
    for (r = 0U; r < rows; ++r) {
        const int32_t p = pos[r / heads];
        const size_t base = r * head_dim;
        size_t d;
        size_t i;
        for (d = (size_t)n_dims; d < head_dim; ++d) {
            out[base + d] = x[base + d];
        }
        for (i = 0U; i < half; ++i) {
            const float inv = powf(theta, -2.0f * (float)i / (float)n_dims);
            const float angle = (float)p * inv;
            const float c = cosf(angle);
            const float s = sinf(angle) * sin_sign;
            const size_t a = interleaved != 0 ? (2U * i) : i;
            const size_t b = interleaved != 0 ? (2U * i + 1U) : (i + half);
            const float x0 = x[base + a];
            const float x1 = x[base + b];
            out[base + a] = x0 * c - x1 * s;
            out[base + b] = x0 * s + x1 * c;
        }
    }
}

static void fill_sequence(float *dst, size_t count, float scale, float bias)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        dst[i] = bias + scale * (float)((i * 13U + 7U) % 29U) - 0.5f * scale * (float)(i % 5U);
    }
}

static void fill_positions(int32_t *dst, size_t count)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        dst[i] = (int32_t)((i * 7U + 3U) & 0x7fU);
    }
}

static void run_case(gd_dtype dtype,
                     const int64_t *shape,
                     uint32_t rank,
                     const int32_t *pos_data,
                     size_t pos_count,
                     gd_rope_config rope,
                     float tol,
                     const char *label)
{
    const size_t count = shape_count(shape, rank);
    const size_t elem_size = gd_dtype_size(dtype);
    gd_memory_config cfg = rope_config(count * elem_size, pos_count * sizeof(pos_data[0]));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor grad;
    gd_tensor pos;
    gd_tensor y;
    gd_tensor dx;
    gd_tensor y_auto;
    gd_tensor dx_auto;
    float *x_data = (float *)calloc(count, sizeof(float));
    float *g_data = (float *)calloc(count, sizeof(float));
    float *x_quant = (float *)calloc(count, sizeof(float));
    float *g_quant = (float *)calloc(count, sizeof(float));
    float *ref = (float *)calloc(count, sizeof(float));
    float *got = (float *)calloc(count, sizeof(float));
    float *dx_ref = (float *)calloc(count, sizeof(float));
    uint32_t i;
    CHECK(x_data != NULL && g_data != NULL && x_quant != NULL && g_quant != NULL && ref != NULL &&
              got != NULL && dx_ref != NULL,
          "allocation");
    fill_sequence(x_data, count, 0.03125f, -0.75f);
    fill_sequence(g_data, count, 0.0234375f, 0.25f);
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx,
                                GD_ARENA_PARAMS,
                                dtype,
                                gd_shape_make(rank, shape),
                                x_data,
                                count,
                                false,
                                &x));
    CHECK_OK(gd_tensor_from_f32(ctx,
                                GD_ARENA_PARAMS,
                                dtype,
                                gd_shape_make(rank, shape),
                                g_data,
                                count,
                                false,
                                &grad));
    CHECK_OK(gd_tensor_empty(ctx,
                             GD_ARENA_PARAMS,
                             GD_DTYPE_I32,
                             gd_shape_make(pos_count == 1U ? 0U : 1U,
                                           pos_count == 1U ? NULL : (const int64_t[]){(int64_t)pos_count}),
                             256U,
                             &pos));
    CHECK_OK(gd_tensor_write(ctx, &pos, pos_data, pos_count * sizeof(pos_data[0])));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &x, x_quant, count));
    CHECK_OK(gd_tensor_read_f32(ctx, &grad, g_quant, count));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_rope(ctx, &x, &pos, &rope, &y));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got, count));
    rope_ref(x_quant, pos_data, ref, shape, rank, rope.theta, rope.n_dims, rope.interleaved ? 1 : 0, 1.0f);
    for (i = 0U; i < (uint32_t)count; ++i) {
        check_close(got[i], ref[i], tol, label);
    }

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_rope_backward(ctx, &x, &pos, &grad, &rope, &dx));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, count * sizeof(got[0]));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx, got, count));
    rope_ref(g_quant, pos_data, dx_ref, shape, rank, rope.theta, rope.n_dims, rope.interleaved ? 1 : 0, -1.0f);
    for (i = 0U; i < (uint32_t)count; ++i) {
        check_close(got[i], dx_ref[i], tol, label);
    }

    x.requires_grad = true;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_rope(ctx, &x, &pos, &rope, &y_auto));
    CHECK_OK(gd_backward(ctx, &y_auto, &grad));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx_auto));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, count * sizeof(got[0]));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx_auto, got, count));
    for (i = 0U; i < (uint32_t)count; ++i) {
        check_close(got[i], dx_ref[i], tol, label);
    }

    gd_context_destroy(ctx);
    free(dx_ref);
    free(got);
    free(ref);
    free(g_quant);
    free(x_quant);
    free(g_data);
    free(x_data);
}

static void test_rope_f32_half_split(void)
{
    const int64_t shape[3] = {2, 3, 6};
    const int32_t pos[2] = {0, 3};
    const gd_rope_config rope = {.theta = 10000.0f, .n_dims = 6, .interleaved = false};
    run_case(GD_DTYPE_F32, shape, 3U, pos, 2U, rope, 2.5e-5f, "f32 half-split rope");
}

static void test_rope_f16_interleaved_tail(void)
{
    const int64_t shape[3] = {2, 2, 8};
    const int32_t pos[2] = {5, 9};
    const gd_rope_config rope = {.theta = 10000.0f, .n_dims = 4, .interleaved = true};
    run_case(GD_DTYPE_F16, shape, 3U, pos, 2U, rope, 1.5e-3f, "f16 interleaved rope tail");
}

static void test_rope_f32_full_head_many_tokens(void)
{
    const int64_t shape[3] = {1024, 2, 6};
    const gd_rope_config rope = {.theta = 10000.0f, .n_dims = 6, .interleaved = false};
    int32_t *pos = (int32_t *)malloc((size_t)shape[0] * sizeof(pos[0]));
    CHECK(pos != NULL, "position allocation");
    fill_positions(pos, (size_t)shape[0]);
    run_case(GD_DTYPE_F32, shape, 3U, pos, (size_t)shape[0], rope, 2.5e-5f, "f32 full-head rope");
    free(pos);
}

static void test_rope_f16_full_head_many_tokens(void)
{
    const int64_t shape[3] = {1024, 4, 64};
    const gd_rope_config rope = {.theta = 10000.0f, .n_dims = 64, .interleaved = false};
    int32_t *pos = (int32_t *)malloc((size_t)shape[0] * sizeof(pos[0]));
    CHECK(pos != NULL, "position allocation");
    fill_positions(pos, (size_t)shape[0]);
    run_case(GD_DTYPE_F16, shape, 3U, pos, (size_t)shape[0], rope, 1.5e-3f, "f16 full-head rope");
    free(pos);
}

static void test_rope_rejects_bad_n_dims(void)
{
    const int64_t shape[2] = {1, 6};
    const int32_t pos_data[1] = {0};
    gd_memory_config cfg = rope_config(6U * sizeof(float), sizeof(pos_data));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor pos;
    gd_tensor y;
    float x_data[6] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    const gd_rope_config bad = {.theta = 10000.0f, .n_dims = 5, .interleaved = false};
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx,
                                GD_ARENA_PARAMS,
                                GD_DTYPE_F32,
                                gd_shape_make(2U, shape),
                                x_data,
                                6U,
                                false,
                                &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, GD_SCALAR_SHAPE, 256U, &pos));
    CHECK_OK(gd_tensor_write(ctx, &pos, pos_data, sizeof(pos_data)));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_STATUS(gd_rope(ctx, &x, &pos, &bad, &y), GD_ERR_INVALID_ARGUMENT);
    CHECK_OK(gd_end_step(ctx));
    gd_context_destroy(ctx);
}

int main(void)
{
    test_rope_f32_half_split();
    test_rope_f16_interleaved_tail();
    test_rope_f32_full_head_many_tokens();
    test_rope_f16_full_head_many_tokens();
    test_rope_rejects_bad_n_dims();
    printf("test_rope ok\n");
    return 0;
}
