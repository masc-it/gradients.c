#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_embedding failed: %s (%s:%d)\n", (msg),     \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)
#define CHECK_STATUS(expr, status) CHECK((expr) == (status), #expr)

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static gd_memory_config embedding_config(size_t bytes)
{
    gd_memory_config cfg = gd_memory_config_default();
    cfg.params_bytes = align_up(bytes * 8U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes * 16U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static void check_close(float have, float want, float tol, const char *msg)
{
    if (abs_f32(have - want) > tol) {
        fprintf(stderr,
                "test_embedding failed: %s have=%.8f want=%.8f tol=%.8f\n",
                msg,
                (double)have,
                (double)want,
                (double)tol);
        exit(1);
    }
}

static void embedding_reference(const float *table,
                                const int32_t *ids,
                                const float *grad_out,
                                uint32_t n_ids,
                                uint32_t vocab,
                                uint32_t dim,
                                float *out,
                                float *dtable)
{
    uint32_t i;
    uint32_t c;
    if (dtable != NULL) {
        for (i = 0U; i < vocab * dim; ++i) {
            dtable[i] = 0.0f;
        }
    }
    for (i = 0U; i < n_ids; ++i) {
        int32_t id = ids[i];
        for (c = 0U; c < dim; ++c) {
            size_t out_idx = (size_t)i * dim + c;
            if (id < 0 || (uint32_t)id >= vocab) {
                if (out != NULL) {
                    out[out_idx] = NAN;
                }
            } else {
                size_t table_idx = (size_t)(uint32_t)id * dim + c;
                if (out != NULL) {
                    out[out_idx] = table[table_idx];
                }
                if (dtable != NULL && grad_out != NULL) {
                    dtable[table_idx] += grad_out[out_idx];
                }
            }
        }
    }
}

static void make_i32_tensor(gd_context *ctx,
                            gd_arena_kind arena,
                            gd_shape shape,
                            const int32_t *src,
                            size_t count,
                            gd_tensor *out)
{
    CHECK_OK(gd_tensor_empty(ctx, arena, GD_DTYPE_I32, shape, 256U, out));
    CHECK_OK(gd_tensor_write(ctx, out, src, count * sizeof(*src)));
}

static void test_embedding_f32_forward_backward(void)
{
    const int64_t table_shape[2] = {5, 4};
    const int64_t ids_shape[2] = {2, 3};
    const float table_data[20] = {
        0.0f, 0.1f, 0.2f, 0.3f,
        1.0f, 1.1f, 1.2f, 1.3f,
        2.0f, 2.1f, 2.2f, 2.3f,
        3.0f, 3.1f, 3.2f, 3.3f,
        4.0f, 4.1f, 4.2f, 4.3f,
    };
    const int32_t ids_data[6] = {3, 1, 3, 0, 4, 1};
    const float grad_data[24] = {
        0.25f, 0.5f, -0.25f, 1.0f,
        1.5f, -1.0f, 0.75f, -0.5f,
        0.125f, 0.25f, 0.375f, 0.5f,
        -0.25f, -0.5f, -0.75f, -1.0f,
        2.0f, 1.0f, 0.5f, 0.25f,
        -1.5f, 0.5f, -0.75f, 1.25f,
    };
    float want_y[24];
    float want_dw[20];
    float got_y[24];
    float got_dw[20];
    gd_memory_config cfg = embedding_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor table;
    gd_tensor ids;
    gd_tensor grad;
    gd_tensor y;
    gd_tensor dw;
    uint32_t i;
    embedding_reference(table_data, ids_data, grad_data, 6U, 5U, 4U, want_y, want_dw);
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, table_shape), table_data, 20U, false, &table));
    make_i32_tensor(ctx, GD_ARENA_PARAMS, gd_shape_make(2U, ids_shape), ids_data, 6U, &ids);
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                GD_SHAPE(2, 3, 4), grad_data, 24U, false, &grad));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_embedding(ctx, &table, &ids, &y));
    CHECK_OK(gd_embedding_backward(ctx, &table, &ids, &grad, &dw));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK(y.rank == 3U && y.shape[0] == 2 && y.shape[1] == 3 && y.shape[2] == 4,
          "embedding output shape");
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got_y, 24U));
    for (i = 0U; i < 24U; ++i) {
        check_close(got_y[i], want_y[i], 1.0e-6f, "f32 embedding forward");
    }
    CHECK_OK(gd_tensor_read_f32(ctx, &dw, got_dw, 20U));
    for (i = 0U; i < 20U; ++i) {
        check_close(got_dw[i], want_dw[i], 1.0e-6f, "f32 embedding backward");
    }
    gd_context_destroy(ctx);
}

static void test_embedding_f16_autograd(void)
{
    const int64_t table_shape[2] = {4, 8};
    const int64_t ids_shape[1] = {7};
    const float table_data[32] = {
        0.00f, 0.25f, 0.50f, 0.75f, 1.00f, 1.25f, 1.50f, 1.75f,
        -0.50f, -0.25f, 0.00f, 0.25f, 0.50f, 0.75f, 1.00f, 1.25f,
        2.00f, 1.50f, 1.00f, 0.50f, 0.00f, -0.50f, -1.00f, -1.50f,
        0.125f, -0.125f, 0.375f, -0.375f, 0.625f, -0.625f, 0.875f, -0.875f,
    };
    const int32_t ids_data[7] = {2, 0, 2, 3, 1, 0, 2};
    float grad_data[56];
    float want_dw[32];
    float got_dw[32];
    gd_memory_config cfg = embedding_config(8192U);
    gd_context *ctx = NULL;
    gd_tensor table;
    gd_tensor ids;
    gd_tensor grad;
    gd_tensor y;
    gd_tensor dw;
    uint32_t i;
    for (i = 0U; i < 56U; ++i) {
        grad_data[i] = (float)((int)(i % 9U) - 4) * 0.125f;
    }
    embedding_reference(table_data, ids_data, grad_data, 7U, 4U, 8U, NULL, want_dw);
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(2U, table_shape), table_data, 32U, true, &table));
    make_i32_tensor(ctx, GD_ARENA_PARAMS, gd_shape_make(1U, ids_shape), ids_data, 7U, &ids);
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                GD_SHAPE(7, 8), grad_data, 56U, false, &grad));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_embedding(ctx, &table, &ids, &y));
    CHECK_OK(gd_backward(ctx, &y, &grad));
    CHECK_OK(gd_tensor_grad(ctx, &table, &dw));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dw, got_dw, 32U));
    for (i = 0U; i < 32U; ++i) {
        check_close(got_dw[i], want_dw[i], 1.0e-3f, "f16 embedding autograd dtable");
    }
    gd_context_destroy(ctx);
}

static void test_embedding_invalid_id_outputs_nan(void)
{
    const int64_t table_shape[2] = {2, 3};
    const int64_t ids_shape[1] = {3};
    const float table_data[6] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const int32_t ids_data[3] = {0, 7, 1};
    float got[9];
    gd_memory_config cfg = embedding_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor table;
    gd_tensor ids;
    gd_tensor y;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, table_shape), table_data, 6U, false, &table));
    make_i32_tensor(ctx, GD_ARENA_PARAMS, gd_shape_make(1U, ids_shape), ids_data, 3U, &ids);
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_embedding(ctx, &table, &ids, &y));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got, 9U));
    CHECK(isnan(got[3]) && isnan(got[4]) && isnan(got[5]), "invalid embedding id writes NaN row");
    gd_context_destroy(ctx);
}

static void test_embedding_validation(void)
{
    const int64_t table_shape[2] = {3, 4};
    const int64_t ids_shape[1] = {2};
    const float table_data[12] = {0};
    const float bad_ids_data[2] = {0.0f, 1.0f};
    const int32_t ids_data[2] = {0, 1};
    gd_memory_config cfg = embedding_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor table;
    gd_tensor bad_ids;
    gd_tensor ids;
    gd_tensor y;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, table_shape), table_data, 12U, false, &table));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(1U, ids_shape), bad_ids_data, 2U, false, &bad_ids));
    make_i32_tensor(ctx, GD_ARENA_PARAMS, gd_shape_make(1U, ids_shape), ids_data, 2U, &ids);
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_STATUS(gd_embedding(ctx, &table, &bad_ids, &y), GD_ERR_UNSUPPORTED);
    table.rank = 1U;
    CHECK_STATUS(gd_embedding(ctx, &table, &ids, &y), GD_ERR_INVALID_ARGUMENT);
    gd_context_destroy(ctx);
}

int main(void)
{
    test_embedding_f32_forward_backward();
    test_embedding_f16_autograd();
    test_embedding_invalid_id_outputs_nan();
    test_embedding_validation();
    printf("test_embedding: ok\n");
    return 0;
}
