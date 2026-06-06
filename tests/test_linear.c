#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "test_linear failed: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static void check_close(float have, float want, float tol, const char *msg)
{
    if (abs_f32(have - want) > tol) {
        fprintf(stderr,
                "test_linear failed: %s have=%.8f want=%.8f tol=%.8f\n",
                msg,
                (double)have,
                (double)want,
                (double)tol);
        exit(1);
    }
}

static gd_memory_config test_config(void)
{
    gd_memory_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = 8U * 1024U * 1024U;
    cfg.state_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slot_bytes = 16U * 1024U * 1024U;
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static size_t shape_count(const int64_t *shape, uint32_t rank)
{
    size_t count = 1U;
    uint32_t axis;
    for (axis = 0U; axis < rank; ++axis) {
        count *= (size_t)shape[axis];
    }
    return count;
}

static void fill_values(float *dst, size_t count, float scale, float bias)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        dst[i] = bias + scale * (float)((i * 17U + 5U) % 31U) - 0.5f * scale * (float)(i % 7U);
    }
}

static void ref_linear_forward(const float *x,
                               const float *w,
                               const float *bias,
                               size_t rows,
                               size_t k,
                               size_t n,
                               float *out)
{
    size_t row;
    for (row = 0U; row < rows; ++row) {
        size_t col;
        for (col = 0U; col < n; ++col) {
            float sum = bias != NULL ? bias[col] : 0.0f;
            size_t kk;
            for (kk = 0U; kk < k; ++kk) {
                sum += x[row * k + kk] * w[kk * n + col];
            }
            out[row * n + col] = sum;
        }
    }
}

static void ref_linear_backward(const float *x,
                                const float *w,
                                const float *grad,
                                size_t rows,
                                size_t k,
                                size_t n,
                                float *dx,
                                float *dw,
                                float *db)
{
    size_t row;
    memset(dx, 0, rows * k * sizeof(dx[0]));
    memset(dw, 0, k * n * sizeof(dw[0]));
    memset(db, 0, n * sizeof(db[0]));
    for (row = 0U; row < rows; ++row) {
        size_t kk;
        size_t col;
        for (kk = 0U; kk < k; ++kk) {
            float sum = 0.0f;
            for (col = 0U; col < n; ++col) {
                sum += grad[row * n + col] * w[kk * n + col];
            }
            dx[row * k + kk] = sum;
        }
        for (kk = 0U; kk < k; ++kk) {
            for (col = 0U; col < n; ++col) {
                dw[kk * n + col] += x[row * k + kk] * grad[row * n + col];
            }
        }
        for (col = 0U; col < n; ++col) {
            db[col] += grad[row * n + col];
        }
    }
}

static void assert_tensor_shape(const gd_tensor *tensor,
                                uint32_t rank,
                                const int64_t *shape,
                                const char *msg)
{
    uint32_t axis;
    CHECK(tensor != NULL && tensor->rank == rank, msg);
    for (axis = 0U; axis < rank; ++axis) {
        CHECK(tensor->shape[axis] == shape[axis], msg);
    }
}

static void test_rank3_forward_with_bias(void)
{
    const int64_t x_shape[3] = {2, 3, 4};
    const int64_t w_shape[2] = {4, 5};
    const int64_t b_shape[1] = {5};
    const int64_t y_shape[3] = {2, 3, 5};
    const size_t x_count = shape_count(x_shape, 3U);
    const size_t w_count = shape_count(w_shape, 2U);
    const size_t b_count = shape_count(b_shape, 1U);
    const size_t y_count = shape_count(y_shape, 3U);
    gd_memory_config cfg = test_config();
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor w;
    gd_tensor b;
    gd_tensor y;
    float x_data[24];
    float w_data[20];
    float b_data[5];
    float xq[24];
    float wq[20];
    float bq[5];
    float got[30];
    float ref[30];
    size_t i;
    fill_values(x_data, x_count, 0.03125f, -0.20f);
    fill_values(w_data, w_count, 0.0234375f, -0.10f);
    fill_values(b_data, b_count, 0.015625f, 0.05f);
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(3U, x_shape), x_data, x_count, false, &x));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(2U, w_shape), w_data, w_count, false, &w));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(1U, b_shape), b_data, b_count, false, &b));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &x, xq, x_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &w, wq, w_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &b, bq, b_count));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_linear(ctx, &x, &w, &b, &y));
    assert_tensor_shape(&y, 3U, y_shape, "rank3 linear output shape");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got, y_count));
    ref_linear_forward(xq, wq, bq, 6U, 4U, 5U, ref);
    for (i = 0U; i < y_count; ++i) {
        check_close(got[i], ref[i], 3.0e-3f, "rank3 linear forward");
    }
    gd_context_destroy(ctx);
}

static void test_rank1_forward_backward_no_bias(void)
{
    const int64_t x_shape[1] = {4};
    const int64_t w_shape[2] = {4, 3};
    const int64_t y_shape[1] = {3};
    const size_t x_count = shape_count(x_shape, 1U);
    const size_t w_count = shape_count(w_shape, 2U);
    const size_t y_count = shape_count(y_shape, 1U);
    gd_memory_config cfg = test_config();
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor w;
    gd_tensor g;
    gd_tensor y;
    gd_tensor dx;
    gd_tensor dw;
    float x_data[4];
    float w_data[12];
    float g_data[3];
    float xq[4];
    float wq[12];
    float gq[3];
    float got_y[3];
    float got_dx[4];
    float got_dw[12];
    float ref_y[3];
    float ref_dx[4];
    float ref_dw[12];
    float ref_db[3];
    size_t i;
    fill_values(x_data, x_count, 0.04f, -0.15f);
    fill_values(w_data, w_count, 0.03f, -0.05f);
    fill_values(g_data, y_count, 0.02f, 0.10f);
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(1U, x_shape), x_data, x_count, false, &x));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(2U, w_shape), w_data, w_count, false, &w));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(1U, y_shape), g_data, y_count, false, &g));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &x, xq, x_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &w, wq, w_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &g, gq, y_count));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_linear(ctx, &x, &w, NULL, &y));
    assert_tensor_shape(&y, 1U, y_shape, "rank1 linear output shape");
    CHECK_OK(gd_linear_backward(ctx, &x, &w, NULL, &g, &dx, &dw, NULL));
    assert_tensor_shape(&dx, 1U, x_shape, "rank1 linear dx shape");
    assert_tensor_shape(&dw, 2U, w_shape, "rank1 linear dw shape");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got_y, y_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx, got_dx, x_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &dw, got_dw, w_count));
    ref_linear_forward(xq, wq, NULL, 1U, 4U, 3U, ref_y);
    ref_linear_backward(xq, wq, gq, 1U, 4U, 3U, ref_dx, ref_dw, ref_db);
    for (i = 0U; i < y_count; ++i) {
        check_close(got_y[i], ref_y[i], 3.0e-3f, "rank1 linear forward");
    }
    for (i = 0U; i < x_count; ++i) {
        check_close(got_dx[i], ref_dx[i], 3.0e-3f, "rank1 linear dx");
    }
    for (i = 0U; i < w_count; ++i) {
        check_close(got_dw[i], ref_dw[i], 3.0e-3f, "rank1 linear dw");
    }
    gd_context_destroy(ctx);
}

static void test_rank3_backward_and_autograd_with_bias(void)
{
    const int64_t x_shape[3] = {2, 3, 4};
    const int64_t w_shape[2] = {4, 5};
    const int64_t b_shape[1] = {5};
    const int64_t y_shape[3] = {2, 3, 5};
    const size_t x_count = shape_count(x_shape, 3U);
    const size_t w_count = shape_count(w_shape, 2U);
    const size_t b_count = shape_count(b_shape, 1U);
    const size_t y_count = shape_count(y_shape, 3U);
    gd_memory_config cfg = test_config();
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor w;
    gd_tensor b;
    gd_tensor g;
    gd_tensor y;
    gd_tensor dx;
    gd_tensor dw;
    gd_tensor db;
    gd_tensor dx_auto;
    gd_tensor dw_auto;
    gd_tensor db_auto;
    float x_data[24];
    float w_data[20];
    float b_data[5];
    float g_data[30];
    float xq[24];
    float wq[20];
    float gq[30];
    float got_dx[24];
    float got_dw[20];
    float got_db[5];
    float ref_dx[24];
    float ref_dw[20];
    float ref_db[5];
    size_t i;
    fill_values(x_data, x_count, 0.025f, -0.12f);
    fill_values(w_data, w_count, 0.018f, 0.03f);
    fill_values(b_data, b_count, 0.01f, -0.02f);
    fill_values(g_data, y_count, 0.015f, 0.07f);
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(3U, x_shape), x_data, x_count, true, &x));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(2U, w_shape), w_data, w_count, true, &w));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(1U, b_shape), b_data, b_count, true, &b));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(3U, y_shape), g_data, y_count, false, &g));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &x, xq, x_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &w, wq, w_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &g, gq, y_count));
    ref_linear_backward(xq, wq, gq, 6U, 4U, 5U, ref_dx, ref_dw, ref_db);

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_linear_backward(ctx, &x, &w, &b, &g, &dx, &dw, &db));
    assert_tensor_shape(&dx, 3U, x_shape, "rank3 linear dx shape");
    assert_tensor_shape(&dw, 2U, w_shape, "rank3 linear dw shape");
    assert_tensor_shape(&db, 1U, b_shape, "rank3 linear db shape");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx, got_dx, x_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &dw, got_dw, w_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &db, got_db, b_count));
    for (i = 0U; i < x_count; ++i) {
        check_close(got_dx[i], ref_dx[i], 4.0e-3f, "rank3 direct dx");
    }
    for (i = 0U; i < w_count; ++i) {
        check_close(got_dw[i], ref_dw[i], 5.0e-3f, "rank3 direct dw");
    }
    for (i = 0U; i < b_count; ++i) {
        check_close(got_db[i], ref_db[i], 4.0e-3f, "rank3 direct db");
    }

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_linear(ctx, &x, &w, &b, &y));
    CHECK_OK(gd_backward(ctx, &y, &g));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx_auto));
    CHECK_OK(gd_tensor_grad(ctx, &w, &dw_auto));
    CHECK_OK(gd_tensor_grad(ctx, &b, &db_auto));
    CHECK_OK(gd_end_step(ctx));
    memset(got_dx, 0, sizeof(got_dx));
    memset(got_dw, 0, sizeof(got_dw));
    memset(got_db, 0, sizeof(got_db));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx_auto, got_dx, x_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &dw_auto, got_dw, w_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &db_auto, got_db, b_count));
    for (i = 0U; i < x_count; ++i) {
        check_close(got_dx[i], ref_dx[i], 4.0e-3f, "rank3 autograd dx");
    }
    for (i = 0U; i < w_count; ++i) {
        check_close(got_dw[i], ref_dw[i], 5.0e-3f, "rank3 autograd dw");
    }
    for (i = 0U; i < b_count; ++i) {
        check_close(got_db[i], ref_db[i], 4.0e-3f, "rank3 autograd db");
    }
    gd_context_destroy(ctx);
}

int main(void)
{
    gd_context *probe_ctx = NULL;
    gd_memory_config cfg = test_config();
    gd_status st = gd_context_create(&cfg, &probe_ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_linear: skipped (no supported GPU backend)\n");
        return 0;
    }
    CHECK_OK(st);
    gd_context_destroy(probe_ctx);

    test_rank3_forward_with_bias();
    test_rank1_forward_backward_no_bias();
    test_rank3_backward_and_autograd_with_bias();
    printf("test_linear: ok\n");
    return 0;
}
