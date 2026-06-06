#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_permute failed: %s (%s:%d)\n", (msg),       \
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

static void check_close(float have, float want, float tol, const char *msg)
{
    if (abs_f32(have - want) > tol) {
        fprintf(stderr,
                "test_permute failed: %s have=%.8f want=%.8f tol=%.8f\n",
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

static gd_memory_config permute_config(size_t bytes)
{
    gd_memory_config cfg = gd_memory_config_default();
    cfg.params_bytes = align_up(bytes * 4U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes * 10U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 4U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static void fill_seq(float *data, uint32_t count)
{
    uint32_t i;
    for (i = 0U; i < count; ++i) {
        data[i] = (float)i + 0.25f;
    }
}

static void test_permute_f32_forward(void)
{
    const int64_t x_shape[3] = {2, 3, 4};
    const int32_t axes[3] = {1, 0, 2};
    float data[24];
    gd_memory_config cfg = permute_config(sizeof(data));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    float got[24];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    fill_seq(data, 24U);
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(3U, x_shape), data, 24U, false, &x));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_permute(ctx, &x, axes, 3U, &y));
    CHECK(y.rank == 3U && y.shape[0] == 3 && y.shape[1] == 2 && y.shape[2] == 4,
          "f32 permute shape");
    CHECK(gd_tensor_is_contiguous(&y), "f32 permute output contiguous");
    CHECK(!y.is_view, "f32 permute materializes output");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got, 24U));
    for (a = 0U; a < 3U; ++a) {
        for (b = 0U; b < 2U; ++b) {
            for (c = 0U; c < 4U; ++c) {
                uint32_t out_index = (a * 2U + b) * 4U + c;
                uint32_t in_index = (b * 3U + a) * 4U + c;
                check_close(got[out_index], data[in_index], 1.0e-6f, "f32 permute order");
            }
        }
    }
    gd_context_destroy(ctx);
}

static void test_permute_u8_negative_axes(void)
{
    const int64_t x_shape[3] = {2, 3, 4};
    const int32_t axes[3] = {1, -3, 2};
    const uint8_t data[24] = {
        0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U,
        8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U,
        16U, 17U, 18U, 19U, 20U, 21U, 22U, 23U};
    gd_memory_config cfg = permute_config(sizeof(data));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    uint8_t got[24];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_U8,
                             gd_shape_make(3U, x_shape), 256U, &x));
    CHECK_OK(gd_tensor_write(ctx, &x, data, sizeof(data)));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_permute(ctx, &x, axes, 3U, &y));
    CHECK(y.rank == 3U && y.shape[0] == 3 && y.shape[1] == 2 && y.shape[2] == 4,
          "u8 negative axes shape");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &y, got, sizeof(got)));
    for (a = 0U; a < 3U; ++a) {
        for (b = 0U; b < 2U; ++b) {
            for (c = 0U; c < 4U; ++c) {
                uint32_t out_index = (a * 2U + b) * 4U + c;
                uint32_t in_index = (b * 3U + a) * 4U + c;
                CHECK(got[out_index] == data[in_index], "u8 permute order");
            }
        }
    }
    gd_context_destroy(ctx);
}

static void test_permute_hwc_chw_fast_paths(void)
{
    const int64_t x_shape[3] = {2, 3, 4};
    const int64_t grad_shape[3] = {4, 2, 3};
    const int32_t axes[3] = {2, 0, 1};
    const uint8_t data[24] = {
        0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U,
        8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U,
        16U, 17U, 18U, 19U, 20U, 21U, 22U, 23U};
    const uint8_t grad_data[24] = {
        100U, 101U, 102U, 103U, 104U, 105U, 106U, 107U,
        108U, 109U, 110U, 111U, 112U, 113U, 114U, 115U,
        116U, 117U, 118U, 119U, 120U, 121U, 122U, 123U};
    gd_memory_config cfg = permute_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    gd_tensor grad_out;
    gd_tensor grad_x;
    uint8_t got[24];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_U8,
                             gd_shape_make(3U, x_shape), 256U, &x));
    CHECK_OK(gd_tensor_write(ctx, &x, data, sizeof(data)));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_U8,
                             gd_shape_make(3U, grad_shape), 256U, &grad_out));
    CHECK_OK(gd_tensor_write(ctx, &grad_out, grad_data, sizeof(grad_data)));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_permute(ctx, &x, axes, 3U, &y));
    CHECK(y.rank == 3U && y.shape[0] == 4 && y.shape[1] == 2 && y.shape[2] == 3,
          "hwc->chw shape");
    CHECK_OK(gd_permute_backward(ctx, &x, &grad_out, axes, 3U, &grad_x));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &y, got, sizeof(got)));
    for (a = 0U; a < 4U; ++a) {
        for (b = 0U; b < 2U; ++b) {
            for (c = 0U; c < 3U; ++c) {
                uint32_t out_index = (a * 2U + b) * 3U + c;
                uint32_t in_index = (b * 3U + c) * 4U + a;
                CHECK(got[out_index] == data[in_index], "hwc->chw order");
            }
        }
    }
    CHECK_OK(gd_tensor_read(ctx, &grad_x, got, sizeof(got)));
    for (a = 0U; a < 2U; ++a) {
        for (b = 0U; b < 3U; ++b) {
            for (c = 0U; c < 4U; ++c) {
                uint32_t dx_index = (a * 3U + b) * 4U + c;
                uint32_t gout_index = (c * 2U + a) * 3U + b;
                CHECK(got[dx_index] == grad_data[gout_index], "chw->hwc backward order");
            }
        }
    }
    gd_context_destroy(ctx);
}

static void test_permute_matrix_transpose_fast_path(void)
{
    const int64_t x_shape[3] = {2, 3, 4};
    const int64_t grad_shape[3] = {2, 4, 3};
    const int32_t axes[3] = {0, 2, 1};
    float data[24];
    float grad_data[24];
    gd_memory_config cfg = permute_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    gd_tensor grad_out;
    gd_tensor grad_x;
    float got[24];
    uint32_t i;
    uint32_t a;
    uint32_t b;
    uint32_t c;
    fill_seq(data, 24U);
    for (i = 0U; i < 24U; ++i) {
        grad_data[i] = 50.0f + (float)i;
    }
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(3U, x_shape), data, 24U, false, &x));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(3U, grad_shape), grad_data, 24U, false, &grad_out));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_permute(ctx, &x, axes, 3U, &y));
    CHECK_OK(gd_permute_backward(ctx, &x, &grad_out, axes, 3U, &grad_x));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got, 24U));
    for (a = 0U; a < 2U; ++a) {
        for (b = 0U; b < 4U; ++b) {
            for (c = 0U; c < 3U; ++c) {
                uint32_t out_index = (a * 4U + b) * 3U + c;
                uint32_t in_index = (a * 3U + c) * 4U + b;
                check_close(got[out_index], data[in_index], 1.0e-6f, "matrix transpose forward order");
            }
        }
    }
    CHECK_OK(gd_tensor_read_f32(ctx, &grad_x, got, 24U));
    for (a = 0U; a < 2U; ++a) {
        for (b = 0U; b < 3U; ++b) {
            for (c = 0U; c < 4U; ++c) {
                uint32_t dx_index = (a * 3U + b) * 4U + c;
                uint32_t gout_index = (a * 4U + c) * 3U + b;
                check_close(got[dx_index], grad_data[gout_index], 1.0e-6f, "matrix transpose backward order");
            }
        }
    }
    gd_context_destroy(ctx);
}

static void test_permute_direct_backward(void)
{
    const int64_t x_shape[3] = {2, 3, 4};
    const int64_t grad_shape[3] = {3, 2, 4};
    const int32_t axes[3] = {1, 0, 2};
    float grad_data[24];
    gd_memory_config cfg = permute_config(sizeof(grad_data));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor grad_out;
    gd_tensor grad_x;
    float got[24];
    uint32_t i;
    uint32_t a;
    uint32_t b;
    uint32_t c;
    for (i = 0U; i < 24U; ++i) {
        grad_data[i] = 100.0f + (float)i;
    }
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(3U, x_shape), 256U, &x));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(3U, grad_shape), grad_data, 24U, false, &grad_out));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_permute_backward(ctx, &x, &grad_out, axes, 3U, &grad_x));
    CHECK(grad_x.rank == 3U && grad_x.shape[0] == 2 && grad_x.shape[1] == 3 && grad_x.shape[2] == 4,
          "permute backward shape");
    CHECK(gd_tensor_is_contiguous(&grad_x), "permute backward output contiguous");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &grad_x, got, 24U));
    for (a = 0U; a < 2U; ++a) {
        for (b = 0U; b < 3U; ++b) {
            for (c = 0U; c < 4U; ++c) {
                uint32_t dx_index = (a * 3U + b) * 4U + c;
                uint32_t gout_index = (b * 2U + a) * 4U + c;
                check_close(got[dx_index], grad_data[gout_index], 1.0e-6f, "permute direct backward order");
            }
        }
    }
    gd_context_destroy(ctx);
}

static void test_permute_autograd(void)
{
    const int64_t x_shape[3] = {2, 3, 4};
    const int32_t axes[3] = {1, 0, 2};
    float data[24];
    gd_memory_config cfg = permute_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    gd_tensor loss;
    gd_tensor dx;
    float got[24];
    uint32_t i;
    fill_seq(data, 24U);
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(3U, x_shape), data, 24U, true, &x));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_permute(ctx, &x, axes, 3U, &y));
    CHECK(y.requires_grad, "permute autograd marks output");
    CHECK_OK(gd_reduce_sum(ctx, &y, &loss));
    CHECK_OK(gd_backward(ctx, &loss, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx, got, 24U));
    for (i = 0U; i < 24U; ++i) {
        check_close(got[i], 1.0f, 1.0e-6f, "permute autograd dx");
    }
    gd_context_destroy(ctx);
}

static void test_permute_scalar_and_invalid(void)
{
    const float scalar_value = 7.0f;
    const int64_t base_shape[2] = {2, 3};
    const int32_t ok_axes[2] = {1, 0};
    const int32_t duplicate_axes[2] = {0, 0};
    const int32_t range_axes[2] = {0, 2};
    const int32_t one_axis[1] = {0};
    gd_memory_config cfg = permute_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor scalar;
    gd_tensor scalar_out;
    gd_tensor base;
    gd_tensor slice;
    gd_tensor out;
    float got_scalar[1];
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                GD_SCALAR_SHAPE, &scalar_value, 1U, false, &scalar));
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(2U, base_shape), 256U, &base));
    CHECK_OK(gd_tensor_slice(ctx, &base, 1U, 0, 2, &slice));
    CHECK(!gd_tensor_is_contiguous(&slice), "slice fixture is non-contiguous");
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_permute(ctx, &scalar, NULL, 0U, &scalar_out));
    CHECK(scalar_out.rank == 0U && !scalar_out.is_view, "scalar permute materializes");
    CHECK_STATUS(gd_permute(ctx, &base, duplicate_axes, 2U, &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_permute(ctx, &base, range_axes, 2U, &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_permute(ctx, &base, one_axis, 1U, &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_permute(ctx, &slice, ok_axes, 2U, &out), GD_ERR_UNSUPPORTED);
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &scalar_out, got_scalar, 1U));
    check_close(got_scalar[0], scalar_value, 1.0e-6f, "scalar permute value");
    gd_context_destroy(ctx);
}

int main(void)
{
    test_permute_f32_forward();
    test_permute_u8_negative_axes();
    test_permute_hwc_chw_fast_paths();
    test_permute_matrix_transpose_fast_path();
    test_permute_direct_backward();
    test_permute_autograd();
    test_permute_scalar_and_invalid();
    printf("test_permute: ok\n");
    return 0;
}
