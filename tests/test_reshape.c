#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_reshape failed: %s (%s:%d)\n", (msg),       \
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
                "test_reshape failed: %s have=%.8f want=%.8f tol=%.8f\n",
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

static gd_memory_config reshape_config(size_t bytes)
{
    gd_memory_config cfg = gd_memory_config_default();
    cfg.params_bytes = align_up(bytes * 4U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes * 8U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static void check_alias(const gd_tensor *base, const gd_tensor *view, const char *msg)
{
    CHECK(base->storage.buffer == view->storage.buffer, msg);
    CHECK(base->storage.offset == view->storage.offset, msg);
    CHECK(base->view_offset == view->view_offset, msg);
    CHECK(view->is_view, msg);
}

static void test_reshape_f32_forward_alias(void)
{
    const int64_t x_shape[2] = {2, 3};
    const int64_t y_shape[2] = {3, 2};
    const float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    gd_memory_config cfg = reshape_config(sizeof(data));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    float got[6];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, x_shape), data, 6U, false, &x));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_reshape(ctx, &x, gd_shape_make(2U, y_shape), &y));
    CHECK(y.rank == 2U && y.shape[0] == 3 && y.shape[1] == 2, "f32 reshape shape");
    CHECK(y.strides[0] == 2 && y.strides[1] == 1, "f32 reshape strides");
    check_alias(&x, &y, "f32 reshape aliases input");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got, 6U));
    for (i = 0U; i < 6U; ++i) {
        check_close(got[i], data[i], 1.0e-6f, "f32 reshape forward order");
    }
    gd_context_destroy(ctx);
}

static void test_reshape_infer_f16_and_u8(void)
{
    const int64_t x_shape[3] = {2, 3, 4};
    const int64_t y_shape[3] = {2, -1, 3};
    const int64_t u8_shape[2] = {2, 2};
    const int64_t u8_target[1] = {-1};
    const float data[24] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
        18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f};
    const uint8_t bytes[4] = {9U, 8U, 7U, 6U};
    gd_memory_config cfg = reshape_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    gd_tensor ux;
    gd_tensor uy;
    float got[24];
    uint8_t got_u8[4];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(3U, x_shape), data, 24U, false, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_U8,
                             gd_shape_make(2U, u8_shape), 256U, &ux));
    CHECK_OK(gd_tensor_write(ctx, &ux, bytes, sizeof(bytes)));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_reshape(ctx, &x, gd_shape_make(3U, y_shape), &y));
    CHECK(y.rank == 3U && y.shape[0] == 2 && y.shape[1] == 4 && y.shape[2] == 3,
          "f16 inferred shape");
    check_alias(&x, &y, "f16 reshape aliases input");
    CHECK_OK(gd_reshape(ctx, &ux, gd_shape_make(1U, u8_target), &uy));
    CHECK(uy.rank == 1U && uy.shape[0] == 4, "u8 inferred shape");
    check_alias(&ux, &uy, "u8 reshape aliases input");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got, 24U));
    for (i = 0U; i < 24U; ++i) {
        check_close(got[i], data[i], 1.0e-3f, "f16 reshape forward order");
    }
    CHECK_OK(gd_tensor_read(ctx, &uy, got_u8, sizeof(got_u8)));
    for (i = 0U; i < 4U; ++i) {
        CHECK(got_u8[i] == bytes[i], "u8 reshape forward order");
    }
    gd_context_destroy(ctx);
}

static void test_reshape_direct_backward(void)
{
    const int64_t x_shape[2] = {2, 3};
    const int64_t grad_shape[1] = {6};
    const float grad_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    gd_memory_config cfg = reshape_config(sizeof(grad_data));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor grad_out;
    gd_tensor grad_x;
    float got[6];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(2U, x_shape), 256U, &x));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(1U, grad_shape), grad_data, 6U, false, &grad_out));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_reshape_backward(ctx, &x, &grad_out, &grad_x));
    CHECK(grad_x.rank == 2U && grad_x.shape[0] == 2 && grad_x.shape[1] == 3,
          "reshape backward shape");
    check_alias(&grad_out, &grad_x, "reshape backward aliases grad_out");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &grad_x, got, 6U));
    for (i = 0U; i < 6U; ++i) {
        check_close(got[i], grad_data[i], 1.0e-6f, "reshape direct backward order");
    }
    gd_context_destroy(ctx);
}

static void test_reshape_autograd(void)
{
    const int64_t x_shape[2] = {2, 3};
    const int64_t y_shape[1] = {-1};
    const float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    gd_memory_config cfg = reshape_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    gd_tensor loss;
    gd_tensor dx;
    float got[6];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, x_shape), data, 6U, true, &x));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_reshape(ctx, &x, gd_shape_make(1U, y_shape), &y));
    CHECK(y.requires_grad, "reshape autograd marks output");
    CHECK_OK(gd_reduce_sum(ctx, &y, &loss));
    CHECK_OK(gd_backward(ctx, &loss, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx, got, 6U));
    for (i = 0U; i < 6U; ++i) {
        check_close(got[i], 1.0f, 1.0e-6f, "reshape autograd dx");
    }
    gd_context_destroy(ctx);
}

static void test_reshape_scalar_and_invalid(void)
{
    const int64_t one_shape[1] = {1};
    const int64_t bad_shape[2] = {2, 4};
    const int64_t multi_infer[2] = {-1, -1};
    const int64_t zero_dim[2] = {0, 6};
    const int64_t base_shape[2] = {2, 3};
    const int64_t slice_target[1] = {4};
    const float scalar_value = 7.0f;
    gd_memory_config cfg = reshape_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor scalar;
    gd_tensor vec;
    gd_tensor base;
    gd_tensor slice;
    gd_tensor out;
    float got[1];
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                GD_SCALAR_SHAPE, &scalar_value, 1U, false, &scalar));
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                             gd_shape_make(2U, base_shape), 256U, &base));
    CHECK_OK(gd_tensor_slice(ctx, &base, 1U, 0, 2, &slice));
    CHECK(!gd_tensor_is_contiguous(&slice), "slice fixture is non-contiguous");
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_reshape(ctx, &scalar, gd_shape_make(1U, one_shape), &vec));
    CHECK(vec.rank == 1U && vec.shape[0] == 1, "scalar to vector shape");
    check_alias(&scalar, &vec, "scalar reshape aliases input");
    CHECK_OK(gd_reshape(ctx, &vec, GD_SCALAR_SHAPE, &out));
    CHECK(out.rank == 0U, "vector to scalar shape");
    check_alias(&vec, &out, "vector reshape aliases input");
    CHECK_STATUS(gd_reshape(ctx, &base, gd_shape_make(2U, bad_shape), &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_reshape(ctx, &base, gd_shape_make(2U, multi_infer), &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_reshape(ctx, &base, gd_shape_make(2U, zero_dim), &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_reshape(ctx, &slice, gd_shape_make(1U, slice_target), &out), GD_ERR_UNSUPPORTED);
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &vec, got, 1U));
    check_close(got[0], scalar_value, 1.0e-6f, "scalar reshape value");
    gd_context_destroy(ctx);
}

int main(void)
{
    test_reshape_f32_forward_alias();
    test_reshape_infer_f16_and_u8();
    test_reshape_direct_backward();
    test_reshape_autograd();
    test_reshape_scalar_and_invalid();
    printf("test_reshape: ok\n");
    return 0;
}
