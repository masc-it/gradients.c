#include <gradients/gradients.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_powlu_split_linear failed: %s (%s:%d)\n",   \
                    (msg), __FILE__, __LINE__);                                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static gd_memory_config test_config(size_t bytes)
{
    gd_memory_config cfg = gd_memory_config_default();
    cfg.params_bytes = align_up(bytes * 4U + 4U * 1024U * 1024U, 4096U);
    cfg.state_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes * 32U + 16U * 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static void check_close_array(const float *have,
                              const float *want,
                              size_t count,
                              float tol,
                              const char *msg)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (abs_f32(have[i] - want[i]) > tol) {
            fprintf(stderr,
                    "test_powlu_split_linear failed: %s[%zu] have=%.8f want=%.8f tol=%.8f\n",
                    msg,
                    i,
                    (double)have[i],
                    (double)want[i],
                    (double)tol);
            exit(1);
        }
    }
}

int main(void)
{
    enum { ROWS = 32, HIDDEN = 32, OUT = 16 };
    const int64_t x12_shape[2] = {ROWS, 2 * HIDDEN};
    const int64_t w_shape[2] = {HIDDEN, OUT};
    const int64_t y_shape[2] = {ROWS, OUT};
    const size_t x12_count = (size_t)ROWS * (size_t)(2 * HIDDEN);
    const size_t w_count = (size_t)HIDDEN * (size_t)OUT;
    const size_t y_count = (size_t)ROWS * (size_t)OUT;
    float *x12_data;
    float *w_data;
    float *grad_data;
    float *seq_y;
    float *fused_y;
    float *seq_dx12;
    float *fused_dx12;
    float *seq_dw;
    float *fused_dw;
    gd_memory_config cfg = test_config((x12_count + w_count + y_count) * sizeof(unsigned short));
    gd_context *ctx = NULL;
    gd_tensor x12;
    gd_tensor w;
    gd_tensor grad;
    gd_tensor act;
    gd_tensor y;
    gd_tensor dx12;
    gd_tensor dw;
    size_t i;

    x12_data = (float *)malloc(x12_count * sizeof(float));
    w_data = (float *)malloc(w_count * sizeof(float));
    grad_data = (float *)malloc(y_count * sizeof(float));
    seq_y = (float *)malloc(y_count * sizeof(float));
    fused_y = (float *)malloc(y_count * sizeof(float));
    seq_dx12 = (float *)malloc(x12_count * sizeof(float));
    fused_dx12 = (float *)malloc(x12_count * sizeof(float));
    seq_dw = (float *)malloc(w_count * sizeof(float));
    fused_dw = (float *)malloc(w_count * sizeof(float));
    CHECK(x12_data != NULL && w_data != NULL && grad_data != NULL && seq_y != NULL &&
              fused_y != NULL && seq_dx12 != NULL && fused_dx12 != NULL && seq_dw != NULL &&
              fused_dw != NULL,
          "allocation");
    for (i = 0U; i < x12_count; ++i) {
        x12_data[i] = ((float)((int)(i % 17U) - 8)) * 0.03f;
    }
    for (i = 0U; i < w_count; ++i) {
        w_data[i] = ((float)((int)(i % 13U) - 6)) * 0.02f;
    }
    for (i = 0U; i < y_count; ++i) {
        grad_data[i] = ((float)((int)(i % 11U) - 5)) * 0.015f;
    }

    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, x12_shape), 256U, &x12));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, w_shape), 256U, &w));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, y_shape), 256U, &grad));
    CHECK_OK(gd_tensor_write_f32(ctx, &x12, x12_data, x12_count));
    CHECK_OK(gd_tensor_write_f32(ctx, &w, w_data, w_count));
    CHECK_OK(gd_tensor_write_f32(ctx, &grad, grad_data, y_count));
    CHECK_OK(gd_context_seal_params(ctx));
    x12.requires_grad = true;
    w.requires_grad = true;

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_powlu_split(ctx, &x12, 2.0f, &act));
    CHECK_OK(gd_linear(ctx, &act, &w, NULL, &y));
    CHECK_OK(gd_backward(ctx, &y, &grad));
    CHECK_OK(gd_tensor_grad(ctx, &x12, &dx12));
    CHECK_OK(gd_tensor_grad(ctx, &w, &dw));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, seq_y, y_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx12, seq_dx12, x12_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &dw, seq_dw, w_count));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_powlu_split_linear(ctx, &x12, &w, NULL, 2.0f, &y));
    CHECK_OK(gd_backward(ctx, &y, &grad));
    CHECK_OK(gd_tensor_grad(ctx, &x12, &dx12));
    CHECK_OK(gd_tensor_grad(ctx, &w, &dw));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, fused_y, y_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx12, fused_dx12, x12_count));
    CHECK_OK(gd_tensor_read_f32(ctx, &dw, fused_dw, w_count));

    check_close_array(fused_y, seq_y, y_count, 1.0e-3f, "forward");
    check_close_array(fused_dx12, seq_dx12, x12_count, 3.0e-3f, "dx12");
    check_close_array(fused_dw, seq_dw, w_count, 1.0e-3f, "dw");

    gd_context_destroy(ctx);
    free(fused_dw);
    free(seq_dw);
    free(fused_dx12);
    free(seq_dx12);
    free(fused_y);
    free(seq_y);
    free(grad_data);
    free(w_data);
    free(x12_data);
    printf("test_powlu_split_linear: ok\n");
    return 0;
}
