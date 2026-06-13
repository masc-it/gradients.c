#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "test_backward_null_outputs failed: %s (%s:%d)\n", \
                    (msg), __FILE__, __LINE__);                                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static gd_memory_config test_config(void)
{
    gd_memory_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = 16U * 1024U * 1024U;
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = 8U * 1024U * 1024U;
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static gd_tensor make_tensor(gd_context *ctx,
                             gd_dtype dtype,
                             uint32_t rank,
                             const int64_t *shape)
{
    gd_tensor tensor;
    CHECK_OK(gd_tensor_empty(ctx,
                             GD_ARENA_PARAMS,
                             dtype,
                             gd_shape_make(rank, shape),
                             256U,
                             &tensor));
    return tensor;
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = test_config();
    const int64_t shape_2x3[2] = {2, 3};
    const int64_t shape_3x4[2] = {3, 4};
    const int64_t shape_2x4[2] = {2, 4};
    const int64_t shape_2[1] = {2};
    const int64_t shape_3[1] = {3};
    const int64_t shape_6[1] = {6};
    const int64_t shape_2x6[2] = {2, 6};
    const int64_t shape_3x2[2] = {3, 2};
    const int64_t shape_table[2] = {5, 3};
    const int64_t shape_ids[1] = {2};
    const int64_t shape_qkv[2] = {2, 12};
    const int64_t shape_heads[3] = {2, 1, 4};
    const int64_t shape_cu[1] = {2};
    const int64_t shape_scalar[1] = {0};
    const int32_t axes[2] = {1, 0};
    const int64_t split_sizes[2] = {3, 3};
    gd_tensor x;
    gd_tensor x_alt;
    gd_tensor w;
    gd_tensor g_2x4;
    gd_tensor scalar_f16;
    gd_tensor axis_grad;
    gd_tensor weight;
    gd_tensor x12;
    gd_tensor table;
    gd_tensor ids;
    gd_tensor flat_grad;
    gd_tensor perm_grad;
    gd_tensor concat_grad;
    gd_tensor qkv;
    gd_tensor pos;
    gd_tensor q;
    gd_tensor cu;
    gd_tensor grad_outputs_storage[2];
    const gd_tensor *concat_inputs[2];
    const gd_tensor *split_grads[2];
    gd_rope_config rope = {.theta = 10000.0f, .n_dims = 4, .interleaved = false};
    gd_sdpa_varlen_config sdpa = {.scale = 0.5f,
                                  .causal = true,
                                  .sliding_window = 0,
                                  .prefix_len = 0,
                                  .max_seqlen = 2};

    CHECK_OK(gd_context_create(&cfg, &ctx));

    x = make_tensor(ctx, GD_DTYPE_F16, 2U, shape_2x3);
    x_alt = make_tensor(ctx, GD_DTYPE_F16, 2U, shape_2x3);
    w = make_tensor(ctx, GD_DTYPE_F16, 2U, shape_3x4);
    g_2x4 = make_tensor(ctx, GD_DTYPE_F16, 2U, shape_2x4);
    scalar_f16 = make_tensor(ctx, GD_DTYPE_F16, 0U, shape_scalar);
    axis_grad = make_tensor(ctx, GD_DTYPE_F16, 1U, shape_2);
    weight = make_tensor(ctx, GD_DTYPE_F16, 1U, shape_3);
    x12 = make_tensor(ctx, GD_DTYPE_F16, 2U, shape_2x6);
    table = make_tensor(ctx, GD_DTYPE_F16, 2U, shape_table);
    ids = make_tensor(ctx, GD_DTYPE_I32, 1U, shape_ids);
    flat_grad = make_tensor(ctx, GD_DTYPE_F16, 1U, shape_6);
    perm_grad = make_tensor(ctx, GD_DTYPE_F16, 2U, shape_3x2);
    concat_grad = make_tensor(ctx, GD_DTYPE_F16, 2U, shape_2x6);
    qkv = make_tensor(ctx, GD_DTYPE_F16, 2U, shape_qkv);
    pos = make_tensor(ctx, GD_DTYPE_I32, 1U, shape_ids);
    q = make_tensor(ctx, GD_DTYPE_F16, 3U, shape_heads);
    cu = make_tensor(ctx, GD_DTYPE_I32, 1U, shape_cu);
    grad_outputs_storage[0] = x;
    grad_outputs_storage[1] = x_alt;

    CHECK_OK(gd_relu_backward(ctx, &x, &x, NULL));
    CHECK_OK(gd_dropout_backward(ctx, &x, &x, 0.25f, 123U, NULL));
    CHECK_OK(gd_reduce_sum_backward(ctx, &x, &scalar_f16, NULL));
    CHECK_OK(gd_reduce_sum_axis_backward(ctx, &x, &axis_grad, 1, false, NULL));
    CHECK_OK(gd_matmul_backward(ctx, &x, &w, &g_2x4, NULL, NULL));
    CHECK_OK(gd_linear_backward(ctx, &x, &w, NULL, &g_2x4, NULL, NULL, NULL));
    CHECK_OK(gd_linear_transposed_weight_backward(ctx, &g_2x4, &w, NULL, &x, NULL, NULL, NULL));
    CHECK_OK(gd_rms_norm_backward(ctx, &x, &weight, &x, 1.0e-5f, NULL, NULL));
    CHECK_OK(gd_powlu_split_backward(ctx, &x12, &x, 2.0f, NULL));
    CHECK_OK(gd_embedding_backward(ctx, &table, &ids, &x, NULL));
    CHECK_OK(gd_reshape_backward(ctx, &x, &flat_grad, NULL));
    CHECK_OK(gd_permute_backward(ctx, &x, &perm_grad, axes, 2U, NULL));

    concat_inputs[0] = &x;
    concat_inputs[1] = &x_alt;
    CHECK_OK(gd_concat_backward(ctx, &concat_grad, concat_inputs, 2U, 1, NULL));

    split_grads[0] = &grad_outputs_storage[0];
    split_grads[1] = &grad_outputs_storage[1];
    CHECK_OK(gd_split_backward(ctx, &concat_grad, split_grads, split_sizes, 2U, 1, NULL));

    CHECK_OK(gd_qkv_split_rope_backward(ctx, &qkv, &pos, &q, &q, &q, 1, 4, &rope, NULL));
    CHECK_OK(gd_sdpa_varlen_backward(ctx, &q, &q, &q, &cu, &q, &sdpa, NULL, NULL, NULL));

    gd_context_destroy(ctx);
    printf("test_backward_null_outputs ok\n");
    return 0;
}
