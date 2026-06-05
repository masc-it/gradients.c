#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct xor_mlp {
    gd_module mod;
    gd_linear_layer fc1;
    gd_linear_layer fc2;
} xor_mlp;

static void fail_status(gd_context *ctx, gd_status st, const char *expr, int line)
{
    fprintf(stderr,
            "mlp_xor failed at line %d: %s -> %s (%s)\n",
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

static uint16_t f32_to_f16_bits(float value)
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

static float f16_bits_to_f32(uint16_t bits)
{
    uint32_t sign = ((uint32_t)bits & 0x8000U) << 16;
    int32_t exp = (int32_t)(((uint32_t)bits >> 10) & 0x1fU);
    uint32_t mant = (uint32_t)bits & 0x3ffU;
    union {
        uint32_t u;
        float f;
    } v;
    if (exp == 0) {
        if (mant == 0U) {
            v.u = sign;
            return v.f;
        }
        while ((mant & 0x400U) == 0U) {
            mant <<= 1U;
            exp -= 1;
        }
        mant &= 0x3ffU;
        exp += 1;
    } else if (exp == 31) {
        v.u = sign | 0x7f800000U | (mant << 13);
        return v.f;
    }
    v.u = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (mant << 13);
    return v.f;
}

static void pack_f16(const float *src, uint16_t *dst, size_t count)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        dst[i] = f32_to_f16_bits(src[i]);
    }
}

static gd_memory_config xor_memory_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 64U * 1024U;
    cfg.state_bytes = 128U * 1024U;
    cfg.scratch_slot_bytes = 256U * 1024U;
    cfg.data_slot_bytes = 16U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 3U;
    cfg.default_alignment = 256U;
    return cfg;
}

static void write_initial_weights(gd_context *ctx, xor_mlp *model)
{
    static const float fc1_w_f32[2U * 4U] = {
         0.80f,  0.80f, -0.30f,  0.20f,
         0.80f,  0.80f,  0.25f, -0.25f,
    };
    static const float fc1_b_f32[4U] = {-0.10f, -0.90f, 0.0f, 0.0f};
    static const float fc2_w_f32[4U * 1U] = {1.20f, -1.70f, 0.10f, -0.10f};
    static const float fc2_b_f32[1U] = {0.0f};
    uint16_t fc1_w[2U * 4U];
    uint16_t fc1_b[4U];
    uint16_t fc2_w[4U * 1U];
    uint16_t fc2_b[1U];
    pack_f16(fc1_w_f32, fc1_w, 2U * 4U);
    pack_f16(fc1_b_f32, fc1_b, 4U);
    pack_f16(fc2_w_f32, fc2_w, 4U * 1U);
    pack_f16(fc2_b_f32, fc2_b, 1U);
    TRY(ctx, gd_tensor_write(ctx, &model->fc1.weight, fc1_w, sizeof(fc1_w)));
    TRY(ctx, gd_tensor_write(ctx, &model->fc1.bias, fc1_b, sizeof(fc1_b)));
    TRY(ctx, gd_tensor_write(ctx, &model->fc2.weight, fc2_w, sizeof(fc2_w)));
    TRY(ctx, gd_tensor_write(ctx, &model->fc2.bias, fc2_b, sizeof(fc2_b)));
}

static void xor_mlp_init(gd_context *ctx, xor_mlp *model)
{
    gd_linear_layer_config fc1_cfg;
    gd_linear_layer_config fc2_cfg;
    memset(model, 0, sizeof(*model));
    fc1_cfg = gd_linear_layer_config_make(2, 4, GD_DTYPE_F16, 11U);
    fc2_cfg = gd_linear_layer_config_make(4, 1, GD_DTYPE_F16, 22U);
    TRY(ctx, gd_module_init(ctx, &model->mod, "xor_mlp"));
    TRY(ctx, gd_linear_layer_init_child(ctx, &model->mod, "fc1", &model->fc1, &fc1_cfg));
    TRY(ctx, gd_linear_layer_init_child(ctx, &model->mod, "fc2", &model->fc2, &fc2_cfg));
    write_initial_weights(ctx, model);
}

static void xor_mlp_deinit(xor_mlp *model)
{
    gd_linear_layer_deinit(&model->fc2);
    gd_linear_layer_deinit(&model->fc1);
    gd_module_deinit(&model->mod);
}

static gd_status xor_mlp_forward(gd_context *ctx,
                                 xor_mlp *model,
                                 const gd_tensor *x,
                                 gd_tensor *out)
{
    gd_tensor h;
    gd_tensor a;
    gd_status st;
    st = gd_linear_layer_forward(ctx, &model->fc1, x, &h);
    if (st != GD_OK) {
        return st;
    }
    st = gd_relu(ctx, &h, &a);
    if (st != GD_OK) {
        return st;
    }
    return gd_linear_layer_forward(ctx, &model->fc2, &a, out);
}

static float mean_squared_error(const float *pred, const float *target, size_t count)
{
    float loss = 0.0f;
    size_t i;
    for (i = 0U; i < count; ++i) {
        float d = pred[i] - target[i];
        loss += d * d;
    }
    return loss / (float)count;
}

static void print_param_set(const gd_param_set *params)
{
    uint32_t i;
    printf("parameters:\n");
    for (i = 0U; i < params->count; ++i) {
        printf("  %-22s lr_mult=%.2f weight_decay=%.4f trainable=%s\n",
               params->items[i].path,
               (double)params->items[i].lr_mult,
               (double)params->items[i].weight_decay,
               params->items[i].trainable ? "yes" : "no");
    }
}

int main(void)
{
    enum { BATCH = 4, IN = 2, OUT = 1, STEPS = 240 };
    static const float x_f32[BATCH * IN] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
    };
    static const float y_f32[BATCH * OUT] = {0.0f, 1.0f, 1.0f, 0.0f};
    gd_context *ctx = NULL;
    gd_optimizer *optimizer = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_memory_config mem = xor_memory_config();
    gd_adamw_config adam = gd_adamw_config_default();
    gd_amp_config amp = gd_amp_config_default();
    gd_param_group groups[2];
    gd_param_set params;
    xor_mlp model;
    gd_tensor x;
    gd_tensor target;
    const int64_t x_shape[2] = {BATCH, IN};
    const int64_t y_shape[2] = {BATCH, OUT};
    uint16_t x_f16[BATCH * IN];
    uint16_t y_f16[BATCH * OUT];
    uint16_t pred_f16[BATCH * OUT];
    float pred[BATCH * OUT];
    float final_loss;
    int step;
    int correct;
    gd_status st;

    st = gd_context_create(&mem, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("mlp_xor: skipped (no supported gradients.c backend)\n");
        return 0;
    }
    if (st != GD_OK) {
        fail_status(ctx, st, "gd_context_create", __LINE__);
    }

    xor_mlp_init(ctx, &model);

    pack_f16(x_f32, x_f16, BATCH * IN);
    pack_f16(y_f32, y_f16, BATCH * OUT);
    TRY(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, x_shape, 256U, &x));
    TRY(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, y_shape, 256U, &target));
    TRY(ctx, gd_tensor_write(ctx, &x, x_f16, sizeof(x_f16)));
    TRY(ctx, gd_tensor_write(ctx, &target, y_f16, sizeof(y_f16)));
    x.requires_grad = false;
    target.requires_grad = false;

    memset(groups, 0, sizeof(groups));
    groups[0].name = "hidden";
    groups[0].match = "xor_mlp.fc1.*";
    groups[0].lr_mult = 1.0f;
    groups[0].weight_decay = 0.0f;
    groups[0].trainable = true;
    groups[1].name = "head";
    groups[1].match = "xor_mlp.fc2.*";
    groups[1].lr_mult = 1.0f;
    groups[1].weight_decay = 0.0f;
    groups[1].trainable = true;
    TRY(ctx, gd_module_collect_params(ctx, &model.mod, groups, 2U, &params));
    print_param_set(&params);

    adam.lr = 5.0e-2f;
    adam.weight_decay = 0.0f;
    amp.init_scale = 64.0f;
    amp.growth_interval = 32U;
    amp.max_scale = 4096.0f;
    TRY(ctx, gd_adamw_create(ctx, &params, &adam, &optimizer));
    TRY(ctx, gd_amp_scaler_create(&amp, &scaler));
    TRY(ctx, gd_context_seal_params(ctx));
    gd_module_set_training(&model.mod, true);

    for (step = 0; step < STEPS; ++step) {
        gd_tensor pred_tensor;
        gd_tensor diff;
        gd_tensor sq;
        gd_tensor loss;
        uint16_t loss_bits = 0U;
        float loss_value = 0.0f;
        int report = step == 0 || (step + 1) % 40 == 0;
        TRY(ctx, gd_begin(ctx, GD_SCOPE_TRAIN));
        TRY(ctx, xor_mlp_forward(ctx, &model, &x, &pred_tensor));
        TRY(ctx, gd_sub(ctx, &pred_tensor, &target, &diff));
        TRY(ctx, gd_mul(ctx, &diff, &diff, &sq));
        TRY(ctx, gd_reduce_mean(ctx, &sq, &loss));
        TRY(ctx, gd_backward_scaled(ctx, &loss, NULL, gd_amp_scaler_scale(scaler)));
        TRY(ctx, gd_optimizer_step_amp(ctx, optimizer, scaler));
        TRY(ctx, gd_end(ctx));

        if (report) {
            TRY(ctx, gd_tensor_read(ctx, &loss, &loss_bits, sizeof(loss_bits)));
            loss_value = f16_bits_to_f32(loss_bits);
            printf("step=%3d loss=%.6f scale=%.1f overflow=%s\n",
                   step + 1,
                   (double)loss_value,
                   (double)gd_amp_scaler_scale(scaler),
                   gd_amp_scaler_last_found_inf(scaler) ? "yes" : "no");
        }
    }

    gd_module_set_training(&model.mod, false);
    TRY(ctx, gd_begin(ctx, GD_SCOPE_EVAL));
    {
        gd_tensor pred_tensor;
        size_t i;
        TRY(ctx, xor_mlp_forward(ctx, &model, &x, &pred_tensor));
        TRY(ctx, gd_end(ctx));
        TRY(ctx, gd_tensor_read(ctx, &pred_tensor, pred_f16, sizeof(pred_f16)));
        for (i = 0U; i < BATCH * OUT; ++i) {
            pred[i] = f16_bits_to_f32(pred_f16[i]);
        }
    }

    final_loss = mean_squared_error(pred, y_f32, BATCH * OUT);
    correct = 0;
    printf("\nfinal predictions:\n");
    for (step = 0; step < BATCH; ++step) {
        int label = pred[step] >= 0.5f ? 1 : 0;
        int target = y_f32[step] >= 0.5f ? 1 : 0;
        if (label == target) {
            correct += 1;
        }
        printf("  [%.0f %.0f] -> %.4f class=%d target=%d\n",
               (double)x_f32[step * IN + 0],
               (double)x_f32[step * IN + 1],
               (double)pred[step],
               label,
               target);
    }
    printf("final_loss=%.6f accuracy=%d/%d optimizer_steps=%llu\n",
           (double)final_loss,
           correct,
           BATCH,
           (unsigned long long)gd_optimizer_step_count(optimizer));

    if (correct != BATCH || final_loss > 0.05f) {
        fprintf(stderr, "mlp_xor: training did not solve XOR\n");
        gd_amp_scaler_destroy(scaler);
        gd_optimizer_destroy(optimizer);
        gd_param_set_free(&params);
        xor_mlp_deinit(&model);
        gd_context_destroy(ctx);
        return 1;
    }

    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(optimizer);
    gd_param_set_free(&params);
    xor_mlp_deinit(&model);
    gd_context_destroy(ctx);
    printf("mlp_xor: ok\n");
    return 0;
}
