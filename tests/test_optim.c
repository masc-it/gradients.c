#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_optim failed: %s (%s:%d)\n", (msg),         \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static gd_memory_config optim_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 8192U;
    cfg.state_bytes = 16384U;
    cfg.scratch_slot_bytes = 8192U;
    cfg.data_slot_bytes = 4096U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static int close_to(float got, float want, float tol)
{
    return fabsf(got - want) <= tol;
}

static void run_lr_scheduler_value_test(void)
{
    gd_lr_scheduler_config cfg = gd_lr_scheduler_config_default();
    float lr = -1.0f;
    float expect_mid;

    cfg.max_lr = 0.1f;
    cfg.min_lr = 0.01f;
    cfg.warmup_steps = 10U;
    cfg.total_steps = 110U;

    CHECK_OK(gd_lr_scheduler_value(&cfg, 0U, &lr));
    CHECK(close_to(lr, 0.0f, 1.0e-7f), "scheduler starts warmup at zero");
    CHECK_OK(gd_lr_scheduler_value(&cfg, 5U, &lr));
    CHECK(close_to(lr, 0.05f, 1.0e-7f), "scheduler warmup midpoint");
    CHECK_OK(gd_lr_scheduler_value(&cfg, 10U, &lr));
    CHECK(close_to(lr, cfg.max_lr, 1.0e-7f), "scheduler warmup boundary");
    CHECK_OK(gd_lr_scheduler_value(&cfg, 60U, &lr));
    expect_mid = cfg.min_lr + 0.5f * (cfg.max_lr - cfg.min_lr);
    CHECK(close_to(lr, expect_mid, 1.0e-7f), "scheduler cosine midpoint");
    CHECK_OK(gd_lr_scheduler_value(&cfg, 110U, &lr));
    CHECK(close_to(lr, cfg.min_lr, 1.0e-7f), "scheduler reaches min lr");
    CHECK_OK(gd_lr_scheduler_value(&cfg, 999U, &lr));
    CHECK(close_to(lr, cfg.min_lr, 1.0e-7f), "scheduler clamps after horizon");

    cfg.min_lr = 0.2f;
    CHECK(gd_lr_scheduler_value(&cfg, 0U, &lr) == GD_ERR_INVALID_ARGUMENT,
          "scheduler rejects min_lr > max_lr");
    cfg.min_lr = 0.01f;
    cfg.total_steps = 0U;
    CHECK(gd_lr_scheduler_value(&cfg, 0U, &lr) == GD_ERR_INVALID_ARGUMENT,
          "scheduler rejects zero total steps");
    cfg.total_steps = 4U;
    cfg.warmup_steps = 5U;
    CHECK(gd_lr_scheduler_value(&cfg, 0U, &lr) == GD_ERR_INVALID_ARGUMENT,
          "scheduler rejects warmup beyond total steps");
}

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

static void run_adamw_dynamic_lr_test(void)
{
    gd_context *ctx = NULL;
    gd_optimizer *opt = NULL;
    gd_memory_config mem = optim_config();
    gd_adamw_config adam = gd_adamw_config_default();
    gd_tensor param;
    gd_tensor y;
    gd_param_ref ref;
    gd_param_set set;
    int64_t shape[1] = {1};
    const uint16_t initial = (uint16_t)0x3c00U;
    uint16_t got = 0U;

    adam.lr = 0.0f;
    adam.beta1 = 0.0f;
    adam.beta2 = 0.0f;
    adam.eps = 1.0e-8f;
    adam.weight_decay = 0.0f;
    adam.bias_correction = true;

    CHECK_OK(gd_context_create(&mem, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &param));
    CHECK_OK(gd_tensor_write(ctx, &param, &initial, sizeof(initial)));
    param.requires_grad = true;

    gd_param_set_init(&set);
    memset(&ref, 0, sizeof(ref));
    (void)snprintf(ref.path, sizeof(ref.path), "%s", "param");
    ref.tensor = &param;
    ref.group_index = -1;
    ref.lr_mult = 1.0f;
    ref.weight_decay = adam.weight_decay;
    ref.trainable = true;
    set.items = &ref;
    set.count = 1U;
    set.capacity = 1U;

    CHECK_OK(gd_adamw_create(ctx, &set, &adam, &opt));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_relu(ctx, &param, &y));
    CHECK_OK(gd_backward(ctx, &y, NULL));
    CHECK_OK(gd_optimizer_step_lr(ctx, opt, 0.1f));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &param, &got, sizeof(got)));
    CHECK(got == f32_to_f16_bits(0.9f), "explicit lr drives first adamw update");
    CHECK(gd_optimizer_step_count(opt) == 1U, "explicit lr increments optimizer step");

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_relu(ctx, &param, &y));
    CHECK_OK(gd_backward(ctx, &y, NULL));
    CHECK_OK(gd_optimizer_step_lr(ctx, opt, 0.01f));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &param, &got, sizeof(got)));
    CHECK(got == f32_to_f16_bits(0.89f), "explicit lr drives second adamw update");
    CHECK(gd_optimizer_step_count(opt) == 2U, "second explicit lr increments optimizer step");

    gd_optimizer_destroy(opt);
    gd_context_destroy(ctx);
}

static void expect_adamw_first_step(uint16_t *out,
                                    const float *param,
                                    const float *grad,
                                    uint32_t count,
                                    const gd_adamw_config *cfg)
{
    uint32_t i;
    for (i = 0U; i < count; ++i) {
        float m = grad[i];
        float v = grad[i] * grad[i];
        float update = m / (sqrtf(v) + cfg->eps);
        float next = param[i] - cfg->lr * (update + cfg->weight_decay * param[i]);
        out[i] = f32_to_f16_bits(next);
    }
}

static void run_amp_adamw_test(void)
{
    gd_context *ctx = NULL;
    gd_optimizer *opt = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_memory_config mem = optim_config();
    gd_adamw_config adam = gd_adamw_config_default();
    gd_adamw_config expected_adam;
    gd_amp_config amp = gd_amp_config_default();
    gd_tensor param;
    gd_tensor grad_seed;
    gd_tensor y;
    gd_param_ref ref;
    gd_param_set set;
    int64_t shape[1] = {3};
    const float p0_f32[3] = {1.0f, 2.0f, 3.0f};
    const float g0_f32[3] = {0.1f, -0.2f, 0.3f};
    uint16_t p0[3];
    uint16_t grad_bits[3];
    uint16_t got[3];
    uint16_t want[3];
    uint16_t before_overflow[3];
    const float dynamic_lr = 1.0e-2f;
    uint32_t i;

    adam.lr = 0.0f;
    adam.beta1 = 0.9f;
    adam.beta2 = 0.999f;
    adam.eps = 1.0e-8f;
    adam.weight_decay = 1.0e-2f;
    adam.bias_correction = true;
    expected_adam = adam;
    expected_adam.lr = dynamic_lr;
    amp.init_scale = 8.0f;
    amp.growth_interval = 1U;
    amp.max_scale = 1024.0f;

    CHECK_OK(gd_context_create(&mem, &ctx));
    for (i = 0U; i < 3U; ++i) {
        p0[i] = f32_to_f16_bits(p0_f32[i]);
        grad_bits[i] = f32_to_f16_bits(g0_f32[i]);
    }
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &param));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &grad_seed));
    CHECK_OK(gd_tensor_write(ctx, &param, p0, sizeof(p0)));
    CHECK_OK(gd_tensor_write(ctx, &grad_seed, grad_bits, sizeof(grad_bits)));
    param.requires_grad = true;

    gd_param_set_init(&set);
    memset(&ref, 0, sizeof(ref));
    (void)snprintf(ref.path, sizeof(ref.path), "%s", "param");
    ref.tensor = &param;
    ref.group_index = -1;
    ref.lr_mult = 1.0f;
    ref.weight_decay = adam.weight_decay;
    ref.trainable = true;
    set.items = &ref;
    set.count = 1U;
    set.capacity = 1U;

    CHECK_OK(gd_adamw_create(ctx, &set, &adam, &opt));
    CHECK_OK(gd_amp_scaler_create(&amp, &scaler));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_relu(ctx, &param, &y));
    CHECK_OK(gd_backward_scaled(ctx, &y, &grad_seed, gd_amp_scaler_scale(scaler)));
    CHECK_OK(gd_optimizer_step_amp_lr(ctx, opt, scaler, dynamic_lr));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &param, got, sizeof(got)));
    expect_adamw_first_step(want, p0_f32, g0_f32, 3U, &expected_adam);
    for (i = 0U; i < 3U; ++i) {
        CHECK(got[i] == want[i], "amp adamw finite update matches expected f16 writeback");
        before_overflow[i] = got[i];
    }
    CHECK(gd_optimizer_step_count(opt) == 1U, "optimizer step increments after finite amp update");
    CHECK(!gd_amp_scaler_last_found_inf(scaler), "finite amp step reports no overflow");
    CHECK(gd_amp_scaler_scale(scaler) == 16.0f, "amp scale grows after finite interval");

    grad_bits[0] = f32_to_f16_bits(INFINITY);
    grad_bits[1] = f32_to_f16_bits(1.0f);
    grad_bits[2] = f32_to_f16_bits(-1.0f);
    CHECK_OK(gd_tensor_write(ctx, &grad_seed, grad_bits, sizeof(grad_bits)));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_relu(ctx, &param, &y));
    CHECK_OK(gd_backward_scaled(ctx, &y, &grad_seed, gd_amp_scaler_scale(scaler)));
    CHECK_OK(gd_optimizer_step_amp_lr(ctx, opt, scaler, dynamic_lr));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &param, got, sizeof(got)));
    for (i = 0U; i < 3U; ++i) {
        CHECK(got[i] == before_overflow[i], "amp adamw skips update on overflow");
    }
    CHECK(gd_optimizer_step_count(opt) == 1U, "optimizer step does not increment on skipped amp update");
    CHECK(gd_amp_scaler_last_found_inf(scaler), "overflow amp step reports found inf");
    CHECK(gd_amp_scaler_scale(scaler) == 8.0f, "amp scale backs off after overflow");

    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(opt);
    gd_context_destroy(ctx);
}

static void run_amp_adamw_grad_clip_test(void)
{
    gd_context *ctx = NULL;
    gd_optimizer *opt = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_memory_config mem = optim_config();
    gd_adamw_config adam = gd_adamw_config_default();
    gd_adamw_config expected_adam;
    gd_amp_config amp = gd_amp_config_default();
    gd_tensor param;
    gd_tensor grad_seed;
    gd_tensor y;
    gd_param_ref ref;
    gd_param_set set;
    int64_t shape[1] = {2};
    const float p0_f32[2] = {1.0f, 2.0f};
    const float g0_f32[2] = {3.0f, 4.0f};
    float clipped_g[2];
    uint16_t p0[2];
    uint16_t grad_bits[2];
    uint16_t got[2];
    uint16_t want[2];
    const float dynamic_lr = 1.0e-1f;
    const float max_norm = 0.5f;
    const float clip_scale = max_norm / (5.0f + 1.0e-6f);
    uint32_t i;

    adam.lr = 0.0f;
    adam.beta1 = 0.0f;
    adam.beta2 = 0.0f;
    adam.eps = 1.0f;
    adam.weight_decay = 0.0f;
    adam.bias_correction = true;
    expected_adam = adam;
    expected_adam.lr = dynamic_lr;
    amp.init_scale = 8.0f;
    amp.growth_interval = 1000U;

    CHECK_OK(gd_context_create(&mem, &ctx));
    for (i = 0U; i < 2U; ++i) {
        p0[i] = f32_to_f16_bits(p0_f32[i]);
        grad_bits[i] = f32_to_f16_bits(g0_f32[i]);
        clipped_g[i] = g0_f32[i] * clip_scale;
    }
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &param));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &grad_seed));
    CHECK_OK(gd_tensor_write(ctx, &param, p0, sizeof(p0)));
    CHECK_OK(gd_tensor_write(ctx, &grad_seed, grad_bits, sizeof(grad_bits)));
    param.requires_grad = true;

    gd_param_set_init(&set);
    memset(&ref, 0, sizeof(ref));
    (void)snprintf(ref.path, sizeof(ref.path), "%s", "param");
    ref.tensor = &param;
    ref.group_index = -1;
    ref.lr_mult = 1.0f;
    ref.weight_decay = adam.weight_decay;
    ref.trainable = true;
    set.items = &ref;
    set.count = 1U;
    set.capacity = 1U;

    CHECK_OK(gd_adamw_create(ctx, &set, &adam, &opt));
    CHECK_OK(gd_amp_scaler_create(&amp, &scaler));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_relu(ctx, &param, &y));
    CHECK_OK(gd_backward_scaled(ctx, &y, &grad_seed, gd_amp_scaler_scale(scaler)));
    CHECK_OK(gd_optimizer_step_amp_clip_lr(ctx, opt, scaler, dynamic_lr, max_norm));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &param, got, sizeof(got)));
    expect_adamw_first_step(want, p0_f32, clipped_g, 2U, &expected_adam);
    for (i = 0U; i < 2U; ++i) {
        CHECK(got[i] == want[i], "amp adamw grad clip scales global norm before update");
    }
    CHECK(gd_optimizer_step_count(opt) == 1U, "clipped amp adamw increments optimizer step");
    CHECK(!gd_amp_scaler_last_found_inf(scaler), "clipped finite amp step reports no overflow");

    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(opt);
    gd_context_destroy(ctx);
}

int main(void)
{
    run_lr_scheduler_value_test();
    run_adamw_dynamic_lr_test();
    run_amp_adamw_test();
    run_amp_adamw_grad_clip_test();
    printf("test_optim: ok\n");
    return 0;
}
