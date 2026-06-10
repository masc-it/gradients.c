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

static void run_amp_scaler_state_roundtrip_test(void)
{
    gd_amp_config amp = gd_amp_config_default();
    gd_amp_scaler *scaler_a = NULL;
    gd_amp_scaler *scaler_b = NULL;
    gd_amp_scaler_state state;

    amp.init_scale = 16.0f;
    amp.growth_interval = 3U;
    CHECK_OK(gd_amp_scaler_create(&amp, &scaler_a));
    CHECK_OK(gd_amp_scaler_create(NULL, &scaler_b));
    CHECK_OK(gd_amp_scaler_get_state(scaler_a, &state));
    state.scale = 64.0f;
    state.growth_tracker = 2U;
    state.last_found_inf = true;
    state.config.defer_found_inf = true;
    CHECK_OK(gd_amp_scaler_set_state(scaler_b, &state));
    CHECK(gd_amp_scaler_scale(scaler_b) == 64.0f, "scaler state restores scale");
    CHECK(gd_amp_scaler_growth_tracker(scaler_b) == 2U, "scaler state restores tracker");
    CHECK(gd_amp_scaler_last_found_inf(scaler_b), "scaler state restores overflow flag");

    gd_amp_scaler_destroy(scaler_b);
    gd_amp_scaler_destroy(scaler_a);
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

static void run_checkpoint_adamw_step(gd_context *ctx,
                                      gd_optimizer *opt,
                                      gd_tensor *param,
                                      gd_tensor *grad_seed,
                                      float lr)
{
    gd_tensor y;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_relu(ctx, param, &y));
    CHECK_OK(gd_backward(ctx, &y, grad_seed));
    CHECK_OK(gd_optimizer_step_lr(ctx, opt, lr));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
}

static void run_adamw_checkpoint_state_test(void)
{
    const char *path = "build/test_optimizer_state.gdckpt";
    gd_memory_config mem = optim_config();
    gd_adamw_config adam = gd_adamw_config_default();
    gd_context *ctx_a = NULL;
    gd_context *ctx_b = NULL;
    gd_optimizer *opt_a = NULL;
    gd_optimizer *opt_b = NULL;
    gd_tensor param_a;
    gd_tensor grad_seed_a;
    gd_tensor param_b;
    gd_tensor grad_seed_b;
    gd_param_ref ref_a;
    gd_param_ref ref_b;
    gd_param_set set_a;
    gd_param_set set_b;
    int64_t shape[1] = {2};
    const float p0_f32[2] = {1.0f, 2.0f};
    const float g0_f32[2] = {0.125f, -0.25f};
    uint16_t p0[2];
    uint16_t grad_bits[2];
    uint16_t after_step1[2];
    uint16_t got_a[2];
    uint16_t got_b[2];
    uint32_t i;

    (void)remove(path);
    adam.lr = 0.0f;
    adam.beta1 = 0.9f;
    adam.beta2 = 0.99f;
    adam.eps = 1.0e-6f;
    adam.weight_decay = 1.0e-2f;
    adam.bias_correction = true;
    for (i = 0U; i < 2U; ++i) {
        p0[i] = f32_to_f16_bits(p0_f32[i]);
        grad_bits[i] = f32_to_f16_bits(g0_f32[i]);
    }

    CHECK_OK(gd_context_create(&mem, &ctx_a));
    CHECK_OK(gd_tensor_empty(ctx_a, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &param_a));
    CHECK_OK(gd_tensor_empty(ctx_a, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &grad_seed_a));
    CHECK_OK(gd_tensor_write(ctx_a, &param_a, p0, sizeof(p0)));
    CHECK_OK(gd_tensor_write(ctx_a, &grad_seed_a, grad_bits, sizeof(grad_bits)));
    param_a.requires_grad = true;
    gd_param_set_init(&set_a);
    memset(&ref_a, 0, sizeof(ref_a));
    (void)snprintf(ref_a.path, sizeof(ref_a.path), "%s", "param");
    ref_a.tensor = &param_a;
    ref_a.group_index = -1;
    ref_a.lr_mult = 1.0f;
    ref_a.weight_decay = adam.weight_decay;
    ref_a.trainable = true;
    set_a.items = &ref_a;
    set_a.count = 1U;
    set_a.capacity = 1U;
    CHECK_OK(gd_adamw_create(ctx_a, &set_a, &adam, &opt_a));
    CHECK_OK(gd_context_seal_params(ctx_a));
    run_checkpoint_adamw_step(ctx_a, opt_a, &param_a, &grad_seed_a, 1.0e-2f);
    CHECK_OK(gd_tensor_read(ctx_a, &param_a, after_step1, sizeof(after_step1)));
    CHECK_OK(gd_optimizer_save_state(ctx_a, opt_a, path));
    run_checkpoint_adamw_step(ctx_a, opt_a, &param_a, &grad_seed_a, 1.0e-2f);
    CHECK_OK(gd_tensor_read(ctx_a, &param_a, got_a, sizeof(got_a)));
    CHECK(gd_optimizer_step_count(opt_a) == 2U, "reference optimizer reached step 2");

    CHECK_OK(gd_context_create(&mem, &ctx_b));
    CHECK_OK(gd_tensor_empty(ctx_b, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &param_b));
    CHECK_OK(gd_tensor_empty(ctx_b, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &grad_seed_b));
    CHECK_OK(gd_tensor_write(ctx_b, &param_b, after_step1, sizeof(after_step1)));
    CHECK_OK(gd_tensor_write(ctx_b, &grad_seed_b, grad_bits, sizeof(grad_bits)));
    param_b.requires_grad = true;
    gd_param_set_init(&set_b);
    memset(&ref_b, 0, sizeof(ref_b));
    (void)snprintf(ref_b.path, sizeof(ref_b.path), "%s", "param");
    ref_b.tensor = &param_b;
    ref_b.group_index = -1;
    ref_b.lr_mult = 1.0f;
    ref_b.weight_decay = adam.weight_decay;
    ref_b.trainable = true;
    set_b.items = &ref_b;
    set_b.count = 1U;
    set_b.capacity = 1U;
    CHECK_OK(gd_adamw_create(ctx_b, &set_b, &adam, &opt_b));
    CHECK_OK(gd_optimizer_load_state(ctx_b, opt_b, path, true));
    CHECK(gd_optimizer_step_count(opt_b) == 1U, "loaded optimizer restores step count");
    CHECK_OK(gd_context_seal_params(ctx_b));
    run_checkpoint_adamw_step(ctx_b, opt_b, &param_b, &grad_seed_b, 1.0e-2f);
    CHECK_OK(gd_tensor_read(ctx_b, &param_b, got_b, sizeof(got_b)));
    for (i = 0U; i < 2U; ++i) {
        CHECK(got_b[i] == got_a[i], "loaded optimizer continuation matches uninterrupted run");
    }
    CHECK(gd_optimizer_step_count(opt_b) == 2U, "loaded optimizer advances from restored step");

    gd_optimizer_destroy(opt_b);
    gd_context_destroy(ctx_b);
    gd_optimizer_destroy(opt_a);
    gd_context_destroy(ctx_a);
    (void)remove(path);
}

int main(void)
{
    run_lr_scheduler_value_test();
    run_adamw_dynamic_lr_test();
    run_amp_adamw_test();
    run_amp_scaler_state_roundtrip_test();
    run_amp_adamw_grad_clip_test();
    run_adamw_checkpoint_state_test();
    printf("test_optim: ok\n");
    return 0;
}
