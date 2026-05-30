#include "gradients/gradients.h"

#include <stdio.h>
#include <string.h>

#define CHECK_OK(expr)                                                            \
    do {                                                                          \
        gd_status status_ = (expr);                                               \
        if (status_ != GD_OK) {                                                   \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());          \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_STATUS(expr, expected)                                               \
    do {                                                                          \
        gd_status status_ = (expr);                                               \
        if (status_ != (expected)) {                                               \
            fprintf(stderr, "%s got %s expected %s; last_error=%s\n",             \
                    #expr,                                                        \
                    gd_status_name(status_),                                      \
                    gd_status_name(expected),                                     \
                    gd_last_error());                                             \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_TRUE(expr)                                                          \
    do {                                                                          \
        if (!(expr)) {                                                            \
            fprintf(stderr, "%s failed\n", #expr);                              \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static int test_status(void)
{
    CHECK_TRUE(strcmp(gd_status_name(GD_OK), "GD_OK") == 0);
    CHECK_TRUE(strcmp(gd_status_message(GD_OK), "ok") == 0);
    CHECK_TRUE(strcmp(gd_status_name(GD_ERR_INVALID_ARGUMENT),
                      "GD_ERR_INVALID_ARGUMENT") == 0);
    CHECK_TRUE(strcmp(gd_status_name(GD_ERR_OUT_OF_MEMORY),
                      "GD_ERR_OUT_OF_MEMORY") == 0);
    CHECK_TRUE(strcmp(gd_status_name(GD_ERR_UNSUPPORTED),
                      "GD_ERR_UNSUPPORTED") == 0);
    CHECK_TRUE(strcmp(gd_status_name(GD_ERR_BACKEND), "GD_ERR_BACKEND") == 0);
    CHECK_TRUE(strcmp(gd_status_name(GD_ERR_DTYPE), "GD_ERR_DTYPE") == 0);
    CHECK_TRUE(strcmp(gd_status_name(GD_ERR_SHAPE), "GD_ERR_SHAPE") == 0);
    CHECK_TRUE(strcmp(gd_status_name(GD_ERR_DEVICE), "GD_ERR_DEVICE") == 0);
    CHECK_TRUE(strcmp(gd_status_name(GD_ERR_INVALID_STATE),
                      "GD_ERR_INVALID_STATE") == 0);
    CHECK_TRUE(strcmp(gd_status_name(GD_ERR_IO), "GD_ERR_IO") == 0);
    CHECK_TRUE(strcmp(gd_status_name(GD_ERR_INTERNAL), "GD_ERR_INTERNAL") == 0);
    CHECK_TRUE(strcmp(gd_status_name((gd_status)9999), "GD_ERR_UNKNOWN") == 0);
    return 0;
}

static int test_dtype(void)
{
    gd_compute_policy policy = gd_compute_policy_default();

    CHECK_TRUE(policy.compute_dtype == GD_DTYPE_F32);
    CHECK_TRUE(policy.accum_dtype == GD_DTYPE_F32);

    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_INVALID) == 0U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_BOOL) == 1U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_I8) == 1U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_U8) == 1U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_I16) == 2U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_U16) == 2U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_I32) == 4U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_U32) == 4U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_I64) == 8U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_U64) == 8U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_F16) == 2U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_BF16) == 2U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_F32) == 4U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_FP8_E4M3) == 1U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_FP8_E5M2) == 1U);
    CHECK_TRUE(gd_dtype_sizeof(GD_DTYPE_QUANTIZED) == 0U);

    CHECK_TRUE(strcmp(gd_dtype_name(GD_DTYPE_F32), "f32") == 0);
    CHECK_TRUE(strcmp(gd_dtype_name(GD_DTYPE_QUANTIZED), "quantized") == 0);
    CHECK_TRUE(strcmp(gd_dtype_name((gd_dtype)9999), "unknown") == 0);
    return 0;
}

static int test_device(void)
{
    gd_device cpu0 = {GD_DEVICE_CPU, 0};
    gd_device cpu1 = {GD_DEVICE_CPU, 1};
    gd_device metal0 = {GD_DEVICE_METAL, 0};

    CHECK_TRUE(gd_device_equal(cpu0, cpu0));
    CHECK_TRUE(!gd_device_equal(cpu0, cpu1));
    CHECK_TRUE(!gd_device_equal(cpu0, metal0));
    CHECK_TRUE(strcmp(gd_device_type_name(GD_DEVICE_CPU), "CPU") == 0);
    CHECK_TRUE(strcmp(gd_device_type_name(GD_DEVICE_METAL), "METAL") == 0);
    CHECK_TRUE(strcmp(gd_device_type_name(GD_DEVICE_CUDA), "CUDA") == 0);
    CHECK_TRUE(strcmp(gd_device_type_name(GD_DEVICE_VULKAN), "VULKAN") == 0);
    CHECK_TRUE(strcmp(gd_device_type_name((gd_device_type)9999), "UNKNOWN") == 0);
    return 0;
}

static int test_context(void)
{
    gd_context *ctx = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_device unsupported_cpu_index = {GD_DEVICE_CPU, 1};
    /* Vulkan has no backend registered (Metal may be auto-registered on macOS). */
    gd_device unsupported = {GD_DEVICE_VULKAN, 0};
    gd_compute_policy policy;

    CHECK_STATUS(gd_context_create(NULL), GD_ERR_INVALID_ARGUMENT);
    CHECK_TRUE(strstr(gd_last_error(), "GD_ERR_INVALID_ARGUMENT") != NULL);

    CHECK_OK(gd_context_create(&ctx));
    CHECK_TRUE(ctx != NULL);
    CHECK_TRUE(gd_device_equal(gd_context_default_device(ctx), cpu));
    CHECK_TRUE(gd_context_fallback_policy(ctx) == GD_FALLBACK_NONE);

    policy = gd_context_compute_policy(ctx);
    CHECK_TRUE(policy.compute_dtype == GD_DTYPE_F32);
    CHECK_TRUE(policy.accum_dtype == GD_DTYPE_F32);

    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_CPU_REF));
    CHECK_TRUE(gd_context_fallback_policy(ctx) == GD_FALLBACK_CPU_REF);
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_NONE));
    CHECK_TRUE(gd_context_fallback_policy(ctx) == GD_FALLBACK_NONE);
    CHECK_STATUS(gd_context_set_fallback_policy(ctx, (gd_fallback_policy)9999),
                 GD_ERR_INVALID_ARGUMENT);

    policy = (gd_compute_policy){GD_DTYPE_BF16, GD_DTYPE_F32};
    CHECK_OK(gd_context_set_compute_policy(ctx, policy));
    policy = gd_context_compute_policy(ctx);
    CHECK_TRUE(policy.compute_dtype == GD_DTYPE_BF16);
    CHECK_TRUE(policy.accum_dtype == GD_DTYPE_F32);

    policy = (gd_compute_policy){GD_DTYPE_INVALID, GD_DTYPE_INVALID};
    CHECK_OK(gd_context_set_compute_policy(ctx, policy));
    policy = gd_context_compute_policy(ctx);
    CHECK_TRUE(policy.compute_dtype == GD_DTYPE_F32);
    CHECK_TRUE(policy.accum_dtype == GD_DTYPE_F32);

    policy = (gd_compute_policy){GD_DTYPE_I32, GD_DTYPE_F32};
    CHECK_STATUS(gd_context_set_compute_policy(ctx, policy), GD_ERR_DTYPE);

    CHECK_OK(gd_context_set_default_device(ctx, cpu));
    CHECK_STATUS(gd_context_set_default_device(ctx, unsupported_cpu_index),
                 GD_ERR_UNSUPPORTED);
    CHECK_STATUS(gd_context_set_default_device(ctx, unsupported), GD_ERR_UNSUPPORTED);
    CHECK_STATUS(gd_synchronize(ctx, unsupported_cpu_index), GD_ERR_UNSUPPORTED);
    CHECK_STATUS(gd_synchronize(ctx, unsupported), GD_ERR_UNSUPPORTED);
    CHECK_OK(gd_synchronize(ctx, cpu));

    CHECK_TRUE(gd_context_fallback_policy(NULL) == GD_FALLBACK_NONE);
    CHECK_TRUE(gd_context_compute_policy(NULL).compute_dtype == GD_DTYPE_F32);
    CHECK_TRUE(gd_device_equal(gd_context_default_device(NULL), cpu));
    CHECK_STATUS(gd_context_set_default_device(NULL, cpu), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_context_set_fallback_policy(NULL, GD_FALLBACK_NONE),
                 GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_context_set_compute_policy(NULL, gd_compute_policy_default()),
                 GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_synchronize(NULL, cpu), GD_ERR_INVALID_ARGUMENT);

    gd_context_destroy(ctx);
    gd_context_destroy(NULL);
    return 0;
}

int main(void)
{
    if (test_status() != 0) {
        return 1;
    }
    if (test_dtype() != 0) {
        return 1;
    }
    if (test_device() != 0) {
        return 1;
    }
    if (test_context() != 0) {
        return 1;
    }
    return 0;
}
