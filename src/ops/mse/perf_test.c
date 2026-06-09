/*
 * gd_mse Metal performance probe.
 *
 * Run from the repository root with:
 *   make op-perf OP=mse
 *
 * Optional environment:
 *   GD_MSE_PERF_PROFILE=smoke|all|<case-name>
 *   GD_MSE_PERF_WARMUP=10
 *   GD_MSE_PERF_ITERS=100
 */

#include "../_shared/perf/pairwise_loss_perf.h"

int main(void)
{
    static const gd_pairwise_loss_perf_case cases[] = {
        {.name = "tail_1x513_f16", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {1, 513}, .smoke = true},
        {.name = "activation_256x1024_f16", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {256, 1024}, .smoke = true},
        {.name = "image_32x3x224x224_f16", .dtype = GD_DTYPE_F16, .rank = 4U, .shape = {32, 3, 224, 224}},
        {.name = "activation_4096x4096_f16", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {4096, 4096}},
        {.name = "regression_64x1024_f32", .dtype = GD_DTYPE_F32, .rank = 2U, .shape = {64, 1024}, .smoke = true},
        {.name = "regression_1024x4096_f32", .dtype = GD_DTYPE_F32, .rank = 2U, .shape = {1024, 4096}},
    };
    static const gd_pairwise_loss_perf_spec spec = {
        .tag = "MSE",
        .profile_env = "GD_MSE_PERF_PROFILE",
        .warmup_env = "GD_MSE_PERF_WARMUP",
        .iters_env = "GD_MSE_PERF_ITERS",
        .x_seed = UINT64_C(0x4d5345),
        .y_seed = UINT64_C(0x4d5346),
        .random_min = -1.0f,
        .random_max = 1.0f,
        .forward = gd_mse,
        .backward = gd_mse_backward,
    };
    return gd_pairwise_loss_perf_main(&spec, cases, GD_PERF_ARRAY_LEN(cases));
}
