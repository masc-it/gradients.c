/*
 * gd_reduce_mean Metal performance probe.
 *
 * Run with:
 *   make op-perf OP=reduce_mean
 *
 * Optional environment:
 *   GD_REDUCE_MEAN_PERF_PROFILE=smoke|all|<case-name>
 *   GD_REDUCE_MEAN_PERF_WARMUP=10
 *   GD_REDUCE_MEAN_PERF_ITERS=100
 */

#include "../_shared/perf/reduce_perf.h"

int main(void)
{
    static const gd_reduce_perf_case cases[] = {
        {"tail_1x513_f16_last", GD_DTYPE_F16, 2U, {1, 513}, -1, true},
        {"act_4096x1024_f16_last", GD_DTYPE_F16, 2U, {4096, 1024}, -1, true},
        {"act_4096x1024_f16_axis0", GD_DTYPE_F16, 2U, {4096, 1024}, 0, true},
        {"act_8192x2048_f16_last", GD_DTYPE_F16, 2U, {8192, 2048}, -1, false},
        {"rank3_8x512x1024_f16_last", GD_DTYPE_F16, 3U, {8, 512, 1024}, -1, false},
        {"rank3_8x512x1024_f16_middle", GD_DTYPE_F16, 3U, {8, 512, 1024}, 1, false},
        {"mlp_2048x11008_f16_last", GD_DTYPE_F16, 2U, {2048, 11008}, -1, false},
        {"act_4096x1024_f32_last", GD_DTYPE_F32, 2U, {4096, 1024}, -1, true},
    };
    static const gd_reduce_perf_spec spec = {
        .tag = "REDUCE_MEAN",
        .profile_env = "GD_REDUCE_MEAN_PERF_PROFILE",
        .warmup_env = "GD_REDUCE_MEAN_PERF_WARMUP",
        .iters_env = "GD_REDUCE_MEAN_PERF_ITERS",
        .include_keepdims = true,
        .scalar_f16_uses_f32 = true,
        .all_forward = gd_reduce_mean,
        .all_backward = gd_reduce_mean_backward,
        .axis_forward = gd_reduce_mean_axis,
        .axis_backward = gd_reduce_mean_axis_backward,
    };
    return gd_reduce_perf_main(&spec, cases, GD_PERF_ARRAY_LEN(cases));
}
