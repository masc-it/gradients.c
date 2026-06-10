/*
 * Tanh public API performance probe.
 *
 * Run with:
 *   make op-perf OP=tanh
 *
 * Optional environment:
 *   GD_TANH_PERF_PROFILE=all|smoke|<case-name>
 *   GD_TANH_PERF_WARMUP=10
 *   GD_TANH_PERF_ITERS=100
 *
 * Times include public validation, scratch allocation, Metal encoding, command
 * buffer submission, and synchronization. Bandwidth is a logical lower bound;
 * tanh forward/direct backward are SFU-heavy, so Gelem/s is also printed.
 */

#include "../_shared/perf/unary_perf.h"

int main(void)
{
    static const gd_unary_perf_case cases[] = {
        {.name = "tail_1x513_f16", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {1, 513}, .smoke = true},
        {.name = "act_4096x1024_f16", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {4096, 1024}, .smoke = true},
        {.name = "act_8192x2048_f16", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {8192, 2048}},
        {.name = "rank3_8x512x1024_f16", .dtype = GD_DTYPE_F16, .rank = 3U, .shape = {8, 512, 1024}},
        {.name = "rank4_4x16x128x128_f16", .dtype = GD_DTYPE_F16, .rank = 4U, .shape = {4, 16, 128, 128}},
        {.name = "mlp_2048x11008_f16", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {2048, 11008}},
        {.name = "act_4096x1024_f32", .dtype = GD_DTYPE_F32, .rank = 2U, .shape = {4096, 1024}, .smoke = true},
        {.name = "rank3_4x256x1024_f32", .dtype = GD_DTYPE_F32, .rank = 3U, .shape = {4, 256, 1024}},
    };
    static const gd_unary_perf_spec spec = {
        .tag = "TANH",
        .profile_env = "GD_TANH_PERF_PROFILE",
        .warmup_env = "GD_TANH_PERF_WARMUP",
        .iters_env = "GD_TANH_PERF_ITERS",
        .title = "public API perf",
        .note = "logical bytes: fwd=read_x+write_y, direct_bwd=read_x+read_grad+write_dx, "
                "autograd=saved-output backward.",
        .x_seed = 4321U,
        .grad_seed = 8765U,
        .x_min = -6.0f,
        .x_max = 6.0f,
        .grad_min = -0.5f,
        .grad_max = 0.5f,
        .state_bytes = 8U * 1024U * 1024U,
        .data_slot_bytes = 8U * 1024U * 1024U,
        .forward = gd_tanh,
        .backward = gd_tanh_backward,
    };
    return gd_unary_perf_main(&spec, cases, GD_PERF_ARRAY_LEN(cases));
}
