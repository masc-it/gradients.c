/*
 * FP16 ReLU public API performance probe.
 *
 * Run with:
 *   make op-perf OP=relu
 *
 * Optional environment:
 *   GD_RELU_PERF_PROFILE=all|smoke|<case-name>
 *   GD_RELU_PERF_WARMUP=10
 *   GD_RELU_PERF_ITERS=100
 *
 * Times include public validation, scratch allocation, Metal encoding, command
 * buffer submission, and synchronization. Bandwidth is reported as a logical
 * lower bound for the kernel work; the autograd pair also reports an estimated
 * public-training bandwidth that includes grad-slot zero/accumulate traffic.
 */

#include "../_shared/perf/unary_perf.h"

int main(void)
{
    static const gd_unary_perf_case cases[] = {
        {.name = "tail_1x513", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {1, 513}, .smoke = true},
        {.name = "bert_4096x768", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {4096, 768}, .smoke = true},
        {.name = "gpt_4096x4096", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {4096, 4096}},
        {.name = "mlp_2048x11008", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {2048, 11008}},
    };
    static const gd_unary_perf_spec spec = {
        .tag = "RELU",
        .profile_env = "GD_RELU_PERF_PROFILE",
        .warmup_env = "GD_RELU_PERF_WARMUP",
        .iters_env = "GD_RELU_PERF_ITERS",
        .title = "FP16 public API perf",
        .note = "logical bytes: fwd=read_x+write_y, bwd=read_x+read_grad+write_dx; "
                "autograd estimate includes grad zero/accumulate traffic.",
        .x_seed = 1234U,
        .grad_seed = 5678U,
        .x_min = -1.0f,
        .x_max = 1.0f,
        .grad_min = -0.5f,
        .grad_max = 0.5f,
        .state_bytes = 4U * 1024U * 1024U,
        .data_slot_bytes = 4U * 1024U * 1024U,
        .autograd_estimated_public_multiplier = 13.0,
        .forward = gd_relu,
        .backward = gd_relu_backward,
    };
    return gd_unary_perf_main(&spec, cases, GD_PERF_ARRAY_LEN(cases));
}
