/*
 * gd_split Metal performance probe.
 *
 * Run from the repository root with:
 *   make op-perf OP=split
 *
 * Optional environment:
 *   GD_SPLIT_PERF_PROFILE=smoke|all|<case-name>
 *   GD_SPLIT_PERF_WARMUP=10
 *   GD_SPLIT_PERF_ITERS=100
 */

#include "../_shared/perf/partition_perf.h"

int main(void)
{
    static const gd_partition_perf_case cases[] = {
        {.name = "qkv_bth3d_axis2_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 5U,
         .axis = 2,
         .n_parts = 3U,
         .shape = {4, 2048, 3, 16, 64},
         .part_sizes = {1, 1, 1},
         .smoke = true},
        {.name = "qkv_flat_last_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 3U,
         .axis = -1,
         .n_parts = 3U,
         .shape = {4, 1024, 3072},
         .part_sizes = {1024, 1024, 1024},
         .smoke = true},
        {.name = "mlp_branches_last_f32",
         .dtype = GD_DTYPE_F32,
         .rank = 2U,
         .axis = -1,
         .n_parts = 2U,
         .shape = {4096, 4096},
         .part_sizes = {1024, 3072},
         .smoke = true},
        {.name = "vision_qkv_axis1_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 5U,
         .axis = 1,
         .n_parts = 3U,
         .shape = {1, 3, 196, 12, 64},
         .part_sizes = {1, 1, 1}},
        {.name = "token_ids_last_i32",
         .dtype = GD_DTYPE_I32,
         .rank = 3U,
         .axis = -1,
         .n_parts = 3U,
         .shape = {64, 2048, 3},
         .part_sizes = {1, 1, 1}},
        {.name = "image_rgb_alpha_u8",
         .dtype = GD_DTYPE_U8,
         .rank = 3U,
         .axis = -1,
         .n_parts = 2U,
         .shape = {1024, 1024, 4},
         .part_sizes = {3, 1}},
    };
    static const gd_partition_perf_spec spec = {
        .tag = "SPLIT",
        .profile_env = "GD_SPLIT_PERF_PROFILE",
        .warmup_env = "GD_SPLIT_PERF_WARMUP",
        .iters_env = "GD_SPLIT_PERF_ITERS",
        .note = "materialized-contiguous-outputs",
        .bandwidth_label = "effective",
        .kind = GD_PARTITION_PERF_SPLIT,
    };
    return gd_partition_perf_main(&spec, cases, GD_PERF_ARRAY_LEN(cases));
}
