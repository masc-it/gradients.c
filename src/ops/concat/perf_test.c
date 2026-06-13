/*
 * gd_concat Metal performance probe.
 *
 * Run from the repository root with:
 *   make op-perf OP=concat
 *
 * Optional environment:
 *   GD_CONCAT_PERF_PROFILE=smoke|all|<case-name>
 *   GD_CONCAT_PERF_WARMUP=10
 *   GD_CONCAT_PERF_ITERS=100
 */

#include "../_shared/perf/partition_perf.h"

int main(void)
{
    static const gd_partition_perf_case cases[] = {
        {.name = "tokens_axis0_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 2U,
         .axis = 0,
         .n_parts = 2U,
         .shape = {0, 1024},
         .part_sizes = {4096, 4096},
         .smoke = true},
        {.name = "features_axis1_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 2U,
         .axis = 1,
         .n_parts = 3U,
         .shape = {8192, 0},
         .part_sizes = {256, 512, 256},
         .smoke = true},
        {.name = "features_axis1_f32",
         .dtype = GD_DTYPE_F32,
         .rank = 2U,
         .axis = 1,
         .n_parts = 2U,
         .shape = {2048, 0},
         .part_sizes = {1024, 1024},
         .smoke = true},
        {.name = "heads_axis1_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 3U,
         .axis = 1,
         .n_parts = 2U,
         .shape = {4096, 0, 64},
         .part_sizes = {8, 8}},
        {.name = "three_way_axis0_f32",
         .dtype = GD_DTYPE_F32,
         .rank = 2U,
         .axis = 0,
         .n_parts = 3U,
         .shape = {0, 2048},
         .part_sizes = {1024, 2048, 1024}},
        {.name = "token_ids_axis0_i32",
         .dtype = GD_DTYPE_I32,
         .rank = 1U,
         .axis = 0,
         .n_parts = 3U,
         .shape = {0},
         .part_sizes = {65536, 65536, 32768}},
    };
    static const gd_partition_perf_spec spec = {
        .tag = "CONCAT",
        .profile_env = "GD_CONCAT_PERF_PROFILE",
        .warmup_env = "GD_CONCAT_PERF_WARMUP",
        .iters_env = "GD_CONCAT_PERF_ITERS",
        .bandwidth_label = "logical",
        .kind = GD_PARTITION_PERF_CONCAT,
    };
    return gd_partition_perf_main(&spec, cases, GD_PERF_ARRAY_LEN(cases));
}
