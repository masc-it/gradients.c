# Metal GEMM precision probe

Standalone micro-benchmark comparing register-blocked fp32 vs `simdgroup_matrix`
fp32/bf16 GEMM throughput. Not part of `make`; ad-hoc. See
`docs/plan_gemm_perf_tuning_metal.md` §8.1 for the findings (on M1 Pro the
register-blocked kernel wins; bf16 matrix units are not accelerated).

    clang -fobjc-arc -O2 bench/metal/gemm_bench.m -framework Metal -framework Foundation -o /tmp/gemm_bench
    # edit the @"/tmp/gemm_bf16.metal" path in gemm_bench.m to point at bench/metal/gemm_bf16.metal
    /tmp/gemm_bench 2048 30
