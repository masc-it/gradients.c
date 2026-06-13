#!/usr/bin/env sh
# Build and run v2 QA stress probes without editing Makefile.

set -u

BUILD_DIR=${BUILD_DIR:-build-qa-probes}
CC=${CC:-cc}
OBJC=${OBJC:-clang}
PYTHON=${PYTHON:-python3}
RUN_PERF=${RUN_PERF:-1}
RUN_ELEM_PERF=${RUN_ELEM_PERF:-1}
RUN_METAL_BASELINE=${RUN_METAL_BASELINE:-1}
PERF_BUILD_DIR=${PERF_BUILD_DIR:-$BUILD_DIR-perf}
PERF_WARMUP=${GD_QA_PERF_WARMUP:-2}
PERF_ITERS=${GD_QA_PERF_ITERS:-5}
PERF_CFLAGS=${PERF_CFLAGS:--std=c11 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wdouble-promotion -Wstrict-prototypes -Wmissing-prototypes -Wno-unused-parameter}
PERF_OBJCFLAGS=${PERF_OBJCFLAGS:--Iinclude -DGD_ENABLE_METAL=1 -O3 -DNDEBUG -Wall -Wextra -Werror -fobjc-arc}
PERF_PROBE_CFLAGS=${PERF_PROBE_CFLAGS:--std=c11 -O3 -DNDEBUG -Wall -Wextra -Werror -Wpedantic}
STATUS=0
UNAME_S=$(uname -s)

case "$UNAME_S" in
    Darwin)
        API_LDLIBS="-framework Foundation -framework Metal"
        ;;
    *)
        API_LDLIBS=""
        ;;
esac

run_step() {
    printf '\n[qa] %s\n' "$*"
    "$@"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        printf '[qa] step failed rc=%s: %s\n' "$rc" "$*"
        STATUS=1
    fi
}

mkdir -p "$BUILD_DIR/probes"
run_step make BUILD_DIR="$BUILD_DIR" build

run_step "$CC" -Iinclude -std=c11 -Wall -Wextra -Werror -Wpedantic \
    probes/v2_api_contract_stress_probe.c "$BUILD_DIR/libgradients.a" \
    $API_LDLIBS -o "$BUILD_DIR/probes/v2_api_contract_stress_probe"

run_step "$CC" -std=c11 -Wall -Wextra -Werror -Wpedantic \
    probes/v2_semantics_oracle_probe.c -lm \
    -o "$BUILD_DIR/probes/v2_semantics_oracle_probe"

if [ "$RUN_PERF" != 0 ]; then
    mkdir -p "$PERF_BUILD_DIR/probes"
    run_step make BUILD_DIR="$PERF_BUILD_DIR" CFLAGS="$PERF_CFLAGS" \
        OBJCFLAGS="$PERF_OBJCFLAGS" build
    run_step "$CC" -Iinclude $PERF_PROBE_CFLAGS \
        probes/v2_matmul_training_perf_probe.c "$PERF_BUILD_DIR/libgradients.a" \
        $API_LDLIBS -o "$PERF_BUILD_DIR/probes/v2_matmul_training_perf_probe"
fi

if [ "$RUN_ELEM_PERF" != 0 ]; then
    mkdir -p "$PERF_BUILD_DIR/probes"
    run_step make BUILD_DIR="$PERF_BUILD_DIR" CFLAGS="$PERF_CFLAGS" \
        OBJCFLAGS="$PERF_OBJCFLAGS" build
    run_step "$CC" -Iinclude $PERF_PROBE_CFLAGS \
        probes/v2_elementwise_reduce_perf_probe.c "$PERF_BUILD_DIR/libgradients.a" \
        $API_LDLIBS -o "$PERF_BUILD_DIR/probes/v2_elementwise_reduce_perf_probe"
fi

if [ "$RUN_METAL_BASELINE" != 0 ] && [ "$UNAME_S" = "Darwin" ]; then
    mkdir -p "$PERF_BUILD_DIR/probes"
    run_step "$OBJC" -O3 -DNDEBUG -fobjc-arc -Wall -Wextra -Werror \
        -framework Foundation -framework Metal -framework MetalPerformanceShaders \
        probes/v2_metal_arena_probe.m -o "$PERF_BUILD_DIR/probes/v2_metal_arena_probe"
fi

if [ -x "$BUILD_DIR/probes/v2_api_contract_stress_probe" ]; then
    run_step env \
        GRADIENTS_METALLIB="$BUILD_DIR/gradients.metallib" \
        "$BUILD_DIR/probes/v2_api_contract_stress_probe"
fi

if [ -x "$BUILD_DIR/probes/v2_semantics_oracle_probe" ]; then
    run_step "$BUILD_DIR/probes/v2_semantics_oracle_probe"
fi

if [ "$RUN_PERF" != 0 ] && [ -x "$PERF_BUILD_DIR/probes/v2_matmul_training_perf_probe" ]; then
    run_step env \
        GRADIENTS_METALLIB="$PERF_BUILD_DIR/gradients.metallib" \
        GD_QA_PERF_WARMUP="$PERF_WARMUP" \
        GD_QA_PERF_ITERS="$PERF_ITERS" \
        "$PERF_BUILD_DIR/probes/v2_matmul_training_perf_probe"
fi

if [ "$RUN_ELEM_PERF" != 0 ] && [ -x "$PERF_BUILD_DIR/probes/v2_elementwise_reduce_perf_probe" ]; then
    run_step env \
        GRADIENTS_METALLIB="$PERF_BUILD_DIR/gradients.metallib" \
        GD_QA_ELEM_WARMUP="${GD_QA_ELEM_WARMUP:-$PERF_WARMUP}" \
        GD_QA_ELEM_ITERS="${GD_QA_ELEM_ITERS:-$PERF_ITERS}" \
        GD_QA_ELEM_PROFILE="${GD_QA_ELEM_PROFILE:-all}" \
        "$PERF_BUILD_DIR/probes/v2_elementwise_reduce_perf_probe"
fi

if [ "$RUN_METAL_BASELINE" != 0 ] && [ -x "$PERF_BUILD_DIR/probes/v2_metal_arena_probe" ]; then
    run_step env \
        GD_PROBE_MPS_WARMUP="${GD_PROBE_MPS_WARMUP:-$PERF_WARMUP}" \
        GD_PROBE_MPS_ITERS="${GD_PROBE_MPS_ITERS:-$PERF_ITERS}" \
        GD_PROBE_BENCH_PROFILE="${GD_PROBE_BENCH_PROFILE:-256h4}" \
        "$PERF_BUILD_DIR/probes/v2_metal_arena_probe"
fi

if command -v "$PYTHON" >/dev/null 2>&1; then
    run_step "$PYTHON" probes/v2_design_spec_audit.py
else
    printf '[qa] skipped static audit: python not found (%s)\n' "$PYTHON"
fi

if [ "$RUN_PERF" != 0 ] || [ "$RUN_ELEM_PERF" != 0 ]; then
    printf '\n[qa] performance report: inspect [PERF]/[ELEM][PERF] lines above for optimized public API metallib throughput (PERF_BUILD_DIR=%s).\n' "$PERF_BUILD_DIR"
    if [ "$RUN_METAL_BASELINE" != 0 ] && [ "$UNAME_S" = "Darwin" ]; then
        printf '[qa] raw Metal/MPS baseline used GD_PROBE_BENCH_PROFILE=%s (override for all profiles).\n' "${GD_PROBE_BENCH_PROFILE:-256h4}"
    fi
fi

printf '\n[qa] done status=%s\n' "$STATUS"
exit "$STATUS"
