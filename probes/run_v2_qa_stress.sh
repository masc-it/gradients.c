#!/usr/bin/env sh
# Build and run v2 QA stress probes without editing Makefile.

set -u

BUILD_DIR=${BUILD_DIR:-build-qa-probes}
CC=${CC:-cc}
PYTHON=${PYTHON:-python3}
STATUS=0

case "$(uname -s)" in
    Darwin)
        API_LDLIBS="-framework Foundation -framework Metal -framework MetalPerformanceShaders"
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

if [ -x "$BUILD_DIR/probes/v2_api_contract_stress_probe" ]; then
    run_step "$BUILD_DIR/probes/v2_api_contract_stress_probe"
fi

if [ -x "$BUILD_DIR/probes/v2_semantics_oracle_probe" ]; then
    run_step "$BUILD_DIR/probes/v2_semantics_oracle_probe"
fi

if command -v "$PYTHON" >/dev/null 2>&1; then
    run_step "$PYTHON" probes/v2_design_spec_audit.py
else
    printf '[qa] skipped static audit: python not found (%s)\n' "$PYTHON"
fi

printf '\n[qa] done status=%s\n' "$STATUS"
exit "$STATUS"
