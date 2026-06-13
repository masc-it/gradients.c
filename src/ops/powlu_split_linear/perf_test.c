/*
 * Performance probe scaffold for gd_powlu_split_linear.
 *
 * Run from the repository root with:
 *   make op-perf OP=powlu_split_linear
 *
 * Define realistic cases, allocate one reusable model/context per case,
 * warm up, then report avg_ms plus FLOP/s or logical bandwidth.
 * Keep public API overhead in the timed path.
 *
 * See docs/guides/metal_tips.md before implementing Metal hot paths.
 * Custom backend mode: generated backend stubs are omitted; add custom backend declarations/PSOs manually.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <gradients/gradients.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#else
#include <time.h>
#endif

#define GD_POWLU_SPLIT_LINEAR_PERF_GIB (1024.0 * 1024.0 * 1024.0)

#if defined(__APPLE__)
static double perf_now_seconds(void)
{
    static mach_timebase_info_data_t info;
    static double scale = 0.0;
    if (scale == 0.0) {
        if (mach_timebase_info(&info) != 0 || info.denom == 0U) {
            return 0.0;
        }
        scale = ((double)info.numer / (double)info.denom) * 1.0e-9;
    }
    return (double)mach_absolute_time() * scale;
}
#else
static double perf_now_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}
#endif

static int perf_env_int(const char *name, int fallback, int min_value, int max_value)
{
    const char *value = getenv(name);
    char *end = NULL;
    long parsed;
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }
    if (parsed < (long)min_value) {
        return min_value;
    }
    if (parsed > (long)max_value) {
        return max_value;
    }
    return (int)parsed;
}

typedef struct perf_case {
    const char *name;
    int64_t rows;
    int64_t cols;
} perf_case;

static bool perf_case_selected(const perf_case *pcase, const char *profile)
{
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "all") == 0) {
        return true;
    }
    if (strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "small_tail") == 0 ||
               strcmp(pcase->name, "activation_mid") == 0;
    }
    return strcmp(profile, pcase->name) == 0;
}

int main(void)
{
    static const perf_case cases[] = {
        {"small_tail", 1, 513},
        {"activation_mid", 4096, 768},
        {"activation_large", 4096, 4096},
    };
    const char *profile = getenv("GD_POWLU_SPLIT_LINEAR_PERF_PROFILE");
    int warmup = perf_env_int("GD_POWLU_SPLIT_LINEAR_PERF_WARMUP", 10, 0, 10000);
    int iters = perf_env_int("GD_POWLU_SPLIT_LINEAR_PERF_ITERS", 100, 1, 1000000);
    double timer_start = perf_now_seconds();
    double timer_elapsed;
    bool ran = false;
    size_t i;
    timer_elapsed = perf_now_seconds() - timer_start;
    printf("[POWLU_SPLIT_LINEAR] perf scaffold for gd_powlu_split_linear: warmup=%d iters=%d profile=%s timer_overhead_ms=%.6f\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "all",
           timer_elapsed * 1.0e3);
    printf("[POWLU_SPLIT_LINEAR] TODO: replace scaffold loop with real model allocation and timed gd_powlu_split_linear calls.\n");
    printf("[POWLU_SPLIT_LINEAR] Report avg_ms plus FLOP/s for compute-bound ops or logical_GiB/s for memory-bound ops.\n");
    (void)GD_POWLU_SPLIT_LINEAR_PERF_GIB;
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        size_t elems;
        if (!perf_case_selected(&cases[i], profile)) {
            continue;
        }
        ran = true;
        elems = (size_t)cases[i].rows * (size_t)cases[i].cols;
        printf("[POWLU_SPLIT_LINEAR][TODO] case=%s shape=%lldx%lld elems=%zu\n",
               cases[i].name,
               (long long)cases[i].rows,
               (long long)cases[i].cols,
               elems);
    }
    if (!ran) {
        fprintf(stderr, "[POWLU_SPLIT_LINEAR][FAIL] no cases selected for GD_POWLU_SPLIT_LINEAR_PERF_PROFILE=%s\n",
                profile != NULL ? profile : "(null)");
        return 2;
    }
    return 0;
}
