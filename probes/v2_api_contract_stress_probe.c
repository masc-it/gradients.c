/*
 * v2 API contract stress probe.
 *
 * Links against libgradients and exercises public v2 foundation APIs against
 * docs/design_spec.md invariants. This probe aggregates failures so QA gets a
 * report instead of the first abort.
 */

#include <gradients/gradients.h>

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QA_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

#define QA_EXPECT_OK(qa, expr) qa_expect_status((qa), (expr), GD_OK, #expr)
#define QA_EXPECT_STATUS(qa, expr, want) qa_expect_status((qa), (expr), (want), #expr)

typedef struct qa_counts {
    uint32_t pass;
    uint32_t fail;
    uint32_t skip;
} qa_counts;

static void qa_pass(qa_counts *qa, const char *name)
{
    qa->pass += 1U;
    printf("[PASS] %s\n", name);
}

static void qa_fail(qa_counts *qa, const char *name, const char *detail)
{
    qa->fail += 1U;
    printf("[FAIL] %s: %s\n", name, detail);
}

static void qa_skip(qa_counts *qa, const char *name, const char *detail)
{
    qa->skip += 1U;
    printf("[SKIP] %s: %s\n", name, detail);
}

static void qa_check(qa_counts *qa, bool condition, const char *name, const char *detail)
{
    if (condition) {
        qa_pass(qa, name);
    } else {
        qa_fail(qa, name, detail);
    }
}

static gd_status qa_expect_status(qa_counts *qa,
                                  gd_status got,
                                  gd_status want,
                                  const char *name)
{
    char detail[128];
    if (got == want) {
        qa_pass(qa, name);
    } else {
        snprintf(detail, sizeof(detail), "got %s (%d), want %s (%d)",
                 gd_status_string(got), (int)got, gd_status_string(want), (int)want);
        qa_fail(qa, name, detail);
    }
    return got;
}

static gd_memory_config qa_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 8192U;
    cfg.state_bytes = 8192U;
    cfg.scratch_slot_bytes = 8192U;
    cfg.data_slot_bytes = 4096U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static bool qa_create_context(qa_counts *qa, gd_context **out_ctx, const char *test_name)
{
    gd_memory_config cfg = qa_config();
    gd_status st;
    *out_ctx = NULL;
    st = gd_context_create(&cfg, out_ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        qa_skip(qa, test_name, "no supported GPU backend");
        return false;
    }
    if (st != GD_OK || *out_ctx == NULL) {
        char detail[128];
        snprintf(detail, sizeof(detail), "gd_context_create failed: %s (%d)",
                 gd_status_string(st), (int)st);
        qa_fail(qa, test_name, detail);
        return false;
    }
    return true;
}

static bool f32x4_equal(const float got[4], const float want[4])
{
    uint32_t i;
    for (i = 0U; i < 4U; ++i) {
        if (got[i] != want[i]) {
            return false;
        }
    }
    return true;
}

static void qa_check_f32x4(qa_counts *qa,
                           const char *name,
                           const float got[4],
                           const float want[4])
{
    char detail[192];
    if (f32x4_equal(got, want)) {
        qa_pass(qa, name);
        return;
    }
    snprintf(detail, sizeof(detail),
             "got [%g, %g, %g, %g], want [%g, %g, %g, %g]",
             (double)got[0], (double)got[1], (double)got[2], (double)got[3],
             (double)want[0], (double)want[1], (double)want[2], (double)want[3]);
    qa_fail(qa, name, detail);
}

static void test_context_and_arena_edges(qa_counts *qa)
{
    gd_memory_config bad_cfg = qa_config();
    gd_context *ctx = NULL;
    gd_span a;
    gd_span bad_span;
    gd_memory_stats before;
    gd_memory_stats after;

    bad_cfg.default_alignment = 192U;
    QA_EXPECT_STATUS(qa, gd_context_create(&bad_cfg, &ctx), GD_ERR_INVALID_ARGUMENT);
    qa_check(qa, ctx == NULL, "invalid context config does not publish context",
             "context pointer was published after invalid config");

    if (!qa_create_context(qa, &ctx, "context_and_arena_edges")) {
        return;
    }

    memset(&a, 0, sizeof(a));
    memset(&bad_span, 0xff, sizeof(bad_span));
    QA_EXPECT_STATUS(qa, gd_alloc_scratch(ctx, 16U, 64U, &bad_span), GD_ERR_BAD_STATE);
    qa_check(qa, bad_span.nbytes == 0U && bad_span.slot == -1,
             "scratch alloc outside scope clears output span",
             "bad span retained stale fields");
    gd_context_clear_error(ctx);

    QA_EXPECT_OK(qa, gd_alloc_params(ctx, 1U, 512U, &a));
    qa_check(qa, a.arena == GD_ARENA_PARAMS && a.slot == -1 && (a.offset % 512U) == 0U,
             "params span carries arena metadata and requested alignment",
             "params span metadata/alignment mismatch");

    QA_EXPECT_OK(qa, gd_memory_stats_query(ctx, &before));
    memset(&bad_span, 0xff, sizeof(bad_span));
    QA_EXPECT_STATUS(qa, gd_alloc_params(ctx, 65536U, 256U, &bad_span), GD_ERR_OUT_OF_MEMORY);
    QA_EXPECT_OK(qa, gd_memory_stats_query(ctx, &after));
    qa_check(qa, before.params.offset == after.params.offset,
             "params OOM does not advance offset",
             "params offset advanced after OOM");
    qa_check(qa, bad_span.nbytes == 0U && bad_span.slot == -1,
             "params OOM does not publish span",
             "OOM span was partially published");
    gd_context_clear_error(ctx);

    QA_EXPECT_OK(qa, gd_context_seal_params(ctx));
    QA_EXPECT_OK(qa, gd_memory_stats_query(ctx, &before));
    memset(&bad_span, 0xff, sizeof(bad_span));
    QA_EXPECT_STATUS(qa, gd_alloc_params(ctx, 8U, 8U, &bad_span), GD_ERR_FROZEN);
    QA_EXPECT_OK(qa, gd_memory_stats_query(ctx, &after));
    qa_check(qa, before.params.offset == after.params.offset,
             "sealed params reject allocation without offset change",
             "params offset advanced after sealed allocation");
    qa_check(qa, after.params.sealed, "params sealed flag visible in stats",
             "params stats did not expose sealed state");
    gd_context_clear_error(ctx);

    gd_context_destroy(ctx);
}

static void test_invalid_init_no_arena_leak(qa_counts *qa)
{
    gd_context *ctx = NULL;
    gd_memory_stats before;
    gd_memory_stats after;
    gd_tensor bad;
    const int64_t shape[1] = {4};

    if (!qa_create_context(qa, &ctx, "invalid_init_no_arena_leak")) {
        return;
    }

    memset(&bad, 0xff, sizeof(bad));
    QA_EXPECT_OK(qa, gd_memory_stats_query(ctx, &before));
    QA_EXPECT_STATUS(qa,
                     gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(1U, shape), 64U, 7U, 0.0f, 1.0f, &bad),
                     GD_ERR_UNSUPPORTED);
    QA_EXPECT_OK(qa, gd_memory_stats_query(ctx, &after));
    qa_check(qa, bad.storage.nbytes == 0U,
             "unsupported rand does not publish tensor descriptor",
             "unsupported rand published storage in output descriptor");
    qa_check(qa, before.params.offset == after.params.offset,
             "unsupported rand validates before params allocation",
             "params offset advanced before unsupported rand failure");
    gd_context_clear_error(ctx);

    memset(&bad, 0xff, sizeof(bad));
    QA_EXPECT_OK(qa, gd_memory_stats_query(ctx, &before));
    QA_EXPECT_STATUS(qa,
                     gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(1U, shape), 64U, 9U, 2.0f, 1.0f, &bad),
                     GD_ERR_INVALID_ARGUMENT);
    QA_EXPECT_OK(qa, gd_memory_stats_query(ctx, &after));
    qa_check(qa, bad.storage.nbytes == 0U,
             "invalid-range rand does not publish tensor descriptor",
             "invalid-range rand published storage in output descriptor");
    qa_check(qa, before.params.offset == after.params.offset,
             "invalid-range rand validates before params allocation",
             "params offset advanced before invalid-range rand failure");
    gd_context_clear_error(ctx);

    gd_context_destroy(ctx);
}

static void test_active_scope_blocking_read(qa_counts *qa)
{
    gd_context *ctx = NULL;
    gd_tensor t;
    const int64_t shape[1] = {4};
    const float want[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float got_before_end[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float got_after_end[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (!qa_create_context(qa, &ctx, "active_scope_blocking_read")) {
        return;
    }

    QA_EXPECT_OK(qa, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    QA_EXPECT_OK(qa, gd_tensor_ones(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(1U, shape), 64U, &t));
    QA_EXPECT_OK(qa, gd_tensor_read(ctx, &t, got_before_end, sizeof(got_before_end)));
    qa_check_f32x4(qa,
                   "blocking tensor_read inside active scope observes queued fill",
                   got_before_end,
                   want);
    QA_EXPECT_OK(qa, gd_end_step(ctx));
    QA_EXPECT_OK(qa, gd_tensor_read(ctx, &t, got_after_end, sizeof(got_after_end)));
    qa_check_f32x4(qa, "post-gd_end_step tensor_read observes queued fill", got_after_end, want);

    gd_context_destroy(ctx);
}

static void test_active_scope_write_order(qa_counts *qa)
{
    gd_context *ctx = NULL;
    gd_tensor t;
    const int64_t shape[1] = {4};
    const float src[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float got[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (!qa_create_context(qa, &ctx, "active_scope_write_order")) {
        return;
    }

    QA_EXPECT_OK(qa, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    QA_EXPECT_OK(qa, gd_tensor_ones(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(1U, shape), 64U, &t));
    QA_EXPECT_OK(qa, gd_tensor_write(ctx, &t, src, sizeof(src)));
    QA_EXPECT_OK(qa, gd_end_step(ctx));
    QA_EXPECT_OK(qa, gd_tensor_read(ctx, &t, got, sizeof(got)));
    qa_check_f32x4(qa,
                   "blocking tensor_write after queued op preserves program order",
                   got,
                   src);

    gd_context_destroy(ctx);
}

static void test_ring_generation_and_stale_views(qa_counts *qa)
{
    gd_context *ctx = NULL;
    gd_tensor hidden;
    gd_tensor suffix;
    const int64_t shape[3] = {2, 6, 4};
    int32_t first_slot;
    uint64_t first_generation;

    if (!qa_create_context(qa, &ctx, "ring_generation_and_stale_views")) {
        return;
    }

    QA_EXPECT_OK(qa, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    QA_EXPECT_OK(qa, gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(3U, shape), 64U, &hidden));
    QA_EXPECT_OK(qa, gd_tensor_slice(ctx, &hidden, 1U, 2, 3, &suffix));
    first_slot = hidden.storage.slot;
    first_generation = hidden.storage.generation;
    qa_check(qa, !gd_tensor_is_contiguous(&suffix),
             "middle-dim suffix slice is non-contiguous view",
             "suffix slice was reported contiguous");
    QA_EXPECT_OK(qa, gd_end_step(ctx));

    QA_EXPECT_OK(qa, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    qa_check(qa, gd_debug_current_ring_slot(ctx, GD_ARENA_SCRATCH) != first_slot,
             "second scope advances scratch ring slot",
             "scratch ring reused first slot too early");
    QA_EXPECT_OK(qa, gd_end_step(ctx));

    QA_EXPECT_OK(qa, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    qa_check(qa, gd_debug_current_ring_slot(ctx, GD_ARENA_SCRATCH) == first_slot,
             "third scope selects oldest scratch ring slot",
             "scratch ring did not cycle back to oldest slot");
    qa_check(qa,
             gd_debug_ring_slot_generation(ctx, GD_ARENA_SCRATCH, (uint32_t)first_slot) >
                 first_generation,
             "scratch slot generation bumps before reuse",
             "scratch slot generation did not bump on reuse");
    QA_EXPECT_STATUS(qa, gd_tensor_validate(ctx, &hidden), GD_ERR_BAD_STATE);
    gd_context_clear_error(ctx);
    QA_EXPECT_STATUS(qa, gd_tensor_validate(ctx, &suffix), GD_ERR_BAD_STATE);
    gd_context_clear_error(ctx);
    QA_EXPECT_OK(qa, gd_end_step(ctx));

    gd_context_destroy(ctx);
}

static void test_hot_path_heap_guard_and_transfers(qa_counts *qa)
{
    gd_context *ctx = NULL;
    gd_tensor tokens;
    gd_tensor hidden;
    gd_tensor suffix;
    gd_tensor compact;
    gd_memory_stats stats;
    const int64_t token_shape[2] = {2, 4};
    const int64_t hidden_shape[3] = {2, 6, 4};
    int32_t tokens_src[8] = {1, 1, 2, 3, 5, 8, 13, 21};
    int32_t tokens_dst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t heap_before;

    if (!qa_create_context(qa, &ctx, "hot_path_heap_guard_and_transfers")) {
        return;
    }

    QA_EXPECT_OK(qa, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    heap_before = gd_debug_heap_alloc_count();
    gd_debug_set_heap_guard(true);

    QA_EXPECT_OK(qa, gd_tensor_empty(ctx, GD_ARENA_DATA, GD_DTYPE_I32, gd_shape_make(2U, token_shape), 64U, &tokens));
    QA_EXPECT_OK(qa, gd_tensor_write(ctx, &tokens, tokens_src, sizeof(tokens_src)));
    QA_EXPECT_OK(qa, gd_tensor_read(ctx, &tokens, tokens_dst, sizeof(tokens_dst)));
    qa_check(qa, memcmp(tokens_src, tokens_dst, sizeof(tokens_src)) == 0,
             "data tensor write/read roundtrip in active scope",
             "data tensor write/read mismatch");

    QA_EXPECT_OK(qa, gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(3U, hidden_shape), 64U, &hidden));
    QA_EXPECT_OK(qa, gd_tensor_slice(ctx, &hidden, 1U, 2, 3, &suffix));
    QA_EXPECT_OK(qa, gd_tensor_contiguous(ctx, GD_ARENA_SCRATCH, &suffix, 64U, &compact));
    qa_check(qa, compact.storage.offset != hidden.storage.offset && gd_tensor_is_contiguous(&compact),
             "explicit contiguous allocates distinct packed output",
             "contiguous output reused source storage or stayed non-contiguous");

    gd_debug_set_heap_guard(false);
    qa_check(qa, gd_debug_heap_alloc_count() == heap_before,
             "debug heap guard sees no core heap allocation in tensor hot path",
             "core heap allocation count changed during guarded hot path");
    QA_EXPECT_OK(qa, gd_end_step(ctx));
    QA_EXPECT_OK(qa, gd_memory_stats_query(ctx, &stats));
    qa_check(qa, stats.scratch.max_slot_watermark >= compact.storage.offset + compact.storage.nbytes,
             "scratch watermark includes explicit contiguous output",
             "scratch watermark missed compact output allocation");

    gd_context_destroy(ctx);
}

static void test_dtype_fill_and_rand(qa_counts *qa)
{
    gd_context *ctx = NULL;
    gd_tensor f16_one;
    gd_tensor bf16_one;
    gd_tensor f32_rand_a;
    gd_tensor f32_rand_b;
    const int64_t shape[1] = {4};
    uint16_t f16_bits[4] = {0U, 0U, 0U, 0U};
    uint16_t bf16_bits[4] = {0U, 0U, 0U, 0U};
    float rand_a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float rand_b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t i;
    bool in_range = true;

    if (!qa_create_context(qa, &ctx, "dtype_fill_and_rand")) {
        return;
    }

    QA_EXPECT_OK(qa, gd_tensor_ones(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 64U, &f16_one));
    QA_EXPECT_OK(qa, gd_tensor_read(ctx, &f16_one, f16_bits, sizeof(f16_bits)));
    qa_check(qa, f16_bits[0] == 0x3c00U && f16_bits[1] == 0x3c00U &&
                    f16_bits[2] == 0x3c00U && f16_bits[3] == 0x3c00U,
             "f16 ones uses canonical one bit pattern",
             "f16 ones bit pattern mismatch");

    QA_EXPECT_OK(qa, gd_tensor_ones(ctx, GD_ARENA_PARAMS, GD_DTYPE_BF16, gd_shape_make(1U, shape), 64U, &bf16_one));
    QA_EXPECT_OK(qa, gd_tensor_read(ctx, &bf16_one, bf16_bits, sizeof(bf16_bits)));
    qa_check(qa, bf16_bits[0] == 0x3f80U && bf16_bits[1] == 0x3f80U &&
                    bf16_bits[2] == 0x3f80U && bf16_bits[3] == 0x3f80U,
             "bf16 ones uses canonical one bit pattern",
             "bf16 ones bit pattern mismatch");

    QA_EXPECT_OK(qa, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(1U, shape), 64U, 12345U, -0.25f, 0.75f, &f32_rand_a));
    QA_EXPECT_OK(qa, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(1U, shape), 64U, 12345U, -0.25f, 0.75f, &f32_rand_b));
    QA_EXPECT_OK(qa, gd_tensor_read(ctx, &f32_rand_a, rand_a, sizeof(rand_a)));
    QA_EXPECT_OK(qa, gd_tensor_read(ctx, &f32_rand_b, rand_b, sizeof(rand_b)));
    for (i = 0U; i < (uint32_t)QA_ARRAY_LEN(rand_a); ++i) {
        if (rand_a[i] < -0.25f || rand_a[i] > 0.75f) {
            in_range = false;
        }
    }
    qa_check(qa, memcmp(rand_a, rand_b, sizeof(rand_a)) == 0,
             "rand_uniform deterministic for same seed",
             "same seed produced different bytes");
    qa_check(qa, in_range, "rand_uniform values stay in requested range",
             "rand_uniform produced value outside range");

    gd_context_destroy(ctx);
}

int main(void)
{
    qa_counts qa;
    memset(&qa, 0, sizeof(qa));

    printf("v2_api_contract_stress_probe: start\n");
    test_context_and_arena_edges(&qa);
    test_invalid_init_no_arena_leak(&qa);
    test_active_scope_blocking_read(&qa);
    test_active_scope_write_order(&qa);
    test_ring_generation_and_stale_views(&qa);
    test_hot_path_heap_guard_and_transfers(&qa);
    test_dtype_fill_and_rand(&qa);

    printf("v2_api_contract_stress_probe: summary pass=%" PRIu32 " fail=%" PRIu32 " skip=%" PRIu32 "\n",
           qa.pass, qa.fail, qa.skip);
    return qa.fail == 0U ? 0 : 1;
}
