#include <gradients/gradients.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_memory failed: %s (%s:%d)\n", (msg),       \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)
#define CHECK_STATUS(expr, status) CHECK((expr) == (status), #expr)

static gd_memory_config tiny_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 4096U;
    cfg.state_bytes = 8192U;
    cfg.scratch_slot_bytes = 4096U;
    cfg.data_slot_bytes = 2048U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static void test_arena_contract(gd_context *ctx)
{
    gd_span a;
    gd_span b;
    gd_span fail;
    gd_memory_stats before;
    gd_memory_stats after;

    fail.arena = GD_ARENA_PARAMS;
    fail.slot = 123;
    fail.offset = 456U;
    fail.nbytes = 789U;
    fail.generation = 99U;
    fail.buffer = (void *)(uintptr_t)0x1U;
    fail.host_ptr = (void *)(uintptr_t)0x2U;
    CHECK_STATUS(gd_alloc_scratch(ctx, 8U, 8U, &fail), GD_ERR_BAD_STATE);
    CHECK(fail.nbytes == 0U && fail.buffer == NULL && fail.host_ptr == NULL &&
              fail.cookie == 0U && fail.slot == -1,
          "failed scratch alloc clears output span");
    gd_context_clear_error(ctx);

    CHECK_OK(gd_alloc_params(ctx, 1U, 256U, &a));
    CHECK_OK(gd_alloc_params(ctx, 7U, 64U, &b));
    CHECK(a.arena == GD_ARENA_PARAMS && a.slot == -1, "params span metadata");
    CHECK(a.buffer != NULL, "params span exposes backend buffer");
    CHECK(a.offset % 256U == 0U, "params offset 256B aligned");
    if (a.host_ptr != NULL) {
        CHECK(((uintptr_t)a.host_ptr % 256U) == 0U, "params host ptr 256B aligned");
    }
    CHECK(b.offset % 64U == 0U, "params offset 64B aligned");

    CHECK_OK(gd_memory_stats_query(ctx, &before));
    CHECK_STATUS(gd_alloc_params(ctx, 65536U, 256U, &fail), GD_ERR_OUT_OF_MEMORY);
    CHECK_OK(gd_memory_stats_query(ctx, &after));
    CHECK(before.params.offset == after.params.offset, "OOM does not advance params offset");
    gd_context_clear_error(ctx);

    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_memory_stats_query(ctx, &before));
    CHECK_STATUS(gd_alloc_params(ctx, 8U, 8U, &fail), GD_ERR_FROZEN);
    CHECK_OK(gd_memory_stats_query(ctx, &after));
    CHECK(before.params.offset == after.params.offset, "sealed alloc does not advance params offset");
    CHECK(after.params.sealed, "params arena sealed");
    gd_context_clear_error(ctx);
}

static void test_scope_rings_and_state(gd_context *ctx)
{
    gd_state_object *state = NULL;
    gd_span state_span;
    gd_memory_stats stats;
    gd_span scratch;
    gd_span data;
    uint64_t heap_before;
    uint64_t scratch_gen0;
    uint64_t data_gen0;
    int32_t scratch_slot0;
    int32_t data_slot0;
    bool relocated;
    uint64_t waits_before_reset;

    CHECK_OK(gd_state_object_create(ctx, 1024U, 256U, &state));
    CHECK(state != NULL, "state object created");

    heap_before = gd_debug_heap_alloc_count();
    gd_debug_set_heap_guard(true);

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    scratch_slot0 = gd_debug_current_ring_slot(ctx, GD_ARENA_SCRATCH);
    data_slot0 = gd_debug_current_ring_slot(ctx, GD_ARENA_DATA);
    scratch_gen0 = gd_debug_ring_slot_generation(ctx, GD_ARENA_SCRATCH, (uint32_t)scratch_slot0);
    data_gen0 = gd_debug_ring_slot_generation(ctx, GD_ARENA_DATA, (uint32_t)data_slot0);
    CHECK_OK(gd_alloc_scratch(ctx, 512U, 256U, &scratch));
    CHECK_OK(gd_alloc_data(ctx, 128U, 128U, &data));
    CHECK(scratch.arena == GD_ARENA_SCRATCH && scratch.slot == scratch_slot0,
          "scratch span slot metadata");
    CHECK(data.arena == GD_ARENA_DATA && data.slot == data_slot0,
          "data span slot metadata");
    CHECK(scratch.buffer != NULL && data.buffer != NULL, "ring spans carry backend buffers");
    CHECK(scratch.offset % 256U == 0U, "scratch offset aligned");
    CHECK(data.offset % 128U == 0U, "data offset aligned");
    CHECK_OK(gd_state_object_acquire_span(ctx, state, GD_STATE_READ_WRITE, &state_span));
    CHECK(state_span.offset % 256U == 0U, "state object aligned");
    CHECK_OK(gd_end_step(ctx));
    CHECK(gd_debug_state_object_last_use_fence(state) != 0U,
          "state acquire records fence at end");

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(gd_debug_current_ring_slot(ctx, GD_ARENA_SCRATCH) != scratch_slot0,
          "scratch second scope uses next slot");
    CHECK(gd_debug_current_ring_slot(ctx, GD_ARENA_DATA) != data_slot0,
          "data second scope uses next slot");
    CHECK_OK(gd_alloc_scratch(ctx, 256U, 256U, &scratch));
    CHECK_OK(gd_alloc_data(ctx, 64U, 64U, &data));
    CHECK_OK(gd_end_step(ctx));

    CHECK_OK(gd_memory_stats_query(ctx, &stats));
    CHECK(stats.backend_waits == 0U, "two ring slots avoid wait initially");

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK(gd_debug_current_ring_slot(ctx, GD_ARENA_SCRATCH) == scratch_slot0,
          "scratch third scope reuses oldest slot");
    CHECK(gd_debug_current_ring_slot(ctx, GD_ARENA_DATA) == data_slot0,
          "data third scope reuses completed-by-wait slot");
    CHECK(gd_debug_ring_slot_generation(ctx, GD_ARENA_SCRATCH, (uint32_t)scratch_slot0) > scratch_gen0,
          "scratch generation bumps before reuse");
    CHECK(gd_debug_ring_slot_generation(ctx, GD_ARENA_DATA, (uint32_t)data_slot0) > data_gen0,
          "data generation bumps before reuse");
    CHECK_OK(gd_end_step(ctx));

    CHECK_OK(gd_memory_stats_query(ctx, &stats));
    CHECK(stats.backend_waits <= 1U, "ring exhaustion waits only if oldest fence still in flight");

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
    CHECK_OK(gd_state_object_acquire_span(ctx, state, GD_STATE_READ_WRITE, &state_span));
    CHECK_STATUS(gd_state_object_reset(ctx, state, 0U, 0U, &relocated), GD_ERR_BUSY);
    gd_context_clear_error(ctx);
    CHECK_OK(gd_end_step(ctx));
    CHECK(gd_debug_state_object_last_use_fence(state) != 0U,
          "state acquire records fence after reset rejection");

    CHECK_OK(gd_memory_stats_query(ctx, &stats));
    waits_before_reset = stats.backend_waits;
    CHECK_OK(gd_state_object_reset(ctx, state, 0U, 0U, &relocated));
    CHECK(!relocated, "state reset can reuse after wait");
    CHECK_OK(gd_memory_stats_query(ctx, &stats));
    CHECK(stats.backend_waits == waits_before_reset || stats.backend_waits == waits_before_reset + 1U,
          "state reset waits only if fence still in flight");

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
    CHECK_OK(gd_state_object_reset(ctx, state, 0U, 0U, &relocated));
    CHECK(!relocated, "state reset before acquire in active scope can reuse");
    CHECK_OK(gd_state_object_acquire_span(ctx, state, GD_STATE_READ_WRITE, &state_span));
    CHECK_OK(gd_end_step(ctx));

    CHECK(gd_debug_heap_alloc_count() == heap_before, "no heap allocation in scoped hot path");
    gd_debug_set_heap_guard(false);
    CHECK_OK(gd_state_object_destroy(ctx, state));
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = tiny_config();
    gd_memory_stats stats;

    {
        gd_status st = gd_context_create(&cfg, &ctx);
        if (st == GD_ERR_UNSUPPORTED) {
            printf("test_memory: skipped (no supported GPU backend)\n");
            return 0;
        }
        CHECK_OK(st);
    }
    CHECK(ctx != NULL, "context created");
    test_arena_contract(ctx);
    test_scope_rings_and_state(ctx);
    CHECK_OK(gd_memory_stats_query(ctx, &stats));
    printf("test_memory: params=%zu state=%zu scratch_waits=%" PRIu64 " data_waits=%" PRIu64 " backend_waits=%" PRIu64 "\n",
           stats.params.watermark, stats.state.watermark,
           stats.scratch.waits, stats.data.waits, stats.backend_waits);
    gd_context_destroy(ctx);
    printf("test_memory: ok\n");
    return 0;
}
