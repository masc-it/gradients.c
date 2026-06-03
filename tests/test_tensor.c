#include <gradients/gradients.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_tensor failed: %s (%s:%d)\n", (msg),        \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)
#define CHECK_STATUS(expr, status) CHECK((expr) == (status), #expr)

static gd_memory_config tensor_config(void)
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

static void test_params_tensor(gd_context *ctx)
{
    const int64_t shape[2] = {4, 8};
    const int64_t bad_shape[1] = {-1};
    gd_tensor param;
    gd_tensor view;
    gd_tensor bad;
    gd_memory_stats before;
    gd_memory_stats after;
    int64_t numel;
    size_t nbytes;

    CHECK(gd_dtype_size(GD_DTYPE_F16) == 2U, "f16 dtype size");
    CHECK(gd_dtype_size(GD_DTYPE_F32) == 4U, "f32 dtype size");
    CHECK_OK(gd_memory_stats_query(ctx, &before));
    CHECK_STATUS(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 1U, bad_shape, 64U, &bad),
                 GD_ERR_INVALID_ARGUMENT);
    CHECK_OK(gd_memory_stats_query(ctx, &after));
    CHECK(before.params.offset == after.params.offset, "bad tensor shape does not allocate");
    gd_context_clear_error(ctx);

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, shape, 64U, &param));
    CHECK(param.dtype == GD_DTYPE_F16 && param.device == GD_DEVICE_GPU, "param dtype/device");
    CHECK(param.storage.arena == GD_ARENA_PARAMS && param.storage.slot == -1, "param arena metadata");
    CHECK(param.storage.offset % 64U == 0U, "param storage alignment");
    CHECK(param.shape[0] == 4 && param.shape[1] == 8, "param shape");
    CHECK(param.strides[0] == 8 && param.strides[1] == 1, "param compact strides");
    CHECK(gd_tensor_is_contiguous(&param), "param contiguous");
    CHECK_OK(gd_tensor_numel(&param, &numel));
    CHECK(numel == 32, "param numel");
    CHECK_OK(gd_tensor_logical_nbytes(&param, &nbytes));
    CHECK(nbytes == 64U, "param logical nbytes");
    CHECK(gd_tensor_storage_offset(&param) == param.storage.offset, "param view offset zero");
    CHECK_OK(gd_tensor_validate(ctx, &param));

    CHECK_OK(gd_tensor_slice(ctx, &param, 0U, 1, 2, &view));
    CHECK(view.is_view, "slice is view");
    CHECK(view.storage.buffer == param.storage.buffer, "slice shares storage buffer");
    CHECK(view.storage.offset == param.storage.offset, "slice shares allocation offset");
    CHECK(view.view_offset == 16U, "slice view offset");
    CHECK(view.shape[0] == 2 && view.shape[1] == 8, "slice shape");
    CHECK(gd_tensor_storage_offset(&view) == param.storage.offset + 16U, "slice storage offset");
    CHECK_OK(gd_tensor_validate(ctx, &view));

    CHECK_STATUS(gd_tensor_slice(ctx, &param, 1U, 7, 4, &bad), GD_ERR_INVALID_ARGUMENT);
    CHECK(bad.storage.nbytes == 0U, "bad slice does not publish descriptor");
    gd_context_clear_error(ctx);

    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_STATUS(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, shape, 64U, &bad),
                 GD_ERR_FROZEN);
    gd_context_clear_error(ctx);
}

static void test_scope_tensor_views(gd_context *ctx)
{
    const int64_t token_shape[2] = {2, 16};
    const int64_t hidden_shape[3] = {2, 14, 32};
    gd_tensor tokens;
    gd_tensor hidden;
    gd_tensor suffix;
    gd_tensor compact;
    gd_memory_stats stats;
    uint64_t heap_before;
    int32_t hidden_slot;
    uint64_t hidden_generation;

    CHECK_STATUS(gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, 3U, hidden_shape, 64U, &hidden),
                 GD_ERR_BAD_STATE);
    gd_context_clear_error(ctx);

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    heap_before = gd_debug_heap_alloc_count();
    gd_debug_set_heap_guard(true);

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_DATA, GD_DTYPE_I32, 2U, token_shape, 64U, &tokens));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, 3U, hidden_shape, 64U, &hidden));
    hidden_slot = hidden.storage.slot;
    hidden_generation = hidden.storage.generation;
    CHECK(tokens.storage.arena == GD_ARENA_DATA, "tokens live in data arena");
    CHECK(hidden.storage.arena == GD_ARENA_SCRATCH, "hidden lives in scratch arena");
    CHECK(hidden.strides[0] == 448 && hidden.strides[1] == 32 && hidden.strides[2] == 1,
          "hidden compact strides");
    CHECK_OK(gd_tensor_slice(ctx, &hidden, 1U, 6, 8, &suffix));
    CHECK(suffix.view_offset == 384U, "suffix view offset");
    CHECK(!gd_tensor_is_contiguous(&suffix), "suffix slice is non-contiguous");
    CHECK(suffix.storage.buffer == hidden.storage.buffer, "suffix shares hidden storage");
    CHECK_OK(gd_tensor_contiguous(ctx, GD_ARENA_SCRATCH, &suffix, 64U, &compact));
    CHECK(gd_tensor_is_contiguous(&compact), "explicit contiguous output compact");
    CHECK(!compact.is_view, "contiguous output owns allocation span");
    CHECK(compact.storage.offset != hidden.storage.offset, "contiguous output storage distinct");
    CHECK(compact.shape[0] == 2 && compact.shape[1] == 8 && compact.shape[2] == 32,
          "contiguous output shape");
    CHECK(compact.strides[0] == 256 && compact.strides[1] == 32 && compact.strides[2] == 1,
          "contiguous output strides");
    CHECK_OK(gd_tensor_validate(ctx, &compact));

    gd_debug_set_heap_guard(false);
    CHECK(gd_debug_heap_alloc_count() == heap_before, "tensor hot path does not heap allocate");
    CHECK_OK(gd_end(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK(gd_debug_current_ring_slot(ctx, GD_ARENA_SCRATCH) != hidden_slot,
          "second scope uses next scratch slot");
    CHECK_OK(gd_end(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK(gd_debug_current_ring_slot(ctx, GD_ARENA_SCRATCH) == hidden_slot,
          "third scope reuses original scratch slot");
    CHECK(gd_debug_ring_slot_generation(ctx, GD_ARENA_SCRATCH, (uint32_t)hidden_slot) > hidden_generation,
          "scratch generation bumped on reuse");
    CHECK_STATUS(gd_tensor_validate(ctx, &hidden), GD_ERR_BAD_STATE);
    gd_context_clear_error(ctx);
    CHECK_OK(gd_end(ctx));

    CHECK_OK(gd_memory_stats_query(ctx, &stats));
    CHECK(stats.scratch.max_slot_watermark >= compact.storage.offset + compact.storage.nbytes,
          "scratch watermark includes compact tensor");
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = tensor_config();
    gd_memory_stats stats;

    {
        gd_status st = gd_context_create(&cfg, &ctx);
        if (st == GD_ERR_UNSUPPORTED) {
            printf("test_tensor: skipped (no supported GPU backend)\n");
            return 0;
        }
        CHECK_OK(st);
    }
    CHECK(ctx != NULL, "context created");
    test_params_tensor(ctx);
    test_scope_tensor_views(ctx);
    CHECK_OK(gd_memory_stats_query(ctx, &stats));
    printf("test_tensor: params=%zu scratch_watermark=%zu backend_waits=%" PRIu64 "\n",
           stats.params.watermark, stats.scratch.max_slot_watermark, stats.backend_waits);
    gd_context_destroy(ctx);
    printf("test_tensor: ok\n");
    return 0;
}
