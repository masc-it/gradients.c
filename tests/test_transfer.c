#include <gradients/gradients.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_transfer failed: %s (%s:%d)\n", (msg),      \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)
#define CHECK_STATUS(expr, status) CHECK((expr) == (status), #expr)

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static gd_memory_config transfer_config(void)
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

static void test_param_transfer(gd_context *ctx)
{
    const int64_t shape[2] = {2, 4};
    uint16_t src[8] = {0x3c00U, 0x4000U, 0x4200U, 0x4400U,
                       0x4500U, 0x4600U, 0x4700U, 0x4800U};
    uint16_t dst[8];
    gd_tensor param;
    gd_tensor row;
    gd_tensor bad;

    memset(dst, 0, sizeof(dst));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 64U, &param));
    CHECK_OK(gd_upload(ctx, src, sizeof(src), &param));
    CHECK_OK(gd_download(ctx, &param, dst, sizeof(dst)));
    CHECK(memcmp(src, dst, sizeof(src)) == 0, "param upload/download roundtrip");

    CHECK_OK(gd_tensor_slice(ctx, &param, 0U, 1, 1, &row));
    memset(dst, 0, sizeof(dst));
    CHECK_OK(gd_download(ctx, &row, dst, 4U * sizeof(dst[0])));
    CHECK(memcmp(dst, &src[4], 4U * sizeof(dst[0])) == 0, "contiguous row view downloads from view offset");

    CHECK_STATUS(gd_download(ctx, &param, dst, sizeof(dst) - 1U), GD_ERR_INVALID_ARGUMENT);
    gd_context_clear_error(ctx);

    bad = param;
    bad.storage.offset = param.storage.offset + param.storage.nbytes + 1U;
    CHECK_STATUS(gd_upload(ctx, src, sizeof(src), &bad), GD_ERR_BAD_STATE);
    gd_context_clear_error(ctx);
}

static void test_f32_transfer(gd_context *ctx)
{
    const float src[4] = {0.0f, 1.0f, -2.0f, 3.25f};
    float dst[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    gd_tensor f16;
    gd_tensor f32;
    gd_tensor i32;
    int32_t i32_src[4] = {1, 2, 3, 4};
    uint32_t i;

    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, GD_SHAPE(4), src,
                                GD_ARRAY_LEN(src), true, &f16));
    CHECK(f16.requires_grad, "tensor_from_f32 applies requires_grad flag");
    CHECK_OK(gd_tensor_read_f32(ctx, &f16, dst, GD_ARRAY_LEN(dst)));
    for (i = 0U; i < GD_ARRAY_LEN(src); ++i) {
        CHECK(abs_f32(dst[i] - src[i]) <= 1.0e-3f, "f32 transfer converts through f16");
    }

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, GD_SHAPE(4), 64U, &f32));
    CHECK_OK(gd_tensor_write_f32(ctx, &f32, src, GD_ARRAY_LEN(src)));
    memset(dst, 0, sizeof(dst));
    CHECK_OK(gd_tensor_read_f32(ctx, &f32, dst, GD_ARRAY_LEN(dst)));
    CHECK(memcmp(src, dst, sizeof(src)) == 0, "f32 transfer preserves f32 values");

    CHECK_STATUS(gd_tensor_read_f32(ctx, &f32, dst, GD_ARRAY_LEN(dst) - 1U), GD_ERR_INVALID_ARGUMENT);
    gd_context_clear_error(ctx);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, GD_SHAPE(4), 64U, &i32));
    CHECK_OK(gd_tensor_write(ctx, &i32, i32_src, sizeof(i32_src)));
    CHECK_STATUS(gd_tensor_read_f32(ctx, &i32, dst, GD_ARRAY_LEN(dst)), GD_ERR_UNSUPPORTED);
    gd_context_clear_error(ctx);
}

static void test_scope_transfer(gd_context *ctx)
{
    const int64_t token_shape[2] = {2, 4};
    const int64_t hidden_shape[3] = {2, 6, 4};
    const int64_t ordered_shape[1] = {4};
    int32_t tokens_src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t tokens_dst[8];
    uint16_t hidden_src[48];
    uint16_t hidden_dst[48];
    gd_tensor tokens;
    gd_tensor hidden;
    gd_tensor suffix;
    gd_tensor ordered;
    uint64_t heap_before;
    int32_t scratch_slot;
    uint64_t scratch_generation;
    uint32_t i;
    float ordered_src[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float ordered_dst[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (i = 0U; i < 48U; ++i) {
        hidden_src[i] = (uint16_t)(0x3000U + i);
        hidden_dst[i] = 0U;
    }
    memset(tokens_dst, 0, sizeof(tokens_dst));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    heap_before = gd_debug_heap_alloc_count();
    gd_debug_set_heap_guard(true);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_DATA, GD_DTYPE_I32, gd_shape_make(2U, token_shape), 64U, &tokens));
    CHECK_OK(gd_tensor_write(ctx, &tokens, tokens_src, sizeof(tokens_src)));
    CHECK_OK(gd_tensor_read(ctx, &tokens, tokens_dst, sizeof(tokens_dst)));
    CHECK(memcmp(tokens_src, tokens_dst, sizeof(tokens_src)) == 0, "active data tensor write/read roundtrip");

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(3U, hidden_shape), 64U, &hidden));
    scratch_slot = hidden.storage.slot;
    scratch_generation = hidden.storage.generation;
    CHECK_OK(gd_tensor_write(ctx, &hidden, hidden_src, sizeof(hidden_src)));
    CHECK_OK(gd_tensor_read(ctx, &hidden, hidden_dst, sizeof(hidden_dst)));
    CHECK(memcmp(hidden_src, hidden_dst, sizeof(hidden_src)) == 0, "active scratch tensor write/read roundtrip");
    CHECK_OK(gd_tensor_slice(ctx, &hidden, 1U, 2, 3, &suffix));
    CHECK(!gd_tensor_is_contiguous(&suffix), "middle-dim suffix is non-contiguous");
    CHECK_STATUS(gd_tensor_read(ctx, &suffix, hidden_dst, 24U * sizeof(hidden_dst[0])), GD_ERR_UNSUPPORTED);
    gd_context_clear_error(ctx);
    CHECK_OK(gd_tensor_ones(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(1U, ordered_shape), 64U, &ordered));
    CHECK_OK(gd_tensor_write(ctx, &ordered, ordered_src, sizeof(ordered_src)));
    gd_debug_set_heap_guard(false);
    CHECK(gd_debug_heap_alloc_count() == heap_before, "transfer hot path does not heap allocate");
    CHECK_OK(gd_end(ctx));

    memset(hidden_dst, 0, sizeof(hidden_dst));
    CHECK_OK(gd_tensor_read(ctx, &hidden, hidden_dst, sizeof(hidden_dst)));
    CHECK(memcmp(hidden_src, hidden_dst, sizeof(hidden_src)) == 0, "post-scope scratch read waits relevant fence");
    CHECK_OK(gd_tensor_read(ctx, &ordered, ordered_dst, sizeof(ordered_dst)));
    CHECK(memcmp(ordered_src, ordered_dst, sizeof(ordered_src)) == 0,
          "blocking write after queued fill preserves program order");

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK(gd_debug_current_ring_slot(ctx, GD_ARENA_SCRATCH) != scratch_slot,
          "second scope advances scratch slot");
    CHECK_OK(gd_end(ctx));
    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK(gd_debug_current_ring_slot(ctx, GD_ARENA_SCRATCH) == scratch_slot,
          "third scope reuses original scratch slot");
    CHECK(gd_debug_ring_slot_generation(ctx, GD_ARENA_SCRATCH, (uint32_t)scratch_slot) > scratch_generation,
          "scratch generation bumped before stale transfer check");
    CHECK_STATUS(gd_tensor_read(ctx, &hidden, hidden_dst, sizeof(hidden_dst)), GD_ERR_BAD_STATE);
    gd_context_clear_error(ctx);
    CHECK_OK(gd_end(ctx));
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = transfer_config();
    gd_memory_stats stats;

    {
        gd_status st = gd_context_create(&cfg, &ctx);
        if (st == GD_ERR_UNSUPPORTED) {
            printf("test_transfer: skipped (no supported GPU backend)\n");
            return 0;
        }
        CHECK_OK(st);
    }
    CHECK(ctx != NULL, "context created");
    test_param_transfer(ctx);
    test_f32_transfer(ctx);
    test_scope_transfer(ctx);
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_memory_stats_query(ctx, &stats));
    printf("test_transfer: params=%zu scratch_watermark=%zu backend_waits=%" PRIu64 "\n",
           stats.params.watermark, stats.scratch.max_slot_watermark, stats.backend_waits);
    gd_context_destroy(ctx);
    printf("test_transfer: ok\n");
    return 0;
}
