#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_dataloader failed: %s (%s:%d)\n", (msg),  \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)
#define CHECK_STATUS(expr, status) CHECK((expr) == (status), #expr)

static gd_memory_config loader_memory_config(void)
{
    gd_memory_config cfg = gd_memory_config_default();
    cfg.params_bytes = 64U * 1024U;
    cfg.state_bytes = 64U * 1024U;
    cfg.scratch_slot_bytes = 64U * 1024U;
    cfg.data_slot_bytes = 64U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 3U;
    cfg.default_alignment = 256U;
    return cfg;
}

static void put_le32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xffU);
    p[1] = (uint8_t)((v >> 8U) & 0xffU);
    p[2] = (uint8_t)((v >> 16U) & 0xffU);
    p[3] = (uint8_t)((v >> 24U) & 0xffU);
}

static void put_le64(uint8_t *p, uint64_t v)
{
    int i;
    for (i = 0; i < 8; ++i) {
        p[i] = (uint8_t)((v >> (8U * (uint32_t)i)) & 0xffU);
    }
}

static void write_gdtok_u16(const char *path,
                            uint32_t block_len,
                            const uint16_t *tokens,
                            uint64_t n_tokens)
{
    FILE *f;
    uint8_t header[GD_GDTOK_HEADER_SIZE];
    uint64_t i;
    uint64_t n_samples = (n_tokens - 1U) / block_len;
    memset(header, 0, sizeof(header));
    memcpy(header, GD_GDTOK_MAGIC, strlen(GD_GDTOK_MAGIC));
    put_le32(&header[8], GD_GDTOK_VERSION);
    put_le32(&header[12], GD_GDTOK_HEADER_SIZE);
    put_le32(&header[16], (uint32_t)GD_GDTOK_DTYPE_U16);
    put_le32(&header[20], 512U);
    put_le32(&header[24], block_len);
    put_le64(&header[32], n_tokens);
    put_le64(&header[40], n_samples);
    put_le64(&header[48], UINT64_C(0x1234));
    put_le64(&header[56], GD_GDTOK_HEADER_SIZE);
    f = fopen(path, "wb");
    CHECK(f != NULL, "open fixture shard");
    CHECK(fwrite(header, 1U, sizeof(header), f) == sizeof(header), "write header");
    for (i = 0U; i < n_tokens; ++i) {
        uint8_t b[2];
        b[0] = (uint8_t)(tokens[i] & 0xffU);
        b[1] = (uint8_t)((tokens[i] >> 8U) & 0xffU);
        CHECK(fwrite(b, 1U, sizeof(b), f) == sizeof(b), "write payload");
    }
    CHECK(fclose(f) == 0, "close fixture shard");
}

static void make_lm_fields(gd_batch_field_desc *fields, int batch_size, int block_len)
{
    memset(fields, 0, 3U * sizeof(fields[0]));
    fields[0].name = "tokens";
    fields[0].dtype = GD_DTYPE_I32;
    fields[0].rank = 2;
    fields[0].sizes[0] = batch_size;
    fields[0].sizes[1] = block_len;
    fields[1].name = "targets";
    fields[1].dtype = GD_DTYPE_I32;
    fields[1].rank = 2;
    fields[1].sizes[0] = batch_size;
    fields[1].sizes[1] = block_len;
    fields[2].name = "positions";
    fields[2].dtype = GD_DTYPE_I32;
    fields[2].rank = 2;
    fields[2].sizes[0] = batch_size;
    fields[2].sizes[1] = block_len;
}

int main(void)
{
    const char *path = "/tmp/gd_v2_dataloader.gdtok";
    const uint16_t payload[13] = {
        1U, 2U, 3U, 4U, 5U,
        6U, 7U, 8U, 9U,
        10U, 11U, 12U, 13U,
    };
    const char *paths[1] = {path};
    gd_context *ctx = NULL;
    gd_dataset *dataset = NULL;
    gd_dataloader *loader = NULL;
    gd_batch *batch = NULL;
    gd_dataloader_config cfg;
    gd_batch_field_desc fields[3];
    gd_dataloader_metrics metrics;
    int32_t tokens[8];
    int32_t targets[8];
    int32_t positions[8];
    gd_memory_config mem = loader_memory_config();
    gd_status st;

    write_gdtok_u16(path, 4U, payload, 13U);
    st = gd_context_create(&mem, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_dataloader: skipped (no supported GPU backend)\n");
        (void)remove(path);
        return 0;
    }
    CHECK_OK(st);
    CHECK_OK(gd_dataset_open_gdtok(paths, 1, &dataset));
    make_lm_fields(fields, 2, 4);
    memset(&cfg, 0, sizeof(cfg));
    cfg.batch_size = 2;
    cfg.seed = 0U;
    cfg.sampler = GD_SAMPLER_SEQUENTIAL;
    cfg.expected_dataset_fingerprint = gd_dataset_fingerprint(dataset);
    cfg.num_workers = 1;
    cfg.prefetch_factor = 2;
    CHECK_OK(gd_dataloader_create(ctx, dataset, &cfg, fields, 3,
                                  gd_collate_gdtok_lm, NULL, &loader));
    CHECK(gd_dataloader_slot_count(loader) == 2, "slot count");
    CHECK_OK(gd_dataloader_prefetch(loader));
    CHECK_OK(gd_dataloader_next(loader, &batch));
    CHECK(gd_batch_get_state(batch) == GD_BATCH_IN_USE, "batch delivered");
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, batch));
    CHECK(gd_batch_get_state(batch) == GD_BATCH_IN_STEP, "batch in step");
    CHECK_OK(gd_end_step(ctx));
    CHECK(gd_batch_get_state(batch) == GD_BATCH_RETIRED, "batch retired");
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "tokens"), tokens, sizeof(tokens)));
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "targets"), targets, sizeof(targets)));
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "positions"), positions, sizeof(positions)));
    CHECK(tokens[0] == 1 && tokens[3] == 4 && tokens[4] == 5 && tokens[7] == 8,
          "token tensor contents");
    CHECK(targets[0] == 2 && targets[3] == 5 && targets[4] == 6 && targets[7] == 9,
          "target tensor contents");
    CHECK(positions[0] == 0 && positions[3] == 3 && positions[4] == 0 && positions[7] == 3,
          "position tensor contents");
    CHECK_OK(gd_dataloader_release(loader, batch));
    CHECK(gd_batch_get_state(batch) == GD_BATCH_FREE, "batch released");
    gd_dataloader_metrics_get(loader, &metrics);
    CHECK(metrics.batches_prepared >= 1U && metrics.batches_returned == 1U,
          "metrics updated");

    gd_dataloader_destroy(loader);
    gd_dataset_destroy(dataset);
    gd_context_destroy(ctx);
    (void)remove(path);
    printf("test_dataloader: ok\n");
    return 0;
}
