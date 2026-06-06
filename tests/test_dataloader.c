#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_dataloader failed: %s (%s:%d)\n", (msg),    \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

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

static void put_le16(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xffU);
    p[1] = (uint8_t)((v >> 8U) & 0xffU);
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
        p[i] = (uint8_t)((v >> (8U * (uint32_t)i)) & UINT64_C(0xff));
    }
}

static void write_id_record(FILE *f, int32_t id)
{
    enum { RECORD_HEADER_NBYTES = 108, RECORD_NBYTES = 112 };
    uint8_t record[RECORD_NBYTES];
    uint8_t *entry;
    memset(record, 0, sizeof(record));
    memcpy(record, GD_GDDS_RECORD_MAGIC, strlen(GD_GDDS_RECORD_MAGIC));
    put_le16(&record[4], 1U);
    put_le32(&record[8], RECORD_HEADER_NBYTES);
    put_le64(&record[12], 4U);

    entry = &record[GD_GDDS_RECORD_HEADER_SIZE];
    put_le16(&entry[0], 0U);
    put_le16(&entry[2], 0U);
    put_le64(&entry[72], 0U);
    put_le64(&entry[80], 4U);

    put_le32(&record[RECORD_HEADER_NBYTES], (uint32_t)id);
    CHECK(fwrite(record, 1U, sizeof(record), f) == sizeof(record), "write record");
}

static void write_id_gdds(const char *path, int n_samples)
{
    enum {
        SCHEMA_OFFSET = 128,
        INDEX_OFFSET = 320,
        DATA_OFFSET = 512,
        RECORD_NBYTES = 112,
    };
    FILE *f;
    uint8_t header[GD_GDDS_HEADER_SIZE];
    uint8_t schema[GD_GDDS_FIELD_DESC_SIZE];
    int i;
    CHECK(n_samples > 0 && n_samples <= 128, "fixture sample count");
    memset(header, 0, sizeof(header));
    memcpy(header, GD_GDDS_MAGIC, strlen(GD_GDDS_MAGIC));
    put_le32(&header[8], GD_GDDS_VERSION);
    put_le32(&header[12], GD_GDDS_HEADER_SIZE);
    put_le32(&header[16], 1U);
    put_le64(&header[24], (uint64_t)n_samples);
    put_le64(&header[32], SCHEMA_OFFSET);
    put_le64(&header[40], INDEX_OFFSET);
    put_le64(&header[48], DATA_OFFSET);
    put_le64(&header[56], (uint64_t)n_samples * RECORD_NBYTES);

    memset(schema, 0, sizeof(schema));
    memcpy(&schema[0], "id", 3U);
    put_le32(&schema[64], (uint32_t)GD_DTYPE_I32);
    put_le32(&schema[68], 0U);

    f = fopen(path, "wb");
    CHECK(f != NULL, "open fixture");
    CHECK(fwrite(header, 1U, sizeof(header), f) == sizeof(header), "write header");
    CHECK(fwrite(schema, 1U, sizeof(schema), f) == sizeof(schema), "write schema");
    for (i = 0; i < INDEX_OFFSET - SCHEMA_OFFSET - (int)sizeof(schema); ++i) {
        CHECK(fputc(0, f) != EOF, "pad schema");
    }
    for (i = 0; i < n_samples; ++i) {
        uint8_t idx[GD_GDDS_INDEX_ENTRY_SIZE];
        memset(idx, 0, sizeof(idx));
        put_le64(&idx[0], (uint64_t)DATA_OFFSET + (uint64_t)i * RECORD_NBYTES);
        put_le64(&idx[8], RECORD_NBYTES);
        CHECK(fwrite(idx, 1U, sizeof(idx), f) == sizeof(idx), "write index");
    }
    for (i = 0; i < DATA_OFFSET - INDEX_OFFSET - n_samples * (int)GD_GDDS_INDEX_ENTRY_SIZE; ++i) {
        CHECK(fputc(0, f) != EOF, "pad index");
    }
    for (i = 0; i < n_samples; ++i) {
        write_id_record(f, i);
    }
    CHECK(fclose(f) == 0, "close fixture");
}

static void collect_ids(gd_context *ctx,
                        gd_dataloader *loader,
                        int steps,
                        int batch_size,
                        int32_t *out)
{
    int step;
    for (step = 0; step < steps; ++step) {
        gd_batch *batch = NULL;
        CHECK_OK(gd_dataloader_next(loader, &batch));
        CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, batch));
        CHECK_OK(gd_end_step(ctx));
        CHECK_OK(gd_tensor_read(ctx,
                                gd_batch_tensor(batch, "id"),
                                &out[step * batch_size],
                                (size_t)batch_size * sizeof(out[0])));
        CHECK_OK(gd_dataloader_release(loader, batch));
        CHECK_OK(gd_dataloader_prefetch(loader));
    }
}

static void expect_unique_prefix(const int32_t *ids, int count, int n_samples)
{
    int seen[128];
    int i;
    CHECK(n_samples <= (int)(sizeof(seen) / sizeof(seen[0])), "test helper capacity");
    memset(seen, 0, sizeof(seen));
    for (i = 0; i < count; ++i) {
        CHECK(ids[i] >= 0 && ids[i] < n_samples, "sample id in range");
        CHECK(seen[ids[i]] == 0, "sample id appears once per epoch prefix");
        seen[ids[i]] = 1;
    }
}

static void test_sequential_loader(gd_context *ctx, gd_dataset *dataset)
{
    enum { BATCH = 3, STEPS = 3, COUNT = BATCH * STEPS };
    gd_dataloader *loader = NULL;
    gd_dataloader_config cfg = gd_dataloader_config_default(BATCH);
    int32_t ids[COUNT];
    cfg.num_workers = 1;
    cfg.prefetch_factor = 2;
    CHECK_OK(gd_dataloader_create(ctx, dataset, NULL, &cfg, &loader));
    CHECK(gd_dataloader_slot_count(loader) == 2, "slot count");
    CHECK(gd_dataloader_steps_per_epoch(loader) == STEPS, "sequential steps per epoch drops tail");
    CHECK(gd_dataloader_samples_per_epoch(loader) == COUNT, "sequential samples per epoch drops tail");
    CHECK_OK(gd_dataloader_prefetch(loader));
    collect_ids(ctx, loader, STEPS, BATCH, ids);
    CHECK(ids[0] == 0 && ids[1] == 1 && ids[2] == 2 && ids[8] == 8,
          "sequential order");
    gd_dataloader_destroy(loader);
}

static void test_random_sampler_no_replacement(gd_context *ctx, gd_dataset *dataset)
{
    enum { N = 10, BATCH = 3, STEPS = 3, COUNT = BATCH * STEPS };
    gd_sampler *sampler = NULL;
    gd_sampler *sampler2 = NULL;
    gd_dataloader *loader = NULL;
    gd_dataloader *loader2 = NULL;
    gd_dataloader_config cfg = gd_dataloader_config_default(BATCH);
    int32_t epoch0[COUNT];
    int32_t epoch1[COUNT];
    int32_t epoch0_again[COUNT];

    CHECK_OK(gd_sampler_create_random(dataset, 12345U, &sampler));
    cfg.num_workers = 1;
    cfg.prefetch_factor = 2;
    CHECK_OK(gd_dataloader_create(ctx, dataset, sampler, &cfg, &loader));
    CHECK(gd_dataloader_steps_per_epoch(loader) == STEPS, "steps per epoch drops tail");
    CHECK(gd_dataloader_samples_per_epoch(loader) == COUNT, "samples per epoch drops tail");
    CHECK_OK(gd_dataloader_prefetch(loader));
    collect_ids(ctx, loader, STEPS, BATCH, epoch0);
    collect_ids(ctx, loader, STEPS, BATCH, epoch1);
    expect_unique_prefix(epoch0, COUNT, N);
    expect_unique_prefix(epoch1, COUNT, N);
    CHECK(memcmp(epoch0, epoch1, sizeof(epoch0)) != 0, "random sampler reshuffles next epoch");

    CHECK_OK(gd_sampler_create_random(dataset, 12345U, &sampler2));
    CHECK_OK(gd_dataloader_create(ctx, dataset, sampler2, &cfg, &loader2));
    CHECK_OK(gd_dataloader_prefetch(loader2));
    collect_ids(ctx, loader2, STEPS, BATCH, epoch0_again);
    CHECK(memcmp(epoch0, epoch0_again, sizeof(epoch0)) == 0, "random sampler deterministic seed");

    gd_dataloader_destroy(loader2);
    gd_sampler_destroy(sampler2);
    gd_dataloader_destroy(loader);
    gd_sampler_destroy(sampler);
}

int main(void)
{
    const char *path = "/tmp/gd_v2_dataloader_ids.gdds";
    gd_context *ctx = NULL;
    gd_dataset *dataset = NULL;
    gd_memory_config mem = loader_memory_config();
    gd_status st;

    write_id_gdds(path, 10);
    st = gd_context_create(&mem, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_dataloader: skipped (no supported GPU backend)\n");
        (void)remove(path);
        return 0;
    }
    CHECK_OK(st);
    CHECK_OK(gd_dataset_open_gdds_file(path, &dataset));
    CHECK(gd_dataset_num_samples(dataset) == 10U, "dataset samples");
    test_sequential_loader(ctx, dataset);
    test_random_sampler_no_replacement(ctx, dataset);
    gd_dataset_destroy(dataset);
    gd_context_destroy(ctx);
    (void)remove(path);
    printf("test_dataloader: ok\n");
    return 0;
}
