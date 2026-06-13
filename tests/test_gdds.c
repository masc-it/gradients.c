#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_gdds failed: %s (%s:%d)\n", (msg),          \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static gd_memory_config gdds_memory_config(void)
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

static void put_i64(uint8_t *p, int64_t v)
{
    uint64_t u;
    memcpy(&u, &v, sizeof(u));
    put_le64(p, u);
}

static void write_record(FILE *f, const uint16_t x[2], uint16_t target)
{
    enum { RECORD_NBYTES = 202, RECORD_HEADER_NBYTES = 196 };
    uint8_t record[RECORD_NBYTES];
    uint8_t *entry;
    memset(record, 0, sizeof(record));
    memcpy(record, GD_GDDS_RECORD_MAGIC, strlen(GD_GDDS_RECORD_MAGIC));
    put_le16(&record[4], 2U);
    put_le32(&record[8], RECORD_HEADER_NBYTES);
    put_le64(&record[12], 6U);

    entry = &record[GD_GDDS_RECORD_HEADER_SIZE];
    put_le16(&entry[0], 0U);
    put_le16(&entry[2], 1U);
    put_i64(&entry[8], 2);
    put_le64(&entry[72], 0U);
    put_le64(&entry[80], 4U);

    entry = &record[GD_GDDS_RECORD_HEADER_SIZE + GD_GDDS_RECORD_FIELD_DESC_SIZE];
    put_le16(&entry[0], 1U);
    put_le16(&entry[2], 1U);
    put_i64(&entry[8], 1);
    put_le64(&entry[72], 4U);
    put_le64(&entry[80], 2U);

    put_le16(&record[RECORD_HEADER_NBYTES], x[0]);
    put_le16(&record[RECORD_HEADER_NBYTES + 2], x[1]);
    put_le16(&record[RECORD_HEADER_NBYTES + 4], target);
    CHECK(fwrite(record, 1U, sizeof(record), f) == sizeof(record), "write record");
}

static const gd_dataset_field_spec XOR_U8_FIELDS[] = {
    {
        .name = "x",
        .dtype = GD_DTYPE_U8,
        .rank = 1,
        .shape = {2},
        .collate = GD_GDDS_COLLATE_STACK,
    },
    {
        .name = "target",
        .dtype = GD_DTYPE_U8,
        .rank = 1,
        .shape = {1},
        .collate = GD_GDDS_COLLATE_STACK,
    },
};

static uint8_t f16_one_to_u8(uint16_t bits)
{
    return bits == 0x3c00U ? (uint8_t)1U : (uint8_t)0U;
}

static gd_status xor_f16_to_u8_transform(const gd_sample *src,
                                         gd_sample *dst,
                                         void *user_data)
{
    const uint16_t *x_src;
    const uint16_t *y_src;
    uint8_t *x_dst;
    uint8_t *y_dst;
    (void)user_data;
    if (src == NULL || dst == NULL || gd_sample_field_count(src) != 2 ||
        gd_sample_field_count(dst) != 2 || gd_sample_field_dtype(src, 0) != GD_DTYPE_F16 ||
        gd_sample_field_dtype(src, 1) != GD_DTYPE_F16 ||
        gd_sample_field_dtype(dst, 0) != GD_DTYPE_U8 ||
        gd_sample_field_dtype(dst, 1) != GD_DTYPE_U8) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    x_src = (const uint16_t *)gd_sample_field_data(src, 0);
    y_src = (const uint16_t *)gd_sample_field_data(src, 1);
    x_dst = (uint8_t *)gd_sample_mutable_field_data(dst, 0);
    y_dst = (uint8_t *)gd_sample_mutable_field_data(dst, 1);
    if (x_src == NULL || y_src == NULL || x_dst == NULL || y_dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    x_dst[0] = f16_one_to_u8(x_src[0]);
    x_dst[1] = f16_one_to_u8(x_src[1]);
    y_dst[0] = f16_one_to_u8(y_src[0]);
    return GD_OK;
}

static void write_xor_gdds(const char *path)
{
    enum {
        N_SAMPLES = 4,
        SCHEMA_OFFSET = 128,
        INDEX_OFFSET = 448,
        DATA_OFFSET = 512,
        RECORD_NBYTES = 202,
    };
    static const uint16_t xs[N_SAMPLES][2] = {
        {0x0000U, 0x0000U},
        {0x0000U, 0x3c00U},
        {0x3c00U, 0x0000U},
        {0x3c00U, 0x3c00U},
    };
    static const uint16_t ys[N_SAMPLES] = {0x0000U, 0x3c00U, 0x3c00U, 0x0000U};
    FILE *f;
    uint8_t header[GD_GDDS_HEADER_SIZE];
    uint8_t schema[2U * GD_GDDS_FIELD_DESC_SIZE];
    int i;
    memset(header, 0, sizeof(header));
    memcpy(header, GD_GDDS_MAGIC, strlen(GD_GDDS_MAGIC));
    put_le32(&header[8], GD_GDDS_VERSION);
    put_le32(&header[12], GD_GDDS_HEADER_SIZE);
    put_le32(&header[16], 2U);
    put_le64(&header[24], N_SAMPLES);
    put_le64(&header[32], SCHEMA_OFFSET);
    put_le64(&header[40], INDEX_OFFSET);
    put_le64(&header[48], DATA_OFFSET);
    put_le64(&header[56], (uint64_t)N_SAMPLES * RECORD_NBYTES);

    memset(schema, 0, sizeof(schema));
    memcpy(&schema[0], "x", 2U);
    put_le32(&schema[64], (uint32_t)GD_DTYPE_F16);
    put_le32(&schema[68], 1U);
    put_i64(&schema[72], 2);
    memcpy(&schema[GD_GDDS_FIELD_DESC_SIZE], "target", 7U);
    put_le32(&schema[GD_GDDS_FIELD_DESC_SIZE + 64U], (uint32_t)GD_DTYPE_F16);
    put_le32(&schema[GD_GDDS_FIELD_DESC_SIZE + 68U], 1U);
    put_i64(&schema[GD_GDDS_FIELD_DESC_SIZE + 72U], 1);

    f = fopen(path, "wb");
    CHECK(f != NULL, "open fixture");
    CHECK(fwrite(header, 1U, sizeof(header), f) == sizeof(header), "write header");
    CHECK(fwrite(schema, 1U, sizeof(schema), f) == sizeof(schema), "write schema");
    for (i = 0; i < N_SAMPLES; ++i) {
        uint8_t idx[GD_GDDS_INDEX_ENTRY_SIZE];
        memset(idx, 0, sizeof(idx));
        put_le64(&idx[0], (uint64_t)DATA_OFFSET + (uint64_t)i * RECORD_NBYTES);
        put_le64(&idx[8], RECORD_NBYTES);
        CHECK(fwrite(idx, 1U, sizeof(idx), f) == sizeof(idx), "write index");
    }
    for (i = 0; i < N_SAMPLES; ++i) {
        write_record(f, xs[i], ys[i]);
    }
    CHECK(fclose(f) == 0, "close fixture");
}

int main(void)
{
    const char *path = "/tmp/gd_v2_gdds_xor-00000.gdds";
    gd_dataset *dataset = NULL;
    gd_context *ctx = NULL;
    gd_dataloader *loader = NULL;
    gd_dataset *transformed_dataset = NULL;
    gd_dataloader *transformed_loader = NULL;
    gd_batch *batch = NULL;
    gd_gdds_field_info info;
    gd_gdds_sample_field sample_field;
    uint16_t x[8];
    uint16_t target[4];
    uint8_t x_u8[8];
    uint8_t target_u8[4];
    gd_memory_config mem = gdds_memory_config();
    gd_status st;

    write_xor_gdds(path);
    CHECK_OK(gd_dataset_open_gdds_split("/tmp", "gd_v2_gdds_xor", &dataset));
    CHECK(gd_dataset_num_samples(dataset) == 4U, "sample count");
    CHECK(gd_gdds_dataset_field_count(dataset) == 2, "field count");
    CHECK(gd_gdds_dataset_field_index(dataset, "x") == 0, "x field index");
    CHECK_OK(gd_gdds_dataset_field_info(dataset, 1, &info));
    CHECK(strcmp(info.name, "target") == 0 && info.dtype == GD_DTYPE_F16 &&
              info.rank == 1 && info.shape[0] == 1,
          "target field info");
    CHECK_OK(gd_gdds_dataset_read_field(dataset, 2U, 0, &sample_field));
    CHECK(sample_field.nbytes == 4U && ((const uint16_t *)sample_field.data)[0] == 0x3c00U &&
              ((const uint16_t *)sample_field.data)[1] == 0x0000U,
          "sample field bytes");

    st = gd_context_create(&mem, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_gdds: skipped dataloader part (no supported GPU backend)\n");
        gd_dataset_destroy(dataset);
        (void)remove(path);
        return 0;
    }
    CHECK_OK(st);
    {
        const gd_dataloader_config cfg = {
            .batch_size = 4,
            .num_workers = 1,
            .prefetch_factor = 2,
        };
        CHECK_OK(gd_dataloader_create(ctx, dataset, NULL, &cfg, &loader));
    }
    CHECK_OK(gd_dataloader_prefetch(loader));
    CHECK_OK(gd_dataloader_next(loader, &batch));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, batch));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "x"), x, sizeof(x)));
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "target"), target, sizeof(target)));
    CHECK(x[0] == 0x0000U && x[1] == 0x0000U && x[2] == 0x0000U &&
              x[3] == 0x3c00U && x[4] == 0x3c00U && x[5] == 0x0000U &&
              x[6] == 0x3c00U && x[7] == 0x3c00U,
          "x batch contents");
    CHECK(target[0] == 0x0000U && target[1] == 0x3c00U &&
              target[2] == 0x3c00U && target[3] == 0x0000U,
          "target batch contents");
    CHECK_OK(gd_dataloader_release(loader, batch));
    batch = NULL;
    gd_dataloader_destroy(loader);
    loader = NULL;

    {
        const gd_dataset_transform_config transform_cfg = {
            .transform = xor_f16_to_u8_transform,
            .user_data = NULL,
            .output_fields = XOR_U8_FIELDS,
            .n_output_fields = (int)GD_ARRAY_LEN(XOR_U8_FIELDS),
        };
        const gd_dataloader_config cfg = {
            .batch_size = 4,
            .num_workers = 1,
            .prefetch_factor = 2,
        };
        CHECK_OK(gd_dataset_open_gdds_split_with_transform("/tmp",
                                                           "gd_v2_gdds_xor",
                                                           &transform_cfg,
                                                           &transformed_dataset));
        CHECK(gd_gdds_dataset_field_count(transformed_dataset) == 2, "transformed field count");
        CHECK_OK(gd_gdds_dataset_field_info(transformed_dataset, 0, &info));
        CHECK(strcmp(info.name, "x") == 0 && info.dtype == GD_DTYPE_U8 &&
                  info.rank == 1 && info.shape[0] == 2,
              "transformed x field info");
        CHECK(gd_gdds_dataset_read_field(transformed_dataset, 0U, 0, &sample_field) ==
                  GD_ERR_UNSUPPORTED,
              "transformed read_field unsupported");
        CHECK_OK(gd_dataloader_create(ctx, transformed_dataset, NULL, &cfg, &transformed_loader));
    }
    CHECK_OK(gd_dataloader_next(transformed_loader, &batch));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, batch));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "x"), x_u8, sizeof(x_u8)));
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "target"), target_u8, sizeof(target_u8)));
    CHECK(x_u8[0] == 0U && x_u8[1] == 0U && x_u8[2] == 0U && x_u8[3] == 1U &&
              x_u8[4] == 1U && x_u8[5] == 0U && x_u8[6] == 1U && x_u8[7] == 1U,
          "transformed x contents");
    CHECK(target_u8[0] == 0U && target_u8[1] == 1U && target_u8[2] == 1U &&
              target_u8[3] == 0U,
          "transformed target contents");
    CHECK_OK(gd_dataloader_release(transformed_loader, batch));

    gd_dataloader_destroy(transformed_loader);
    gd_dataloader_destroy(loader);
    gd_context_destroy(ctx);
    gd_dataset_destroy(transformed_dataset);
    gd_dataset_destroy(dataset);
    (void)remove(path);
    printf("test_gdds: ok\n");
    return 0;
}
