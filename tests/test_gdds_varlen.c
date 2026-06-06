#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_gdds_varlen failed: %s (%s:%d)\n", (msg),   \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static gd_memory_config varlen_memory_config(void)
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

static void put_i32_payload(uint8_t *p, const int32_t *values, int count)
{
    int i;
    for (i = 0; i < count; ++i) {
        put_le32(p + (size_t)i * sizeof(uint32_t), (uint32_t)values[i]);
    }
}

static void write_varlen_schema_field(uint8_t *schema,
                                      int field_id,
                                      const char *name,
                                      gd_dtype dtype,
                                      int64_t shape0,
                                      uint64_t flags,
                                      uint64_t pad_value_bits)
{
    uint8_t *base = schema + (size_t)field_id * GD_GDDS_FIELD_DESC_SIZE;
    memcpy(base, name, strlen(name) + 1U);
    put_le32(base + 64, (uint32_t)dtype);
    put_le32(base + 68, 1U);
    put_i64(base + 72, shape0);
    put_le64(base + 136, flags);
    put_le64(base + 144, pad_value_bits);
}

static void write_varlen_record(FILE *f,
                                const int32_t *tokens,
                                int token_len,
                                const int32_t *x,
                                int x_len)
{
    enum { RECORD_HEADER_NBYTES = 196 };
    const uint64_t token_nbytes = (uint64_t)token_len * sizeof(int32_t);
    const uint64_t x_nbytes = (uint64_t)x_len * sizeof(int32_t);
    const uint64_t record_nbytes = RECORD_HEADER_NBYTES + token_nbytes + x_nbytes;
    uint8_t header[RECORD_HEADER_NBYTES];
    uint8_t payload[64];
    uint8_t *entry;
    CHECK(record_nbytes <= RECORD_HEADER_NBYTES + sizeof(payload), "fixture payload capacity");
    memset(header, 0, sizeof(header));
    memset(payload, 0, sizeof(payload));
    memcpy(header, GD_GDDS_RECORD_MAGIC, strlen(GD_GDDS_RECORD_MAGIC));
    put_le16(&header[4], 2U);
    put_le32(&header[8], RECORD_HEADER_NBYTES);
    put_le64(&header[12], token_nbytes + x_nbytes);

    entry = &header[GD_GDDS_RECORD_HEADER_SIZE];
    put_le16(entry + 0, 0U);
    put_le16(entry + 2, 1U);
    put_i64(entry + 8, token_len);
    put_le64(entry + 72, 0U);
    put_le64(entry + 80, token_nbytes);

    entry = &header[GD_GDDS_RECORD_HEADER_SIZE + GD_GDDS_RECORD_FIELD_DESC_SIZE];
    put_le16(entry + 0, 3U);
    put_le16(entry + 2, 1U);
    put_i64(entry + 8, x_len);
    put_le64(entry + 72, token_nbytes);
    put_le64(entry + 80, x_nbytes);

    put_i32_payload(payload, tokens, token_len);
    put_i32_payload(payload + token_nbytes, x, x_len);
    CHECK(fwrite(header, 1U, sizeof(header), f) == sizeof(header), "write record header");
    CHECK(fwrite(payload, 1U, (size_t)(token_nbytes + x_nbytes), f) ==
              (size_t)(token_nbytes + x_nbytes),
          "write record payload");
}

static void write_varlen_gdds(const char *path)
{
    enum {
        N_FIELDS = 6,
        N_SAMPLES = 2,
        SCHEMA_OFFSET = 128,
        INDEX_OFFSET = 1088,
        DATA_OFFSET = 1152,
        RECORD_NBYTES = 216,
    };
    const int32_t tokens0[3] = {1, 2, 3};
    const int32_t tokens1[2] = {4, 5};
    const int32_t x0[2] = {5, 6};
    const int32_t x1[3] = {7, 8, 9};
    FILE *f;
    uint8_t header[GD_GDDS_HEADER_SIZE];
    uint8_t schema[N_FIELDS * GD_GDDS_FIELD_DESC_SIZE];
    int i;

    memset(header, 0, sizeof(header));
    memcpy(header, GD_GDDS_MAGIC, strlen(GD_GDDS_MAGIC));
    put_le32(&header[8], GD_GDDS_VERSION);
    put_le32(&header[12], GD_GDDS_HEADER_SIZE);
    put_le32(&header[16], N_FIELDS);
    put_le64(&header[24], N_SAMPLES);
    put_le64(&header[32], SCHEMA_OFFSET);
    put_le64(&header[40], INDEX_OFFSET);
    put_le64(&header[48], DATA_OFFSET);
    put_le64(&header[56], (uint64_t)N_SAMPLES * RECORD_NBYTES);

    memset(schema, 0, sizeof(schema));
    write_varlen_schema_field(schema, 0, "tokens", GD_DTYPE_I32, -1, UINT64_C(0x00010002), 0U);
    write_varlen_schema_field(schema, 1, "cu_seqlens", GD_DTYPE_I32, -1, UINT64_C(0x01000303), 0U);
    write_varlen_schema_field(schema, 2, "positions", GD_DTYPE_I32, -1, UINT64_C(0x01000403), 0U);
    write_varlen_schema_field(schema, 3, "x", GD_DTYPE_I32, -1, UINT64_C(0x00010001), UINT64_C(0xffffffff));
    write_varlen_schema_field(schema, 4, "x_len", GD_DTYPE_I32, -1, UINT64_C(0x04000103), 0U);
    write_varlen_schema_field(schema, 5, "x_mask", GD_DTYPE_U8, -1, UINT64_C(0x04000203), 0U);

    f = fopen(path, "wb");
    CHECK(f != NULL, "open fixture");
    CHECK(fwrite(header, 1U, sizeof(header), f) == sizeof(header), "write header");
    CHECK(fwrite(schema, 1U, sizeof(schema), f) == sizeof(schema), "write schema");
    for (i = 0; i < INDEX_OFFSET - SCHEMA_OFFSET - (int)sizeof(schema); ++i) {
        CHECK(fputc(0, f) != EOF, "pad schema");
    }
    for (i = 0; i < N_SAMPLES; ++i) {
        uint8_t idx[GD_GDDS_INDEX_ENTRY_SIZE];
        memset(idx, 0, sizeof(idx));
        put_le64(&idx[0], (uint64_t)DATA_OFFSET + (uint64_t)i * RECORD_NBYTES);
        put_le64(&idx[8], RECORD_NBYTES);
        CHECK(fwrite(idx, 1U, sizeof(idx), f) == sizeof(idx), "write index");
    }
    for (i = 0; i < DATA_OFFSET - INDEX_OFFSET - N_SAMPLES * (int)GD_GDDS_INDEX_ENTRY_SIZE; ++i) {
        CHECK(fputc(0, f) != EOF, "pad index");
    }
    write_varlen_record(f, tokens0, 3, x0, 2);
    write_varlen_record(f, tokens1, 2, x1, 3);
    CHECK(fclose(f) == 0, "close fixture");
}

int main(void)
{
    const char *path = "/tmp/gd_v2_gdds_varlen.gdds";
    gd_dataset *dataset = NULL;
    gd_context *ctx = NULL;
    gd_dataloader *loader = NULL;
    gd_batch *batch = NULL;
    gd_gdds_field_info info;
    gd_memory_config mem = varlen_memory_config();
    gd_status st;
    int32_t tokens[5];
    int32_t cu_seqlens[3];
    int32_t positions[5];
    int32_t x[6];
    int32_t x_len[2];
    uint8_t x_mask[6];

    write_varlen_gdds(path);
    CHECK_OK(gd_dataset_open_gdds_file(path, &dataset));
    CHECK(gd_gdds_dataset_field_count(dataset) == 6, "field count includes generated fields");
    CHECK_OK(gd_gdds_dataset_field_info(dataset, 0, &info));
    CHECK(info.collate == GD_GDDS_COLLATE_PACKED_SEQUENCE && info.ragged_dim == 0,
          "packed field metadata");
    CHECK_OK(gd_gdds_dataset_field_info(dataset, 1, &info));
    CHECK(info.collate == GD_GDDS_COLLATE_GENERATED &&
              info.generated == GD_GDDS_GENERATED_CU_SEQLENS && info.source_field == 0,
          "cu_seqlens metadata");

    st = gd_context_create(&mem, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_gdds_varlen: skipped dataloader part (no supported GPU backend)\n");
        gd_dataset_destroy(dataset);
        (void)remove(path);
        return 0;
    }
    CHECK_OK(st);
    {
        const gd_dataloader_config cfg = {
            .batch_size = 2,
            .num_workers = 1,
            .prefetch_factor = 2,
        };
        CHECK_OK(gd_dataloader_create(ctx, dataset, NULL, &cfg, &loader));
    }
    CHECK_OK(gd_dataloader_prefetch(loader));
    CHECK_OK(gd_dataloader_next(loader, &batch));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, batch));
    CHECK_OK(gd_end_step(ctx));
    CHECK(gd_batch_field_dim(batch, gd_batch_field_index(batch, "tokens"), 0) == 5,
          "packed token count");
    CHECK(gd_batch_field_dim(batch, gd_batch_field_index(batch, "x"), 0) == 2 &&
              gd_batch_field_dim(batch, gd_batch_field_index(batch, "x"), 1) == 3,
          "pad longest shape");
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "tokens"), tokens, sizeof(tokens)));
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "cu_seqlens"), cu_seqlens, sizeof(cu_seqlens)));
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "positions"), positions, sizeof(positions)));
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "x"), x, sizeof(x)));
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "x_len"), x_len, sizeof(x_len)));
    CHECK_OK(gd_tensor_read(ctx, gd_batch_tensor(batch, "x_mask"), x_mask, sizeof(x_mask)));
    CHECK(tokens[0] == 1 && tokens[1] == 2 && tokens[2] == 3 &&
              tokens[3] == 4 && tokens[4] == 5,
          "packed tokens");
    CHECK(cu_seqlens[0] == 0 && cu_seqlens[1] == 3 && cu_seqlens[2] == 5,
          "cu seqlens");
    CHECK(positions[0] == 0 && positions[1] == 1 && positions[2] == 2 &&
              positions[3] == 0 && positions[4] == 1,
          "positions");
    CHECK(x[0] == 5 && x[1] == 6 && x[2] == -1 &&
              x[3] == 7 && x[4] == 8 && x[5] == 9,
          "pad longest values");
    CHECK(x_len[0] == 2 && x_len[1] == 3, "lengths");
    CHECK(x_mask[0] == 1 && x_mask[1] == 1 && x_mask[2] == 0 &&
              x_mask[3] == 1 && x_mask[4] == 1 && x_mask[5] == 1,
          "mask");
    CHECK_OK(gd_dataloader_release(loader, batch));

    gd_dataloader_destroy(loader);
    gd_context_destroy(ctx);
    gd_dataset_destroy(dataset);
    (void)remove(path);
    printf("test_gdds_varlen: ok\n");
    return 0;
}
