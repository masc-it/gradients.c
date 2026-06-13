#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_dataset failed: %s (%s:%d)\n", (msg),       \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

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

static void write_scalar_gdds(const char *path)
{
    enum {
        SCHEMA_OFFSET = 128,
        INDEX_OFFSET = 320,
        DATA_OFFSET = 384,
        RECORD_HEADER_NBYTES = 108,
        RECORD_NBYTES = 112,
    };
    FILE *f;
    uint8_t header[GD_GDDS_HEADER_SIZE];
    uint8_t schema[GD_GDDS_FIELD_DESC_SIZE];
    uint8_t index[GD_GDDS_INDEX_ENTRY_SIZE];
    uint8_t record[RECORD_NBYTES];
    uint8_t *entry;
    int i;

    memset(header, 0, sizeof(header));
    memcpy(header, GD_GDDS_MAGIC, strlen(GD_GDDS_MAGIC));
    put_le32(&header[8], GD_GDDS_VERSION);
    put_le32(&header[12], GD_GDDS_HEADER_SIZE);
    put_le32(&header[16], 1U);
    put_le64(&header[24], 1U);
    put_le64(&header[32], SCHEMA_OFFSET);
    put_le64(&header[40], INDEX_OFFSET);
    put_le64(&header[48], DATA_OFFSET);
    put_le64(&header[56], RECORD_NBYTES);

    memset(schema, 0, sizeof(schema));
    memcpy(schema, "value", 6U);
    put_le32(&schema[64], (uint32_t)GD_DTYPE_I32);
    put_le32(&schema[68], 0U);

    memset(index, 0, sizeof(index));
    put_le64(&index[0], DATA_OFFSET);
    put_le64(&index[8], RECORD_NBYTES);

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
    put_le32(&record[RECORD_HEADER_NBYTES], 42U);

    f = fopen(path, "wb");
    CHECK(f != NULL, "open fixture");
    CHECK(fwrite(header, 1U, sizeof(header), f) == sizeof(header), "write header");
    CHECK(fwrite(schema, 1U, sizeof(schema), f) == sizeof(schema), "write schema");
    for (i = 0; i < INDEX_OFFSET - SCHEMA_OFFSET - (int)sizeof(schema); ++i) {
        CHECK(fputc(0, f) != EOF, "pad schema");
    }
    CHECK(fwrite(index, 1U, sizeof(index), f) == sizeof(index), "write index");
    for (i = 0; i < DATA_OFFSET - INDEX_OFFSET - (int)sizeof(index); ++i) {
        CHECK(fputc(0, f) != EOF, "pad index");
    }
    CHECK(fwrite(record, 1U, sizeof(record), f) == sizeof(record), "write record");
    CHECK(fclose(f) == 0, "close fixture");
}

int main(void)
{
    const char *path = "/tmp/gd_v2_dataset_scalar.gdds";
    gd_dataset *dataset = NULL;
    gd_gdds_field_info info;
    gd_gdds_sample_field sample_field;

    write_scalar_gdds(path);
    CHECK_OK(gd_dataset_open_gdds_file(path, &dataset));
    CHECK(strcmp(gd_dataset_name(dataset), "gdds") == 0, "dataset name");
    CHECK(gd_dataset_num_samples(dataset) == 1U, "sample count");
    CHECK(gd_gdds_dataset_field_count(dataset) == 1, "field count");
    CHECK(gd_gdds_dataset_field_index(dataset, "value") == 0, "field index");
    CHECK_OK(gd_gdds_dataset_field_info(dataset, 0, &info));
    CHECK(strcmp(info.name, "value") == 0 && info.dtype == GD_DTYPE_I32 &&
              info.rank == 0 && info.collate == GD_GDDS_COLLATE_STACK,
          "field info");
    CHECK_OK(gd_gdds_dataset_read_field(dataset, 0U, 0, &sample_field));
    CHECK(sample_field.nbytes == 4U && *(const int32_t *)sample_field.data == 42,
          "sample data");

    gd_dataset_destroy(dataset);
    (void)remove(path);
    printf("test_dataset: ok\n");
    return 0;
}
