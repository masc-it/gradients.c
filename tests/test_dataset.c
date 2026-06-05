#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_dataset failed: %s (%s:%d)\n", (msg),     \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

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

int main(void)
{
    const char *path = "/tmp/gd_v2_dataset.gdtok";
    const uint16_t payload[9] = {10U, 11U, 12U, 13U, 14U, 15U, 16U, 17U, 18U};
    const char *paths[1] = {path};
    gd_gdtok_header header;
    gd_dataset *dataset = NULL;
    int32_t tokens[4];
    int32_t targets[4];
    uint64_t block_len = 0U;
    uint64_t vocab_size = 0U;

    write_gdtok_u16(path, 4U, payload, 9U);
    CHECK_OK(gd_gdtok_read_header(path, &header));
    CHECK(header.version == GD_GDTOK_VERSION, "version");
    CHECK(header.header_size == GD_GDTOK_HEADER_SIZE, "header size");
    CHECK(header.dtype == GD_GDTOK_DTYPE_U16, "dtype");
    CHECK(header.block_len == 4U, "block len");
    CHECK(header.n_samples == 2U, "n samples");

    CHECK_OK(gd_dataset_open_gdtok(paths, 1, &dataset));
    CHECK(gd_dataset_num_samples(dataset) == 2U, "dataset samples");
    CHECK(gd_dataset_fingerprint(dataset) != 0U, "fingerprint");
    CHECK_OK(gd_dataset_get_u64(dataset, "block_len", &block_len));
    CHECK_OK(gd_dataset_get_u64(dataset, "vocab_size", &vocab_size));
    CHECK(block_len == 4U && vocab_size == 512U, "metadata");
    CHECK_OK(gd_gdtok_dataset_read_lm_sample(dataset, 1U, tokens, targets));
    CHECK(tokens[0] == 14 && tokens[3] == 17, "sample tokens");
    CHECK(targets[0] == 15 && targets[3] == 18, "sample targets");

    gd_dataset_destroy(dataset);
    (void)remove(path);
    printf("test_dataset: ok\n");
    return 0;
}
