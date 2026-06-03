#include "gradients/gradients.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define CHECK_OK(expr)                                                           \
    do {                                                                         \
        gd_status status_ = (expr);                                              \
        if (status_ != GD_OK) {                                                  \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());        \
            return 1;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_TRUE(expr)                                                         \
    do {                                                                         \
        if (!(expr)) {                                                           \
            fprintf(stderr, "%s failed at %s:%d\n", #expr, __FILE__, __LINE__); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

static void put_le16(unsigned char *p, uint16_t v)
{
    p[0] = (unsigned char)(v & 0xffU);
    p[1] = (unsigned char)((uint32_t)v >> 8U);
}

static void put_le32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v & 0xffU);
    p[1] = (unsigned char)((v >> 8U) & 0xffU);
    p[2] = (unsigned char)((v >> 16U) & 0xffU);
    p[3] = (unsigned char)((v >> 24U) & 0xffU);
}

static void put_le64(unsigned char *p, uint64_t v)
{
    put_le32(p, (uint32_t)(v & UINT64_C(0xffffffff)));
    put_le32(p + 4, (uint32_t)(v >> 32U));
}

static int write_all(FILE *f, const void *data, size_t n)
{
    return fwrite(data, 1U, n, f) == n ? 0 : 1;
}

static int write_header(FILE *f)
{
    unsigned char h[GD_GDVLM_HEADER_SIZE];
    memset(h, 0, sizeof(h));
    memcpy(h, "GDVLMv1\0", 8U);
    put_le32(h + 8, GD_GDVLM_VERSION);
    put_le32(h + 12, GD_GDVLM_HEADER_SIZE);
    put_le32(h + 16, 4U);
    put_le32(h + 20, 4U);
    put_le32(h + 24, 1U);
    put_le32(h + 28, 2U);
    put_le32(h + 32, 4U);
    put_le32(h + 36, 4U);
    put_le32(h + 40, GD_GDVLM_PATCH_DTYPE_F16);
    put_le32(h + 44, GD_GDVLM_TOKEN_DTYPE_U16);
    put_le32(h + 48, 128U);
    put_le32(h + 52, 0U);
    put_le32(h + 56, 1U);
    put_le32(h + 60, GD_GDVLM_HEADER_SIZE);
    put_le64(h + 64, 2U);
    put_le64(h + 72, UINT64_C(0x123456789abcdef0));
    return write_all(f, h, sizeof(h));
}

static int write_record(FILE *f,
                        uint32_t label,
                        const uint16_t *tokens,
                        uint32_t n_tokens,
                        uint16_t patch_base)
{
    uint32_t i;
    unsigned char b4[4];
    unsigned char b2[2];
    put_le32(b4, label);
    if (write_all(f, b4, sizeof(b4)) != 0) {
        return 1;
    }
    put_le32(b4, n_tokens);
    if (write_all(f, b4, sizeof(b4)) != 0) {
        return 1;
    }
    for (i = 0U; i < n_tokens; ++i) {
        put_le16(b2, tokens[i]);
        if (write_all(f, b2, sizeof(b2)) != 0) {
            return 1;
        }
    }
    for (i = 0U; i < 16U; ++i) {
        put_le16(b2, (uint16_t)(patch_base + (uint16_t)i));
        if (write_all(f, b2, sizeof(b2)) != 0) {
            return 1;
        }
    }
    return 0;
}

static int write_idx_entry(FILE *f,
                           uint32_t sample_idx,
                           uint64_t body_offset,
                           uint32_t record_nbytes,
                           uint32_t token_len,
                           uint16_t label,
                           uint32_t raw_pos)
{
    unsigned char e[GD_GDVLM_IDX_ENTRY_SIZE];
    memset(e, 0, sizeof(e));
    put_le32(e, 0U);
    put_le32(e + 4, sample_idx);
    put_le64(e + 8, body_offset);
    put_le32(e + 16, record_nbytes);
    put_le32(e + 20, token_len);
    put_le16(e + 24, label);
    put_le16(e + 26, 0U);
    put_le32(e + 28, raw_pos);
    return write_all(f, e, sizeof(e));
}

static int write_fixture(const char *shard_path, const char *idx_path)
{
    FILE *f;
    const uint16_t tok0[3] = {10U, 11U, 12U};
    const uint16_t tok1[2] = {20U, 21U};
    unsigned char idx_header[GD_GDVLM_IDX_HEADER_SIZE];
    uint32_t rec0 = 8U + 3U * 2U + 16U * 2U;
    uint32_t rec1 = 8U + 2U * 2U + 16U * 2U;

    f = fopen(shard_path, "wb");
    if (f == NULL) {
        return 1;
    }
    if (write_header(f) != 0 || write_record(f, 7U, tok0, 3U, 100U) != 0 ||
        write_record(f, 8U, tok1, 2U, 200U) != 0 || fclose(f) != 0) {
        return 1;
    }
    f = fopen(idx_path, "wb");
    if (f == NULL) {
        return 1;
    }
    memset(idx_header, 0, sizeof(idx_header));
    memcpy(idx_header, "GDVLMIDX", 8U);
    put_le32(idx_header + 8, GD_GDVLM_IDX_VERSION);
    put_le32(idx_header + 12, GD_GDVLM_IDX_ENTRY_SIZE);
    put_le64(idx_header + 16, 2U);
    if (write_all(f, idx_header, sizeof(idx_header)) != 0 ||
        write_idx_entry(f, 0U, 0U, rec0, 3U, 7U, 101U) != 0 ||
        write_idx_entry(f, 1U, rec0, rec1, 2U, 8U, 102U) != 0 ||
        fclose(f) != 0) {
        return 1;
    }
    return 0;
}

static int test_gdvlm_read_and_collate(void)
{
    const char *dir = "/tmp/gdvlm_fixture";
    const char *shard_path = "/tmp/gdvlm_fixture/train-00001-of-00001.gdvlm";
    const char *idx_path = "/tmp/gdvlm_fixture/train.idx";
    gd_gdvlm_header header;
    gd_dataset *ds = NULL;
    gd_gdvlm_sample_info info;
    int32_t text[4];
    uint16_t patches_one[16];
    gd_context *ctx = NULL;
    gd_dataloader *dl = NULL;
    gd_dataloader_config dl_cfg;
    gd_vlm_collate_config vlm_cfg;
    gd_batch_field_desc fields[GD_VLM_BATCH_FIELD_COUNT];
    int n_fields = 0;
    gd_batch *batch = NULL;
    int32_t tokens[14];
    int32_t targets[14];
    int32_t positions[14];
    float loss_mask[14];
    int32_t prefix_len[2];
    int32_t text_len[2];
    int32_t label_id[2];
    uint16_t patches[32];

    (void)mkdir(dir, 0777);
    CHECK_TRUE(write_fixture(shard_path, idx_path) == 0);
    CHECK_OK(gd_gdvlm_read_header(shard_path, &header));
    CHECK_TRUE(header.num_patches == 4U);
    CHECK_TRUE(header.patch_dim == 4U);
    CHECK_TRUE(header.tokenizer_hash == UINT64_C(0x123456789abcdef0));

    CHECK_OK(gd_dataset_open_gdvlm_split(dir, "train", &ds));
    CHECK_TRUE(gd_dataset_num_samples(ds) == 2U);
    CHECK_OK(gd_gdvlm_dataset_sample_info(ds, 0U, &info));
    CHECK_TRUE(info.token_len == 3U && info.label_id == 7);
    CHECK_OK(gd_gdvlm_dataset_read_sample(ds, 0U, &info, text, 4,
                                          patches_one, sizeof(patches_one)));
    CHECK_TRUE(info.tokens_copied == 3U);
    CHECK_TRUE(text[0] == 10 && text[1] == 11 && text[2] == 12);
    CHECK_TRUE(patches_one[0] == 100U && patches_one[15] == 115U);

    memset(&vlm_cfg, 0, sizeof(vlm_cfg));
    vlm_cfg.pad_token_id = 0;
    vlm_cfg.target_pad_id = -100;
    vlm_cfg.target_mode = GD_VLM_TEXT_TARGET_SHIFT_RIGHT;
    CHECK_OK(gd_vlm_init_batch_fields(ds, &vlm_cfg, 2, fields,
                                      GD_VLM_BATCH_FIELD_COUNT, &n_fields));
    CHECK_TRUE(n_fields == GD_VLM_BATCH_FIELD_COUNT);

    memset(&dl_cfg, 0, sizeof(dl_cfg));
    dl_cfg.batch_size = 2;
    dl_cfg.device = (gd_device){GD_DEVICE_CPU, 0};
    dl_cfg.sampler = GD_SAMPLER_SEQUENTIAL;
    dl_cfg.expected_dataset_fingerprint = gd_dataset_fingerprint(ds);
    CHECK_OK(gd_context_create(&ctx));
    CHECK_OK(gd_dataloader_create(ctx, ds, &dl_cfg, fields, n_fields,
                                  gd_collate_gdvlm, &vlm_cfg, &dl));
    CHECK_OK(gd_dataloader_next(dl, &batch));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gd_batch_tensor(batch, "tokens"),
                                   tokens, sizeof(tokens)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gd_batch_tensor(batch, "targets"),
                                   targets, sizeof(targets)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gd_batch_tensor(batch, "positions"),
                                   positions, sizeof(positions)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gd_batch_tensor(batch, "loss_mask"),
                                   loss_mask, sizeof(loss_mask)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gd_batch_tensor(batch, "prefix_len"),
                                   prefix_len, sizeof(prefix_len)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gd_batch_tensor(batch, "text_len"),
                                   text_len, sizeof(text_len)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gd_batch_tensor(batch, "label_id"),
                                   label_id, sizeof(label_id)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gd_batch_tensor(batch, "patches"),
                                   patches, sizeof(patches)));

    CHECK_TRUE(prefix_len[0] == 4 && prefix_len[1] == 4);
    CHECK_TRUE(text_len[0] == 3 && text_len[1] == 2);
    CHECK_TRUE(label_id[0] == 7 && label_id[1] == 8);
    CHECK_TRUE(tokens[4] == 0 && tokens[5] == 10 && tokens[6] == 11);
    CHECK_TRUE(targets[4] == 10 && targets[5] == 11 && targets[6] == 12);
    CHECK_TRUE(loss_mask[0] == 0.0F && loss_mask[4] == 1.0F && loss_mask[6] == 1.0F);
    CHECK_TRUE(tokens[11] == 0 && tokens[12] == 20 && tokens[13] == 0);
    CHECK_TRUE(targets[11] == 20 && targets[12] == 21 && targets[13] == -100);
    CHECK_TRUE(loss_mask[11] == 1.0F && loss_mask[12] == 1.0F && loss_mask[13] == 0.0F);
    CHECK_TRUE(positions[0] == 0 && positions[6] == 6 && positions[13] == 6);
    CHECK_TRUE(patches[0] == 100U && patches[15] == 115U && patches[16] == 200U);
    CHECK_OK(gd_dataloader_release(dl, batch));

    gd_dataloader_destroy(dl);
    gd_context_destroy(ctx);
    gd_dataset_destroy(ds);
    (void)remove(shard_path);
    (void)remove(idx_path);
    (void)rmdir(dir);
    return 0;
}

int main(void)
{
    CHECK_TRUE(test_gdvlm_read_and_collate() == 0);
    return 0;
}
