#include "gradients/gradients.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK_OK(expr)                                                            \
    do {                                                                          \
        gd_status status_ = (expr);                                               \
        if (status_ != GD_OK) {                                                   \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());          \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_STATUS(expr, expected)                                               \
    do {                                                                          \
        gd_status status_ = (expr);                                               \
        if (status_ != (expected)) {                                               \
            fprintf(stderr, "%s got %s expected %s; last_error=%s\n",             \
                    #expr,                                                        \
                    gd_status_name(status_),                                      \
                    gd_status_name(expected),                                     \
                    gd_last_error());                                             \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_TRUE(expr)                                                          \
    do {                                                                          \
        if (!(expr)) {                                                            \
            fprintf(stderr, "%s failed\n", #expr);                              \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static int floats_equal(const float *a, const float *b, size_t n)
{
    size_t i = 0U;

    for (i = 0U; i < n; ++i) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}

static int test_storage(gd_context *ctx)
{
    gd_storage_desc desc = {{GD_DEVICE_CPU, 0}, GD_MEM_HOST, 32U, 16U};
    gd_storage_desc bad_desc = {{GD_DEVICE_CPU, 0}, GD_MEM_HOST, 0U, 0U};
    gd_storage *storage = NULL;
    void *data = NULL;
    uint8_t src[4] = {1U, 2U, 3U, 4U};
    uint8_t dst[4] = {0U, 0U, 0U, 0U};
    gd_device cpu = {GD_DEVICE_CPU, 0};

    CHECK_STATUS(gd_storage_create(NULL, &desc, &storage), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_storage_create(ctx, NULL, &storage), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_storage_create(ctx, &bad_desc, &storage), GD_ERR_INVALID_ARGUMENT);

    bad_desc = (gd_storage_desc){{GD_DEVICE_CPU, 1}, GD_MEM_HOST, 32U, 0U};
    CHECK_STATUS(gd_storage_create(ctx, &bad_desc, &storage), GD_ERR_UNSUPPORTED);
    bad_desc = (gd_storage_desc){{GD_DEVICE_CPU, 0}, GD_MEM_DEVICE, 32U, 0U};
    CHECK_STATUS(gd_storage_create(ctx, &bad_desc, &storage), GD_ERR_UNSUPPORTED);
    bad_desc = (gd_storage_desc){{GD_DEVICE_CPU, 0}, GD_MEM_HOST, 32U, 24U};
    CHECK_STATUS(gd_storage_create(ctx, &bad_desc, &storage), GD_ERR_INVALID_ARGUMENT);

    CHECK_OK(gd_storage_create(ctx, &desc, &storage));
    CHECK_TRUE(storage != NULL);
    CHECK_TRUE(gd_storage_nbytes(storage) == 32U);
    CHECK_TRUE(gd_device_equal(gd_storage_device(storage), cpu));
    CHECK_OK(gd_storage_data_cpu(storage, &data));
    CHECK_TRUE(data != NULL);
    CHECK_TRUE(((uintptr_t)data % 16U) == 0U);

    CHECK_OK(gd_storage_copy_from_cpu(ctx, storage, 4U, src, sizeof(src)));
    CHECK_OK(gd_storage_copy_to_cpu(ctx, storage, 4U, dst, sizeof(dst)));
    CHECK_TRUE(memcmp(src, dst, sizeof(src)) == 0);
    CHECK_STATUS(gd_storage_copy_from_cpu(ctx, storage, 31U, src, 2U),
                 GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_storage_copy_to_cpu(ctx, storage, 31U, dst, 2U),
                 GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_storage_data_cpu(NULL, &data), GD_ERR_INVALID_ARGUMENT);

    CHECK_OK(gd_storage_retain(storage));
    gd_storage_release(storage);
    gd_storage_release(storage);
    return 0;
}

static int test_tensor_create_copy(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t sizes[2] = {2, 3};
    gd_tensor_desc desc;
    gd_tensor *tensor = NULL;
    float src[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    float dst[6] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    size_t nbytes = 0U;
    size_t alignment = 0U;
    gd_tensor *grad = NULL;

    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 2, sizes, &desc));
    CHECK_TRUE(desc.layout == GD_LAYOUT_CONTIGUOUS);
    CHECK_TRUE(desc.strides[0] == 3);
    CHECK_TRUE(desc.strides[1] == 1);
    CHECK_OK(gd_tensor_desc_nbytes(&desc, &nbytes, &alignment));
    CHECK_TRUE(nbytes == sizeof(src));
    CHECK_TRUE(alignment == sizeof(float));

    CHECK_OK(gd_tensor_empty(ctx, &desc, &tensor));
    CHECK_TRUE(tensor != NULL);
    CHECK_TRUE(gd_tensor_ndim(tensor) == 2);
    CHECK_TRUE(gd_tensor_size(tensor, 0) == 2);
    CHECK_TRUE(gd_tensor_size(tensor, 1) == 3);
    CHECK_TRUE(gd_tensor_stride(tensor, 0) == 3);
    CHECK_TRUE(gd_tensor_stride(tensor, 1) == 1);
    CHECK_TRUE(gd_tensor_dtype(tensor) == GD_DTYPE_F32);
    CHECK_TRUE(gd_tensor_layout(tensor) == GD_LAYOUT_CONTIGUOUS);
    CHECK_TRUE(gd_device_equal(gd_tensor_device(tensor), cpu));
    CHECK_TRUE(gd_tensor_storage(tensor) != NULL);
    CHECK_TRUE(gd_tensor_quant(tensor) == NULL);

    CHECK_OK(gd_tensor_copy_from_cpu(ctx, tensor, src, sizeof(src)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, tensor, dst, sizeof(dst)));
    CHECK_TRUE(floats_equal(src, dst, 6U));
    CHECK_STATUS(gd_tensor_copy_to_cpu(ctx, tensor, dst, sizeof(dst) + 4U),
                 GD_ERR_INVALID_ARGUMENT);

    CHECK_OK(gd_tensor_set_requires_grad(tensor, true));
    CHECK_TRUE(gd_tensor_requires_grad(tensor));
    CHECK_OK(gd_tensor_grad(tensor, &grad));
    CHECK_TRUE(grad == NULL);
    CHECK_OK(gd_tensor_set_requires_grad(tensor, false));
    CHECK_TRUE(!gd_tensor_requires_grad(tensor));

    gd_tensor_release(tensor);
    return 0;
}

static int test_tensor_views(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t sizes[2] = {2, 3};
    int64_t reshape_sizes[2] = {3, 2};
    gd_tensor_desc desc;
    gd_tensor *base = NULL;
    gd_tensor *reshaped = NULL;
    gd_tensor *transposed = NULL;
    gd_tensor *slice = NULL;
    gd_tensor *same = NULL;
    gd_tensor *contig = NULL;
    float src[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    float slice_out[3] = {0.0F, 0.0F, 0.0F};
    float expected_slice[3] = {4.0F, 5.0F, 6.0F};
    float contig_out[6] = {0};
    float expected_contig[6] = {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F};
    int ci = 0;

    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 2, sizes, &desc));
    CHECK_OK(gd_tensor_empty(ctx, &desc, &base));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, base, src, sizeof(src)));

    CHECK_OK(gd_tensor_reshape(base, 2, reshape_sizes, &reshaped));
    CHECK_TRUE(gd_tensor_size(reshaped, 0) == 3);
    CHECK_TRUE(gd_tensor_size(reshaped, 1) == 2);
    CHECK_TRUE(gd_tensor_stride(reshaped, 0) == 2);
    CHECK_TRUE(gd_tensor_stride(reshaped, 1) == 1);
    CHECK_TRUE(gd_tensor_storage(reshaped) == gd_tensor_storage(base));

    CHECK_OK(gd_tensor_transpose(base, 0, 1, &transposed));
    CHECK_TRUE(gd_tensor_size(transposed, 0) == 3);
    CHECK_TRUE(gd_tensor_size(transposed, 1) == 2);
    CHECK_TRUE(gd_tensor_stride(transposed, 0) == 1);
    CHECK_TRUE(gd_tensor_stride(transposed, 1) == 3);
    CHECK_TRUE(gd_tensor_layout(transposed) == GD_LAYOUT_STRIDED);
    CHECK_STATUS(gd_tensor_copy_to_cpu(ctx, transposed, slice_out, sizeof(slice_out)),
                 GD_ERR_UNSUPPORTED);

    /* materialize the strided transpose into a contiguous copy */
    CHECK_OK(gd_tensor_contiguous(ctx, transposed, &contig));
    CHECK_TRUE(contig != transposed);
    CHECK_TRUE(gd_tensor_layout(contig) == GD_LAYOUT_CONTIGUOUS);
    CHECK_TRUE(gd_tensor_size(contig, 0) == 3 && gd_tensor_size(contig, 1) == 2);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, contig, contig_out, sizeof(contig_out)));
    for (ci = 0; ci < 6; ++ci) {
        CHECK_TRUE(contig_out[ci] == expected_contig[ci]);
    }
    gd_tensor_release(contig);

    CHECK_OK(gd_tensor_slice(base, 0, 1, 1, &slice));
    CHECK_TRUE(gd_tensor_size(slice, 0) == 1);
    CHECK_TRUE(gd_tensor_size(slice, 1) == 3);
    CHECK_TRUE(gd_tensor_layout(slice) == GD_LAYOUT_CONTIGUOUS);
    gd_tensor_release(base);
    base = NULL;

    CHECK_OK(gd_tensor_copy_to_cpu(ctx, slice, slice_out, sizeof(slice_out)));
    CHECK_TRUE(floats_equal(slice_out, expected_slice, 3U));
    CHECK_OK(gd_tensor_contiguous(ctx, slice, &same));
    CHECK_TRUE(same == slice);

    gd_tensor_release(same);
    gd_tensor_release(slice);
    gd_tensor_release(transposed);
    gd_tensor_release(reshaped);
    return 0;
}

static int test_bad_tensor_descs(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t sizes[2] = {2, 3};
    int64_t bad_sizes[2] = {2, 0};
    gd_tensor_desc desc;
    gd_tensor_desc bad_desc;
    gd_storage_desc storage_desc = {cpu, GD_MEM_HOST, 4U, 0U};
    gd_storage *storage = NULL;
    gd_tensor *tensor = NULL;
    gd_quant_desc *fake_quant = (gd_quant_desc *)(uintptr_t)1U;
    size_t nbytes = 0U;

    CHECK_STATUS(gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 2, NULL, &desc),
                 GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 9, sizes, &desc),
                 GD_ERR_SHAPE);
    CHECK_STATUS(gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 2, bad_sizes, &desc),
                 GD_ERR_SHAPE);

    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 2, sizes, &desc));
    bad_desc = desc;
    bad_desc.dtype = GD_DTYPE_INVALID;
    CHECK_STATUS(gd_tensor_desc_nbytes(&bad_desc, &nbytes, NULL), GD_ERR_DTYPE);

    bad_desc = desc;
    bad_desc.layout = GD_LAYOUT_CHANNELS_LAST;
    CHECK_STATUS(gd_tensor_desc_nbytes(&bad_desc, &nbytes, NULL), GD_ERR_UNSUPPORTED);

    bad_desc = desc;
    bad_desc.layout = GD_LAYOUT_PACKED_QUANT;
    bad_desc.dtype = GD_DTYPE_QUANTIZED;
    bad_desc.quant = fake_quant;
    CHECK_STATUS(gd_tensor_desc_nbytes(&bad_desc, &nbytes, NULL), GD_ERR_UNSUPPORTED);

    bad_desc = desc;
    bad_desc.quant = fake_quant;
    CHECK_STATUS(gd_tensor_desc_nbytes(&bad_desc, &nbytes, NULL), GD_ERR_DTYPE);

    CHECK_OK(gd_storage_create(ctx, &storage_desc, &storage));
    CHECK_STATUS(gd_tensor_from_storage(ctx, storage, &desc, &tensor),
                 GD_ERR_INVALID_ARGUMENT);
    gd_storage_release(storage);

    CHECK_OK(gd_tensor_empty(ctx, &desc, &tensor));
    CHECK_STATUS(gd_tensor_set_requires_grad(tensor, true), GD_OK);
    gd_tensor_release(tensor);

    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_I32, cpu, 2, sizes, &desc));
    CHECK_OK(gd_tensor_empty(ctx, &desc, &tensor));
    CHECK_STATUS(gd_tensor_set_requires_grad(tensor, true), GD_ERR_DTYPE);
    gd_tensor_release(tensor);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;

    CHECK_OK(gd_context_create(&ctx));
    if (test_storage(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    if (test_tensor_create_copy(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    if (test_tensor_views(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    if (test_bad_tensor_descs(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    return 0;
}
