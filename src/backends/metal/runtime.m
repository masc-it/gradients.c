#import "metal_internal.h"

#include <stdlib.h>

@implementation GDMetalState
@end

@implementation GDMPSGemmPlan
@end

gd_metal_sdpa_bwd_layout _gd_metal_sdpa_bwd_scratch_layout(int B, int Hq, int Hkv,
                                                           int Tq, int Tk, int Dh, int S)
{
    gd_metal_sdpa_bwd_layout o;
    int64_t off = 0;
    o.stats_off = off;
    off += (int64_t)B * Hq * Tq * 3;
    o.stats_part_off = off;
    off += (int64_t)B * Hq * Tq * S * 3;
    o.dq_part_off = off;
    off += (int64_t)B * Hq * Tq * S * 2 * Dh;
    o.dkv_part_off = off;
    off += (int64_t)B * Hkv * Tk * S * 2 * Dh;
    o.total = off;
    return o;
}

/* Split-K key partitioning for the forward SDPA, derived purely from Tk so the
 * compile-time scratch size and the encode-time dispatch agree. Returns 1 (no
 * split) for short sequences, keeping them on the single-pass tiled kernel. */
static int metal_env_positive_int(const char *name, int fallback, int min_value, int max_value)
{
    const char *v = getenv(name);
    char *end = NULL;
    long n = fallback;

    if (v == NULL || v[0] == '\0') {
        return fallback;
    }
    n = strtol(v, &end, 10);
    if (end == v || n < (long)min_value) {
        return fallback;
    }
    if (n > (long)max_value) {
        return max_value;
    }
    return (int)n;
}

int _gd_metal_sdpa_num_splits(int Tk)
{
    int split_min = metal_env_positive_int("GD_METAL_SDPA_SPLIT_MIN",
                                           GD_METAL_SDPA_SPLIT_MIN,
                                           GD_METAL_SDPA_BK,
                                           1 << 20);
    int split_max = metal_env_positive_int("GD_METAL_SDPA_SPLIT_MAX",
                                           GD_METAL_SDPA_SPLIT_MAX,
                                           1,
                                           1024);
    int s = (Tk + split_min - 1) / split_min;

    if (s < 1) {
        s = 1;
    }
    if (s > split_max) {
        s = split_max;
    }
    return s;
}

/* Keys per split, rounded up to a GD_METAL_SDPA_BK multiple so each partition
 * starts on a key-tile boundary. */
int _gd_metal_sdpa_split_len(int Tk, int n_splits)
{
    int len = (Tk + n_splits - 1) / n_splits;
    len = ((len + GD_METAL_SDPA_BK - 1) / GD_METAL_SDPA_BK) * GD_METAL_SDPA_BK;
    if (len < GD_METAL_SDPA_BK) {
        len = GD_METAL_SDPA_BK;
    }
    return len;
}

gd_metal_lmce_scratch_layout _gd_metal_lmce_fwd_scratch_layout_for(int rows, int chunk)
{
    gd_metal_lmce_scratch_layout L;
    size_t off = 0U;
    L.logits_off = off;
    off += (size_t)rows * (size_t)chunk * sizeof(float);
    L.target_logit_off = off;
    off += (size_t)rows * sizeof(float);
    L.losses_off = off;
    off += (size_t)rows * sizeof(float);
    L.total = off;
    return L;
}

size_t _gd_metal_lmce_bwd_scratch_bytes_for(int rows, int chunk)
{
    return (size_t)rows * (size_t)chunk * sizeof(float);
}

gd_status _gd_metal_dtype_code(gd_dtype dtype, int *out)
{
    switch (dtype) {
    case GD_DTYPE_F32:
        *out = GD_METAL_DT_F32;
        return GD_OK;
    case GD_DTYPE_I32:
        *out = GD_METAL_DT_I32;
        return GD_OK;
    case GD_DTYPE_F16:
        *out = GD_METAL_DT_F16;
        return GD_OK;
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "metal cast supports F32/F16/I32 only in v1");
    }
}


GDMetalState *_gd_metal_state(_gd_backend *self)
{
    return (__bridge GDMetalState *)self->impl;
}

id<MTLComputePipelineState> _gd_metal_pipeline_for(GDMetalState *st, _gd_op_kind op)
{
    return st.pipelines[@((int)op)];
}

id<MTLComputePipelineState> _gd_metal_pipeline_named(GDMetalState *st, const char *name)
{
    return st.pipelinesByName[[NSString stringWithUTF8String:name]];
}

int64_t _gd_metal_desc_numel(const gd_tensor_desc *desc)
{
    int64_t n = 1;
    int i = 0;
    for (i = 0; i < desc->ndim; ++i) {
        n *= desc->sizes[i];
    }
    return n;
}

bool _gd_metal_desc_same_shape(const gd_tensor_desc *a, const gd_tensor_desc *b)
{
    int i = 0;
    if (a->ndim != b->ndim) {
        return false;
    }
    for (i = 0; i < a->ndim; ++i) {
        if (a->sizes[i] != b->sizes[i]) {
            return false;
        }
    }
    return true;
}

void _gd_metal_build_ew_params(gd_metal_ew_params *p,
                            const gd_tensor_desc *out_desc,
                            const gd_tensor_desc *a_desc,
                            const gd_tensor_desc *b_desc)
{
    int i = 0;

    memset(p, 0, sizeof(*p));
    p->ndim = out_desc->ndim;
    p->numel = (int)_gd_metal_desc_numel(out_desc);
    p->a_ndim = a_desc->ndim;
    p->b_ndim = b_desc->ndim;
    p->dtype = GD_METAL_DT_F32;
    (void)_gd_metal_dtype_code(out_desc->dtype, &p->dtype);
    for (i = 0; i < out_desc->ndim; ++i) {
        p->out_sizes[i] = (int)out_desc->sizes[i];
    }
    for (i = 0; i < a_desc->ndim; ++i) {
        p->a_sizes[i] = (int)a_desc->sizes[i];
    }
    p->same_shape = (a_desc->ndim == out_desc->ndim && b_desc->ndim == out_desc->ndim) ? 1 : 0;
    for (i = 0; i < a_desc->ndim; ++i) {
        if (a_desc->sizes[i] != out_desc->sizes[i]) {
            p->same_shape = 0;
        }
    }
    for (i = 0; i < b_desc->ndim; ++i) {
        p->b_sizes[i] = (int)b_desc->sizes[i];
        if (b_desc->sizes[i] != out_desc->sizes[i]) {
            p->same_shape = 0;
        }
    }
}

id<MTLBuffer> _gd_metal_value_buffer(_gd_executable *exe, int value_id)
{
    return (__bridge id<MTLBuffer>)_gd_storage_handle(exe->values[value_id].storage);
}

void _gd_metal_dispatch_1d(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> pso,
                        NSUInteger numel)
{
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;

    if (tg > numel) {
        tg = numel;
    }
    if (tg == 0) {
        tg = 1;
    }
    [enc dispatchThreads:MTLSizeMake(numel, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

/* Tiled GEMM dispatch: one threadgroup per GD_METAL_GEMM_TILE-square output
 * block, with the K-tiling loop inside the kernel. Uses fixed-size threadgroups
 * (required for the threadgroup-memory tiles) and rounds the grid up. */
void _gd_metal_dispatch_gemm_tiles(id<MTLComputeCommandEncoder> enc,
                                NSUInteger cols,
                                NSUInteger rows,
                                NSUInteger batch)
{
    NSUInteger gx = (cols + GD_METAL_GEMM_BN - 1) / GD_METAL_GEMM_BN;
    NSUInteger gy = (rows + GD_METAL_GEMM_BM - 1) / GD_METAL_GEMM_BM;
    NSUInteger tx = GD_METAL_GEMM_BN / GD_METAL_GEMM_TN;
    NSUInteger ty = GD_METAL_GEMM_BM / GD_METAL_GEMM_TM;

    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, batch)
        threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
}

/* Dispatches `groups` threadgroups of GD_METAL_RMS_TG threads (one threadgroup
 * per reduced row, or per channel tile). */
void _gd_metal_dispatch_reduce_groups(id<MTLComputeCommandEncoder> enc, NSUInteger groups)
{
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(GD_METAL_RMS_TG, 1, 1)];
}

gd_status _gd_metal_encode_mps_gemm(id<MTLCommandBuffer> cmd,
                                 __strong id<MTLComputeCommandEncoder> *enc,
                                 _gd_executable *exe,
                                 GDMPSGemmPlan *plan)
{
    id<MTLBuffer> left_buffer = nil;
    id<MTLBuffer> right_buffer = nil;
    id<MTLBuffer> result_buffer = nil;
    MPSMatrix *left = nil;
    MPSMatrix *right = nil;
    MPSMatrix *result = nil;

    if (cmd == nil || enc == NULL || *enc == nil || exe == NULL || plan == nil ||
        plan.kernel == nil || plan.leftDescriptor == nil || plan.rightDescriptor == nil ||
        plan.resultDescriptor == nil || plan.leftValue < 0 || plan.rightValue < 0 ||
        plan.resultValue < 0 || plan.leftValue >= exe->n_values ||
        plan.rightValue >= exe->n_values || plan.resultValue >= exe->n_values) {
        return _gd_error(GD_ERR_BACKEND, "invalid MPS GEMM encode");
    }
    left_buffer = _gd_metal_value_buffer(exe, plan.leftValue);
    right_buffer = _gd_metal_value_buffer(exe, plan.rightValue);
    result_buffer = _gd_metal_value_buffer(exe, plan.resultValue);
    if (left_buffer == nil || right_buffer == nil || result_buffer == nil) {
        return _gd_error(GD_ERR_BACKEND, "invalid MPS GEMM buffers");
    }
    left = [[MPSMatrix alloc] initWithBuffer:left_buffer offset:0 descriptor:plan.leftDescriptor];
    right = [[MPSMatrix alloc] initWithBuffer:right_buffer offset:0 descriptor:plan.rightDescriptor];
    result = [[MPSMatrix alloc] initWithBuffer:result_buffer offset:0 descriptor:plan.resultDescriptor];
    if (left == nil || right == nil || result == nil) {
        return _gd_error(GD_ERR_BACKEND, "failed to create MPS GEMM matrices");
    }
    plan.left = left;
    plan.right = right;
    plan.result = result;
    [*enc endEncoding];
    *enc = nil;
    [plan.kernel encodeToCommandBuffer:cmd
                            leftMatrix:plan.left
                           rightMatrix:plan.right
                          resultMatrix:plan.result];
    *enc = [cmd computeCommandEncoder];
    if (*enc == nil) {
        return _gd_error(GD_ERR_BACKEND, "failed to resume compute encoder after MPS GEMM");
    }
    return GD_OK;
}

MPSMatrix *_gd_metal_mps_matrix_typed(id<MTLBuffer> buffer,
                                      NSUInteger offset,
                                      NSUInteger rows,
                                      NSUInteger cols,
                                      NSUInteger row_bytes,
                                      MPSDataType data_type)
{
    MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                    columns:cols
                                                                   rowBytes:row_bytes
                                                                   dataType:data_type];
    return [[MPSMatrix alloc] initWithBuffer:buffer offset:offset descriptor:d];
}

MPSMatrix *_gd_metal_mps_matrix(id<MTLBuffer> buffer,
                             NSUInteger offset,
                             NSUInteger rows,
                             NSUInteger cols,
                             NSUInteger row_bytes)
{
    return _gd_metal_mps_matrix_typed(buffer, offset, rows, cols, row_bytes,
                                      MPSDataTypeFloat32);
}

gd_status _gd_metal_encode_mps_mm(id<MTLCommandBuffer> cmd,
                               __strong id<MTLComputeCommandEncoder> *enc,
                               id<MTLDevice> device,
                               MPSMatrix *left,
                               MPSMatrix *right,
                               MPSMatrix *result,
                               BOOL trans_left,
                               BOOL trans_right,
                               NSUInteger rows,
                               NSUInteger cols,
                               NSUInteger inner,
                               double beta)
{
    if (cmd == nil || enc == NULL || *enc == nil || device == nil || left == nil ||
        right == nil || result == nil || rows == 0U || cols == 0U || inner == 0U) {
        return _gd_error(GD_ERR_BACKEND, "invalid dynamic MPS GEMM encode");
    }
    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                    transposeLeft:trans_left
                                                                   transposeRight:trans_right
                                                                       resultRows:rows
                                                                    resultColumns:cols
                                                                  interiorColumns:inner
                                                                            alpha:1.0
                                                                             beta:beta];
    if (mm == nil) {
        return _gd_error(GD_ERR_BACKEND, "failed to create dynamic MPS GEMM");
    }
    [*enc endEncoding];
    *enc = nil;
    [mm encodeToCommandBuffer:cmd leftMatrix:left rightMatrix:right resultMatrix:result];
    *enc = [cmd computeCommandEncoder];
    if (*enc == nil) {
        return _gd_error(GD_ERR_BACKEND, "failed to resume compute encoder after dynamic MPS GEMM");
    }
    return GD_OK;
}
