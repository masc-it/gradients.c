#include "metal_backend_internal.h"
#include "primitives/memory/metal_memory_types.h"
#include "primitives/random/metal_random_types.h"

#import <Foundation/Foundation.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GD_METAL_MAX_THREADS_PER_GROUP 256U

static id<MTLDevice> gd_metal_device(gd_backend *backend)
{
    return (__bridge id<MTLDevice>)backend->device;
}

static id<MTLCommandQueue> gd_metal_queue(gd_backend *backend)
{
    return (__bridge id<MTLCommandQueue>)backend->queue;
}

static id<MTLComputePipelineState> gd_metal_fill_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->fill_pso;
}

static id<MTLComputePipelineState> gd_metal_rand_uniform_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->rand_uniform_pso;
}

static id<MTLComputePipelineState> gd_metal_accumulate_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->accumulate_pso;
}

static id<MTLComputePipelineState> gd_metal_scale_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->scale_pso;
}

static id<MTLCommandBuffer> gd_metal_active_command_buffer(gd_backend *backend)
{
    return (__bridge id<MTLCommandBuffer>)backend->active_command_buffer;
}

static id<MTLBuffer> gd_metal_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static id<MTLCommandBuffer> gd_metal_command_buffer(gd_backend_fence *fence)
{
    return (__bridge id<MTLCommandBuffer>)fence->handle;
}

static NSURL *gd_metal_metallib_url(void)
{
    const char *env = getenv("GRADIENTS_METALLIB");
    NSURL *url;
    if (env != NULL && env[0] != '\0') {
        NSString *path = [NSString stringWithUTF8String:env];
        return path != nil ? [NSURL fileURLWithPath:path] : nil;
    }
    url = [[NSBundle mainBundle] URLForResource:@"gradients" withExtension:@"metallib"];
    if (url != nil) {
        return url;
    }
    return [NSURL fileURLWithPath:@"build/gradients.metallib"];
}

static gd_status gd_metal_make_pipeline(gd_backend *backend,
                                        id<MTLLibrary> library,
                                        const char *kernel_name,
                                        void **out_pso)
{
    NSError *error = nil;
    NSString *name;
    id<MTLFunction> function;
    id<MTLComputePipelineState> pso;
    if (backend == NULL || library == nil || kernel_name == NULL || out_pso == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    name = [NSString stringWithUTF8String:kernel_name];
    if (name == nil) {
        return GD_ERR_INTERNAL;
    }
    function = [library newFunctionWithName:name];
    if (function == nil) {
        return GD_ERR_INTERNAL;
    }
    pso = [gd_metal_device(backend) newComputePipelineStateWithFunction:function error:&error];
    if (pso == nil) {
        return GD_ERR_INTERNAL;
    }
    *out_pso = (void *)CFBridgingRetain(pso);
    return GD_OK;
}

static gd_status gd_metal_make_pipelines(gd_backend *backend)
{
    NSError *error = nil;
    NSURL *url;
    id<MTLLibrary> library;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    url = gd_metal_metallib_url();
    if (url == nil) {
        return GD_ERR_INTERNAL;
    }
    library = [gd_metal_device(backend) newLibraryWithURL:url error:&error];
    if (library == nil) {
        return GD_ERR_UNSUPPORTED;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_fill_kernel", &backend->fill_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rand_uniform_kernel", &backend->rand_uniform_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_accumulate_kernel", &backend->accumulate_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_scale_kernel", &backend->scale_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_amp_unscale_kernel", &backend->amp_unscale_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_binary_reduce_broadcast_kernel", &backend->binary_reduce_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_binary_reduce_broadcast_suffix_kernel", &backend->binary_reduce_suffix_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_mul_backward_direct_kernel", &backend->mul_backward_direct_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_mul_reduce_suffix_kernel", &backend->mul_reduce_suffix_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_mul_reduce_suffix_small_kernel", &backend->mul_reduce_suffix_small_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_reduce_contiguous_f16_to_f16_kernel", &backend->reduce_contiguous_f16_to_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_reduce_contiguous_f16_to_f32_kernel", &backend->reduce_contiguous_f16_to_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_reduce_contiguous_f32_to_f32_kernel", &backend->reduce_contiguous_f32_to_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_reduce_contiguous_f32_to_f16_kernel", &backend->reduce_contiguous_f32_to_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_reduce_axis_f16_kernel", &backend->reduce_axis_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_reduce_axis_f32_kernel", &backend->reduce_axis_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_reduce_axis_last_f16_kernel", &backend->reduce_axis_last_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_reduce_axis_last_f32_kernel", &backend->reduce_axis_last_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_broadcast_axis_f16_kernel", &backend->broadcast_axis_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_broadcast_axis_f32_kernel", &backend->broadcast_axis_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_broadcast_axis_last_f16_kernel", &backend->broadcast_axis_last_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_broadcast_axis_last_f32_kernel", &backend->broadcast_axis_last_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_broadcast_to_f16_kernel", &backend->broadcast_to_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_broadcast_to_f32_kernel", &backend->broadcast_to_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_broadcast_scalar_f16_kernel", &backend->broadcast_scalar_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_broadcast_scalar_f32_kernel", &backend->broadcast_scalar_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_broadcast_scalar_f32_to_f16_kernel", &backend->broadcast_scalar_f32_to_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_cross_entropy_loss_f16_kernel", &backend->cross_entropy_loss_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_cross_entropy_loss_stats_f16_kernel", &backend->cross_entropy_loss_stats_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_cross_entropy_backward_f16_kernel", &backend->cross_entropy_backward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_cross_entropy_backward_stats_f16_kernel", &backend->cross_entropy_backward_stats_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_mse_forward_f16_kernel", &backend->mse_forward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_mse_forward_f32_kernel", &backend->mse_forward_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_mse_backward_f16_kernel", &backend->mse_backward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_mse_backward_f32_kernel", &backend->mse_backward_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_huber_forward_f16_kernel", &backend->huber_forward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_huber_forward_f32_kernel", &backend->huber_forward_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_huber_backward_f16_kernel", &backend->huber_backward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_huber_backward_f32_kernel", &backend->huber_backward_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_powlu_forward_f16_kernel", &backend->powlu_forward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_powlu_backward_f16_kernel", &backend->powlu_backward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_powlu_split_forward_f16_kernel", &backend->powlu_split_forward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_powlu_split_backward_f16_kernel", &backend->powlu_split_backward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_embedding_forward_f16_kernel", &backend->embedding_forward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_embedding_forward_f32_kernel", &backend->embedding_forward_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_embedding_forward_vec16_f16_kernel", &backend->embedding_forward_vec16_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_embedding_forward_vec16_f32_kernel", &backend->embedding_forward_vec16_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_embedding_zero_f32_kernel", &backend->embedding_zero_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_embedding_backward_scatter_f16_kernel", &backend->embedding_backward_scatter_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_embedding_backward_scatter_f32_kernel", &backend->embedding_backward_scatter_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_embedding_cast_f32_to_f16_kernel", &backend->embedding_cast_f32_to_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_forward_f16_kernel", &backend->rms_norm_forward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_forward_stats_f16_kernel", &backend->rms_norm_forward_stats_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_forward_f32_kernel", &backend->rms_norm_forward_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_forward_stats_f32_kernel", &backend->rms_norm_forward_stats_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_inv_f16_kernel", &backend->rms_norm_inv_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_inv_f32_kernel", &backend->rms_norm_inv_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_backward_f16_kernel", &backend->rms_norm_backward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_backward_stats_f16_kernel", &backend->rms_norm_backward_stats_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_backward_f32_kernel", &backend->rms_norm_backward_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_backward_stats_f32_kernel", &backend->rms_norm_backward_stats_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_wgrad_stage_stats_f16_kernel", &backend->rms_norm_wgrad_stage_stats_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_wgrad_stage_stats_f32_kernel", &backend->rms_norm_wgrad_stage_stats_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_wgrad_stage_stats_f16_rb128_kernel", &backend->rms_norm_wgrad_stage_stats_f16_rb128_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_wgrad_stage_stats_f32_rb128_kernel", &backend->rms_norm_wgrad_stage_stats_f32_rb128_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_wgrad_reduce_f16_kernel", &backend->rms_norm_wgrad_reduce_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rms_norm_wgrad_reduce_f32_kernel", &backend->rms_norm_wgrad_reduce_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_concat_to_full_u8_kernel", &backend->concat_to_full_u8_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_concat_to_full_u16_kernel", &backend->concat_to_full_u16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_concat_to_full_u32_kernel", &backend->concat_to_full_u32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_concat_from_full_u8_kernel", &backend->concat_from_full_u8_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_concat_from_full_u16_kernel", &backend->concat_from_full_u16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_concat_from_full_u32_kernel", &backend->concat_from_full_u32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_split_from_full_u8_kernel", &backend->split_from_full_u8_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_split_from_full_u16_kernel", &backend->split_from_full_u16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_split_from_full_u32_kernel", &backend->split_from_full_u32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_split_to_full_u8_kernel", &backend->split_to_full_u8_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_split_to_full_u16_kernel", &backend->split_to_full_u16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_split_to_full_u32_kernel", &backend->split_to_full_u32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_split_from_full_vec16_kernel", &backend->split_from_full_vec16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_split_to_full_vec16_kernel", &backend->split_to_full_vec16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_u8_kernel", &backend->permute_u8_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_u16_kernel", &backend->permute_u16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_u32_kernel", &backend->permute_u32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_block_u8_kernel", &backend->permute_block_u8_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_block_u16_kernel", &backend->permute_block_u16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_block_u32_kernel", &backend->permute_block_u32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_suffix16_kernel", &backend->permute_suffix16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_hwc_to_chw_u8_kernel", &backend->permute_hwc_to_chw_u8_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_hwc_to_chw_u16_kernel", &backend->permute_hwc_to_chw_u16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_hwc_to_chw_u32_kernel", &backend->permute_hwc_to_chw_u32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_chw_to_hwc_u8_kernel", &backend->permute_chw_to_hwc_u8_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_chw_to_hwc_u16_kernel", &backend->permute_chw_to_hwc_u16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_chw_to_hwc_u32_kernel", &backend->permute_chw_to_hwc_u32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_transpose_u8_kernel", &backend->permute_transpose_u8_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_transpose_u16_kernel", &backend->permute_transpose_u16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_permute_transpose_u32_kernel", &backend->permute_transpose_u32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sdpa_varlen_kernel", &backend->sdpa_varlen_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sdpa_varlen_prefix_window_lane8_dh64_f16_kernel", &backend->sdpa_varlen_prefix_window_dh64_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sdpa_varlen_bwd_stats_kernel", &backend->sdpa_varlen_bwd_stats_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sdpa_varlen_bwd_kernel", &backend->sdpa_varlen_bwd_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sdpa_varlen_bwd_dkv_kernel", &backend->sdpa_varlen_bwd_dkv_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sdpa_varlen_bwd_stats_dq_prefix_window_lane8_dh64_f16_kernel", &backend->sdpa_varlen_bwd_stats_dq_dh64_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sdpa_varlen_bwd_dkv_prefix_window_k16_dh64_f16_kernel", &backend->sdpa_varlen_bwd_dkv_dh64_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sdpa_varlen_bwd_dkv_split_prefix_window_k16_dh64_f16_kernel", &backend->sdpa_varlen_bwd_dkv_split_dh64_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sdpa_varlen_bwd_dkv_reduce_f16_kernel", &backend->sdpa_varlen_bwd_dkv_reduce_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sdpa_decode_kernel", &backend->sdpa_decode_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_kv_cache_append_kernel", &backend->kv_cache_append_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_kv_cache_append_packed_kernel", &backend->kv_cache_append_packed_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_adamw_kernel", &backend->adamw_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_matmul_f16_tiled", &backend->matmul_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_linear_f16_tiled", &backend->linear_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_matmul_f16_reg", &backend->matmul_reg_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_linear_f16_reg", &backend->linear_reg_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_matmul_f16_nt_tiled", &backend->matmul_nt_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_matmul_f16_tn_tiled", &backend->matmul_tn_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_matmul_f16_nt_reg", &backend->matmul_nt_reg_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_matmul_f16_tn_reg", &backend->matmul_tn_reg_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_reduce_rows_f16", &backend->reduce_rows_pso);
    if (st != GD_OK) {
        return st;
    }
#include "metal_ops_generated.inc"
    st = gd_metal_make_pipeline(backend, library, "gd_sigmoid_f32_kernel", &backend->sigmoid_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sigmoid_backward_f32_kernel", &backend->sigmoid_backward_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sigmoid_backward_saved_f16_kernel", &backend->sigmoid_backward_saved_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_sigmoid_backward_saved_f32_kernel", &backend->sigmoid_backward_saved_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_dropout_forward_f16_kernel", &backend->dropout_forward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_dropout_forward_f32_kernel", &backend->dropout_forward_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_dropout_add_forward_f16_kernel", &backend->dropout_add_forward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_dropout_backward_recompute_f16_kernel", &backend->dropout_backward_recompute_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_dropout_backward_recompute_f32_kernel", &backend->dropout_backward_recompute_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_dropout_backward_mask_f16_kernel", &backend->dropout_backward_mask_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_dropout_backward_mask_f32_kernel", &backend->dropout_backward_mask_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rope_f16_kernel", &backend->rope_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rope_f32_kernel", &backend->rope_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rope_full_f16_kernel", &backend->rope_full_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rope_full_f32_kernel", &backend->rope_full_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rope_backward_f16_kernel", &backend->rope_backward_f16_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_make_pipeline(backend, library, "gd_rope_backward_f32_kernel", &backend->rope_backward_f32_pso);
    if (st != GD_OK) {
        return st;
    }
    return GD_OK;
}

static bool gd_metal_byte_range_valid(const gd_backend_buffer *buffer, size_t offset, size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_metal_count_bytes(size_t count, size_t elem_size, size_t *out_nbytes)
{
    if (out_nbytes == NULL || count == 0U || elem_size == 0U || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

gd_status gd_metal_command_for_op(gd_backend *backend,
                                  id<MTLCommandBuffer> *out_command_buffer,
                                  bool *out_immediate)
{
    id<MTLCommandBuffer> command_buffer;
    if (backend == NULL || out_command_buffer == NULL || out_immediate == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    command_buffer = gd_metal_active_command_buffer(backend);
    if (command_buffer != nil) {
        *out_command_buffer = command_buffer;
        *out_immediate = false;
        return GD_OK;
    }
    command_buffer = [gd_metal_queue(backend) commandBuffer];
    if (command_buffer == nil) {
        return GD_ERR_INTERNAL;
    }
    if (backend->scope_active) {
        backend->active_command_buffer = (void *)CFBridgingRetain(command_buffer);
        *out_command_buffer = gd_metal_active_command_buffer(backend);
        *out_immediate = false;
        return GD_OK;
    }
    *out_command_buffer = command_buffer;
    *out_immediate = true;
    return GD_OK;
}

gd_status gd_metal_finish_immediate(id<MTLCommandBuffer> command_buffer, bool immediate)
{
    if (!immediate) {
        return GD_OK;
    }
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
        return GD_ERR_INTERNAL;
    }
    return GD_OK;
}

gd_status gd_backend_create_default(gd_backend **out_backend)
{
    gd_backend *backend;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    gd_status st;
    if (out_backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_backend = NULL;
    device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return GD_ERR_UNSUPPORTED;
    }
    queue = [device newCommandQueue];
    if (queue == nil) {
        return GD_ERR_INTERNAL;
    }
    backend = (gd_backend *)calloc(1U, sizeof(*backend));
    if (backend == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    backend->device = (void *)CFBridgingRetain(device);
    backend->queue = (void *)CFBridgingRetain(queue);
    st = gd_metal_make_pipelines(backend);
    if (st != GD_OK) {
        gd_backend_destroy(backend);
        return st;
    }
    *out_backend = backend;
    return GD_OK;
}

void gd_backend_destroy(gd_backend *backend)
{
    uint32_t i;
    if (backend == NULL) {
        return;
    }
    if (backend->active_command_buffer != NULL) {
        CFRelease(backend->active_command_buffer);
    }
    if (backend->adamw_pso != NULL) {
        CFRelease(backend->adamw_pso);
    }
    if (backend->sdpa_decode_pso != NULL) {
        CFRelease(backend->sdpa_decode_pso);
    }
    if (backend->kv_cache_append_pso != NULL) {
        CFRelease(backend->kv_cache_append_pso);
    }
    if (backend->kv_cache_append_packed_pso != NULL) {
        CFRelease(backend->kv_cache_append_packed_pso);
    }
    if (backend->sdpa_varlen_bwd_dkv_reduce_f16_pso != NULL) {
        CFRelease(backend->sdpa_varlen_bwd_dkv_reduce_f16_pso);
    }
    if (backend->sdpa_varlen_bwd_dkv_split_dh64_f16_pso != NULL) {
        CFRelease(backend->sdpa_varlen_bwd_dkv_split_dh64_f16_pso);
    }
    if (backend->sdpa_varlen_bwd_dkv_dh64_f16_pso != NULL) {
        CFRelease(backend->sdpa_varlen_bwd_dkv_dh64_f16_pso);
    }
    if (backend->sdpa_varlen_bwd_stats_dq_dh64_f16_pso != NULL) {
        CFRelease(backend->sdpa_varlen_bwd_stats_dq_dh64_f16_pso);
    }
    if (backend->sdpa_varlen_bwd_dkv_pso != NULL) {
        CFRelease(backend->sdpa_varlen_bwd_dkv_pso);
    }
    if (backend->sdpa_varlen_bwd_pso != NULL) {
        CFRelease(backend->sdpa_varlen_bwd_pso);
    }
    if (backend->sdpa_varlen_bwd_stats_pso != NULL) {
        CFRelease(backend->sdpa_varlen_bwd_stats_pso);
    }
    if (backend->sdpa_varlen_prefix_window_dh64_f16_pso != NULL) {
        CFRelease(backend->sdpa_varlen_prefix_window_dh64_f16_pso);
    }
    if (backend->sdpa_varlen_pso != NULL) {
        CFRelease(backend->sdpa_varlen_pso);
    }
    if (backend->permute_transpose_u32_pso != NULL) {
        CFRelease(backend->permute_transpose_u32_pso);
    }
    if (backend->permute_transpose_u16_pso != NULL) {
        CFRelease(backend->permute_transpose_u16_pso);
    }
    if (backend->permute_transpose_u8_pso != NULL) {
        CFRelease(backend->permute_transpose_u8_pso);
    }
    if (backend->permute_chw_to_hwc_u32_pso != NULL) {
        CFRelease(backend->permute_chw_to_hwc_u32_pso);
    }
    if (backend->permute_chw_to_hwc_u16_pso != NULL) {
        CFRelease(backend->permute_chw_to_hwc_u16_pso);
    }
    if (backend->permute_chw_to_hwc_u8_pso != NULL) {
        CFRelease(backend->permute_chw_to_hwc_u8_pso);
    }
    if (backend->permute_hwc_to_chw_u32_pso != NULL) {
        CFRelease(backend->permute_hwc_to_chw_u32_pso);
    }
    if (backend->permute_hwc_to_chw_u16_pso != NULL) {
        CFRelease(backend->permute_hwc_to_chw_u16_pso);
    }
    if (backend->permute_hwc_to_chw_u8_pso != NULL) {
        CFRelease(backend->permute_hwc_to_chw_u8_pso);
    }
    if (backend->permute_suffix16_pso != NULL) {
        CFRelease(backend->permute_suffix16_pso);
    }
    if (backend->permute_block_u32_pso != NULL) {
        CFRelease(backend->permute_block_u32_pso);
    }
    if (backend->permute_block_u16_pso != NULL) {
        CFRelease(backend->permute_block_u16_pso);
    }
    if (backend->permute_block_u8_pso != NULL) {
        CFRelease(backend->permute_block_u8_pso);
    }
    if (backend->permute_u32_pso != NULL) {
        CFRelease(backend->permute_u32_pso);
    }
    if (backend->permute_u16_pso != NULL) {
        CFRelease(backend->permute_u16_pso);
    }
    if (backend->permute_u8_pso != NULL) {
        CFRelease(backend->permute_u8_pso);
    }
    if (backend->concat_from_full_u32_pso != NULL) {
        CFRelease(backend->concat_from_full_u32_pso);
    }
    if (backend->concat_from_full_u16_pso != NULL) {
        CFRelease(backend->concat_from_full_u16_pso);
    }
    if (backend->concat_from_full_u8_pso != NULL) {
        CFRelease(backend->concat_from_full_u8_pso);
    }
    if (backend->split_to_full_vec16_pso != NULL) {
        CFRelease(backend->split_to_full_vec16_pso);
    }
    if (backend->split_from_full_vec16_pso != NULL) {
        CFRelease(backend->split_from_full_vec16_pso);
    }
    if (backend->split_to_full_u32_pso != NULL) {
        CFRelease(backend->split_to_full_u32_pso);
    }
    if (backend->split_to_full_u16_pso != NULL) {
        CFRelease(backend->split_to_full_u16_pso);
    }
    if (backend->split_to_full_u8_pso != NULL) {
        CFRelease(backend->split_to_full_u8_pso);
    }
    if (backend->split_from_full_u32_pso != NULL) {
        CFRelease(backend->split_from_full_u32_pso);
    }
    if (backend->split_from_full_u16_pso != NULL) {
        CFRelease(backend->split_from_full_u16_pso);
    }
    if (backend->split_from_full_u8_pso != NULL) {
        CFRelease(backend->split_from_full_u8_pso);
    }
    if (backend->concat_to_full_u32_pso != NULL) {
        CFRelease(backend->concat_to_full_u32_pso);
    }
    if (backend->concat_to_full_u16_pso != NULL) {
        CFRelease(backend->concat_to_full_u16_pso);
    }
    if (backend->concat_to_full_u8_pso != NULL) {
        CFRelease(backend->concat_to_full_u8_pso);
    }
    if (backend->rope_backward_f32_pso != NULL) {
        CFRelease(backend->rope_backward_f32_pso);
    }
    if (backend->rope_backward_f16_pso != NULL) {
        CFRelease(backend->rope_backward_f16_pso);
    }
    if (backend->rope_full_f32_pso != NULL) {
        CFRelease(backend->rope_full_f32_pso);
    }
    if (backend->rope_full_f16_pso != NULL) {
        CFRelease(backend->rope_full_f16_pso);
    }
    if (backend->rope_f32_pso != NULL) {
        CFRelease(backend->rope_f32_pso);
    }
    if (backend->rope_f16_pso != NULL) {
        CFRelease(backend->rope_f16_pso);
    }
    if (backend->dropout_backward_mask_f32_pso != NULL) {
        CFRelease(backend->dropout_backward_mask_f32_pso);
    }
    if (backend->dropout_backward_mask_f16_pso != NULL) {
        CFRelease(backend->dropout_backward_mask_f16_pso);
    }
    if (backend->dropout_backward_recompute_f32_pso != NULL) {
        CFRelease(backend->dropout_backward_recompute_f32_pso);
    }
    if (backend->dropout_backward_recompute_f16_pso != NULL) {
        CFRelease(backend->dropout_backward_recompute_f16_pso);
    }
    if (backend->dropout_add_forward_f16_pso != NULL) {
        CFRelease(backend->dropout_add_forward_f16_pso);
    }
    if (backend->dropout_forward_f32_pso != NULL) {
        CFRelease(backend->dropout_forward_f32_pso);
    }
    if (backend->dropout_forward_f16_pso != NULL) {
        CFRelease(backend->dropout_forward_f16_pso);
    }
    if (backend->sigmoid_backward_saved_f32_pso != NULL) {
        CFRelease(backend->sigmoid_backward_saved_f32_pso);
    }
    if (backend->sigmoid_backward_saved_f16_pso != NULL) {
        CFRelease(backend->sigmoid_backward_saved_f16_pso);
    }
    if (backend->sigmoid_backward_f32_pso != NULL) {
        CFRelease(backend->sigmoid_backward_f32_pso);
    }
    if (backend->sigmoid_f32_pso != NULL) {
        CFRelease(backend->sigmoid_f32_pso);
    }
    if (backend->cross_entropy_backward_stats_f16_pso != NULL) {
        CFRelease(backend->cross_entropy_backward_stats_f16_pso);
    }
    if (backend->cross_entropy_backward_f16_pso != NULL) {
        CFRelease(backend->cross_entropy_backward_f16_pso);
    }
    if (backend->cross_entropy_loss_stats_f16_pso != NULL) {
        CFRelease(backend->cross_entropy_loss_stats_f16_pso);
    }
    if (backend->cross_entropy_loss_f16_pso != NULL) {
        CFRelease(backend->cross_entropy_loss_f16_pso);
    }
    if (backend->mse_backward_f32_pso != NULL) {
        CFRelease(backend->mse_backward_f32_pso);
    }
    if (backend->mse_backward_f16_pso != NULL) {
        CFRelease(backend->mse_backward_f16_pso);
    }
    if (backend->mse_forward_f32_pso != NULL) {
        CFRelease(backend->mse_forward_f32_pso);
    }
    if (backend->mse_forward_f16_pso != NULL) {
        CFRelease(backend->mse_forward_f16_pso);
    }
    if (backend->embedding_cast_f32_to_f16_pso != NULL) {
        CFRelease(backend->embedding_cast_f32_to_f16_pso);
    }
    if (backend->embedding_backward_scatter_f32_pso != NULL) {
        CFRelease(backend->embedding_backward_scatter_f32_pso);
    }
    if (backend->embedding_backward_scatter_f16_pso != NULL) {
        CFRelease(backend->embedding_backward_scatter_f16_pso);
    }
    if (backend->embedding_zero_f32_pso != NULL) {
        CFRelease(backend->embedding_zero_f32_pso);
    }
    if (backend->embedding_forward_vec16_f32_pso != NULL) {
        CFRelease(backend->embedding_forward_vec16_f32_pso);
    }
    if (backend->embedding_forward_vec16_f16_pso != NULL) {
        CFRelease(backend->embedding_forward_vec16_f16_pso);
    }
    if (backend->embedding_forward_f32_pso != NULL) {
        CFRelease(backend->embedding_forward_f32_pso);
    }
    if (backend->embedding_forward_f16_pso != NULL) {
        CFRelease(backend->embedding_forward_f16_pso);
    }
    if (backend->rms_norm_wgrad_reduce_f32_pso != NULL) {
        CFRelease(backend->rms_norm_wgrad_reduce_f32_pso);
    }
    if (backend->rms_norm_wgrad_reduce_f16_pso != NULL) {
        CFRelease(backend->rms_norm_wgrad_reduce_f16_pso);
    }
    if (backend->rms_norm_wgrad_stage_stats_f32_rb128_pso != NULL) {
        CFRelease(backend->rms_norm_wgrad_stage_stats_f32_rb128_pso);
    }
    if (backend->rms_norm_wgrad_stage_stats_f16_rb128_pso != NULL) {
        CFRelease(backend->rms_norm_wgrad_stage_stats_f16_rb128_pso);
    }
    if (backend->rms_norm_wgrad_stage_stats_f32_pso != NULL) {
        CFRelease(backend->rms_norm_wgrad_stage_stats_f32_pso);
    }
    if (backend->rms_norm_wgrad_stage_stats_f16_pso != NULL) {
        CFRelease(backend->rms_norm_wgrad_stage_stats_f16_pso);
    }
    if (backend->rms_norm_backward_stats_f32_pso != NULL) {
        CFRelease(backend->rms_norm_backward_stats_f32_pso);
    }
    if (backend->rms_norm_backward_f32_pso != NULL) {
        CFRelease(backend->rms_norm_backward_f32_pso);
    }
    if (backend->rms_norm_backward_stats_f16_pso != NULL) {
        CFRelease(backend->rms_norm_backward_stats_f16_pso);
    }
    if (backend->rms_norm_backward_f16_pso != NULL) {
        CFRelease(backend->rms_norm_backward_f16_pso);
    }
    if (backend->rms_norm_inv_f32_pso != NULL) {
        CFRelease(backend->rms_norm_inv_f32_pso);
    }
    if (backend->rms_norm_inv_f16_pso != NULL) {
        CFRelease(backend->rms_norm_inv_f16_pso);
    }
    if (backend->rms_norm_forward_stats_f32_pso != NULL) {
        CFRelease(backend->rms_norm_forward_stats_f32_pso);
    }
    if (backend->rms_norm_forward_f32_pso != NULL) {
        CFRelease(backend->rms_norm_forward_f32_pso);
    }
    if (backend->rms_norm_forward_stats_f16_pso != NULL) {
        CFRelease(backend->rms_norm_forward_stats_f16_pso);
    }
    if (backend->rms_norm_forward_f16_pso != NULL) {
        CFRelease(backend->rms_norm_forward_f16_pso);
    }
    if (backend->huber_backward_f32_pso != NULL) {
        CFRelease(backend->huber_backward_f32_pso);
    }
    if (backend->huber_backward_f16_pso != NULL) {
        CFRelease(backend->huber_backward_f16_pso);
    }
    if (backend->huber_forward_f32_pso != NULL) {
        CFRelease(backend->huber_forward_f32_pso);
    }
    if (backend->huber_forward_f16_pso != NULL) {
        CFRelease(backend->huber_forward_f16_pso);
    }
    if (backend->powlu_split_backward_f16_pso != NULL) {
        CFRelease(backend->powlu_split_backward_f16_pso);
    }
    if (backend->powlu_split_forward_f16_pso != NULL) {
        CFRelease(backend->powlu_split_forward_f16_pso);
    }
    if (backend->powlu_backward_f16_pso != NULL) {
        CFRelease(backend->powlu_backward_f16_pso);
    }
    if (backend->powlu_forward_f16_pso != NULL) {
        CFRelease(backend->powlu_forward_f16_pso);
    }
    if (backend->broadcast_scalar_f32_to_f16_pso != NULL) {
        CFRelease(backend->broadcast_scalar_f32_to_f16_pso);
    }
    if (backend->broadcast_scalar_f32_pso != NULL) {
        CFRelease(backend->broadcast_scalar_f32_pso);
    }
    if (backend->broadcast_scalar_f16_pso != NULL) {
        CFRelease(backend->broadcast_scalar_f16_pso);
    }
    if (backend->broadcast_to_f32_pso != NULL) {
        CFRelease(backend->broadcast_to_f32_pso);
    }
    if (backend->broadcast_to_f16_pso != NULL) {
        CFRelease(backend->broadcast_to_f16_pso);
    }
    if (backend->broadcast_axis_last_f32_pso != NULL) {
        CFRelease(backend->broadcast_axis_last_f32_pso);
    }
    if (backend->broadcast_axis_last_f16_pso != NULL) {
        CFRelease(backend->broadcast_axis_last_f16_pso);
    }
    if (backend->broadcast_axis_f32_pso != NULL) {
        CFRelease(backend->broadcast_axis_f32_pso);
    }
    if (backend->broadcast_axis_f16_pso != NULL) {
        CFRelease(backend->broadcast_axis_f16_pso);
    }
    if (backend->reduce_axis_last_f32_pso != NULL) {
        CFRelease(backend->reduce_axis_last_f32_pso);
    }
    if (backend->reduce_axis_last_f16_pso != NULL) {
        CFRelease(backend->reduce_axis_last_f16_pso);
    }
    if (backend->reduce_axis_f32_pso != NULL) {
        CFRelease(backend->reduce_axis_f32_pso);
    }
    if (backend->reduce_axis_f16_pso != NULL) {
        CFRelease(backend->reduce_axis_f16_pso);
    }
    if (backend->reduce_contiguous_f32_to_f16_pso != NULL) {
        CFRelease(backend->reduce_contiguous_f32_to_f16_pso);
    }
    if (backend->reduce_contiguous_f32_to_f32_pso != NULL) {
        CFRelease(backend->reduce_contiguous_f32_to_f32_pso);
    }
    if (backend->reduce_contiguous_f16_to_f32_pso != NULL) {
        CFRelease(backend->reduce_contiguous_f16_to_f32_pso);
    }
    if (backend->reduce_contiguous_f16_to_f16_pso != NULL) {
        CFRelease(backend->reduce_contiguous_f16_to_f16_pso);
    }
    if (backend->mul_reduce_suffix_small_pso != NULL) {
        CFRelease(backend->mul_reduce_suffix_small_pso);
    }
    if (backend->mul_reduce_suffix_pso != NULL) {
        CFRelease(backend->mul_reduce_suffix_pso);
    }
    if (backend->mul_backward_direct_pso != NULL) {
        CFRelease(backend->mul_backward_direct_pso);
    }
    if (backend->binary_reduce_suffix_pso != NULL) {
        CFRelease(backend->binary_reduce_suffix_pso);
    }
    if (backend->binary_reduce_pso != NULL) {
        CFRelease(backend->binary_reduce_pso);
    }
    if (backend->amp_unscale_pso != NULL) {
        CFRelease(backend->amp_unscale_pso);
    }
    if (backend->scale_pso != NULL) {
        CFRelease(backend->scale_pso);
    }
    if (backend->accumulate_pso != NULL) {
        CFRelease(backend->accumulate_pso);
    }
    for (i = 0U; i < GD_OP_COUNT; ++i) {
        if (backend->binary_row_bcast_pso[i] != NULL) {
            CFRelease(backend->binary_row_bcast_pso[i]);
        }
        if (backend->binary_bcast_pso[i] != NULL) {
            CFRelease(backend->binary_bcast_pso[i]);
        }
        if (backend->binary_pso[i] != NULL) {
            CFRelease(backend->binary_pso[i]);
        }
        if (backend->unary_backward_pso[i] != NULL) {
            CFRelease(backend->unary_backward_pso[i]);
        }
        if (backend->unary_pso[i] != NULL) {
            CFRelease(backend->unary_pso[i]);
        }
    }
    if (backend->reduce_rows_pso != NULL) {
        CFRelease(backend->reduce_rows_pso);
    }
    if (backend->mps_matmul_kernel != NULL) {
        CFRelease(backend->mps_matmul_kernel);
    }
    if (backend->matmul_tn_reg_pso != NULL) {
        CFRelease(backend->matmul_tn_reg_pso);
    }
    if (backend->matmul_nt_reg_pso != NULL) {
        CFRelease(backend->matmul_nt_reg_pso);
    }
    if (backend->matmul_tn_pso != NULL) {
        CFRelease(backend->matmul_tn_pso);
    }
    if (backend->matmul_nt_pso != NULL) {
        CFRelease(backend->matmul_nt_pso);
    }
    if (backend->linear_reg_pso != NULL) {
        CFRelease(backend->linear_reg_pso);
    }
    if (backend->matmul_reg_pso != NULL) {
        CFRelease(backend->matmul_reg_pso);
    }
    if (backend->linear_pso != NULL) {
        CFRelease(backend->linear_pso);
    }
    if (backend->matmul_pso != NULL) {
        CFRelease(backend->matmul_pso);
    }
    if (backend->rand_uniform_pso != NULL) {
        CFRelease(backend->rand_uniform_pso);
    }
    if (backend->fill_pso != NULL) {
        CFRelease(backend->fill_pso);
    }
    if (backend->queue != NULL) {
        CFRelease(backend->queue);
    }
    if (backend->device != NULL) {
        CFRelease(backend->device);
    }
    free(backend);
}

gd_backend_kind gd_backend_kind_query(const gd_backend *backend)
{
    if (backend == NULL) {
        return 0;
    }
    return GD_BACKEND_METAL;
}

const char *gd_backend_name(const gd_backend *backend)
{
    if (backend == NULL) {
        return "none";
    }
    return "metal";
}

gd_status gd_backend_buffer_create(gd_backend *backend,
                                   size_t nbytes,
                                   gd_backend_buffer **out_buffer)
{
    gd_backend_buffer *buffer;
    id<MTLBuffer> metal_buffer;
    if (backend == NULL || out_buffer == NULL || nbytes == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_buffer = NULL;
    metal_buffer = [gd_metal_device(backend) newBufferWithLength:nbytes
                                                         options:MTLResourceStorageModeShared];
    if (metal_buffer == nil) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    buffer = (gd_backend_buffer *)calloc(1U, sizeof(*buffer));
    if (buffer == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    buffer->buffer = (void *)CFBridgingRetain(metal_buffer);
    buffer->nbytes = nbytes;
    *out_buffer = buffer;
    return GD_OK;
}

void gd_backend_buffer_destroy(gd_backend_buffer *buffer)
{
    if (buffer == NULL) {
        return;
    }
    if (buffer->buffer != NULL) {
        CFRelease(buffer->buffer);
    }
    free(buffer);
}

size_t gd_backend_buffer_nbytes(const gd_backend_buffer *buffer)
{
    return buffer != NULL ? buffer->nbytes : 0U;
}

void *gd_backend_buffer_host_ptr(gd_backend_buffer *buffer)
{
    if (buffer == NULL) {
        return NULL;
    }
    return [gd_metal_buffer(buffer) contents];
}

bool gd_backend_buffer_is_host_visible(const gd_backend_buffer *buffer)
{
    return buffer != NULL;
}

gd_status gd_backend_scope_begin(gd_backend *backend)
{
    id<MTLCommandBuffer> command_buffer;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (backend->scope_active || backend->active_command_buffer != NULL) {
        return GD_ERR_BAD_STATE;
    }
    command_buffer = [gd_metal_queue(backend) commandBuffer];
    if (command_buffer == nil) {
        return GD_ERR_INTERNAL;
    }
    backend->active_command_buffer = (void *)CFBridgingRetain(command_buffer);
    backend->scope_active = true;
    return GD_OK;
}

gd_status gd_backend_flush(gd_backend *backend)
{
    id<MTLCommandBuffer> command_buffer;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (backend->active_command_buffer == NULL) {
        return GD_OK;
    }
    command_buffer = gd_metal_active_command_buffer(backend);
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
        CFRelease(backend->active_command_buffer);
        backend->active_command_buffer = NULL;
        return GD_ERR_INTERNAL;
    }
    CFRelease(backend->active_command_buffer);
    backend->active_command_buffer = NULL;
    return GD_OK;
}

gd_status gd_backend_upload(gd_backend *backend,
                            gd_backend_buffer *buffer,
                            size_t offset,
                            const void *src,
                            size_t nbytes)
{
    unsigned char *dst;
    (void)backend;
    if (buffer == NULL || src == NULL || nbytes == 0U ||
        !gd_metal_byte_range_valid(buffer, offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dst = (unsigned char *)[gd_metal_buffer(buffer) contents];
    if (dst == NULL) {
        return GD_ERR_UNSUPPORTED;
    }
    memcpy(dst + offset, src, nbytes);
    return GD_OK;
}

gd_status gd_backend_download(gd_backend *backend,
                              gd_backend_buffer *buffer,
                              size_t offset,
                              void *dst,
                              size_t nbytes)
{
    unsigned char *src;
    (void)backend;
    if (buffer == NULL || dst == NULL || nbytes == 0U ||
        !gd_metal_byte_range_valid(buffer, offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    src = (unsigned char *)[gd_metal_buffer(buffer) contents];
    if (src == NULL) {
        return GD_ERR_UNSUPPORTED;
    }
    memcpy(dst, src + offset, nbytes);
    return GD_OK;
}

gd_status gd_backend_fill(gd_backend *backend,
                          gd_backend_buffer *buffer,
                          size_t offset,
                          size_t count,
                          size_t elem_size,
                          uint32_t pattern)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_fill_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t nbytes;
    gd_status st;
    if (backend == NULL || buffer == NULL ||
        (elem_size != 1U && elem_size != 2U && elem_size != 4U) ||
        count > UINT32_MAX || !gd_metal_count_bytes(count, elem_size, &nbytes) ||
        !gd_metal_byte_range_valid(buffer, offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    args.byte_offset = (uint64_t)offset;
    args.count = (uint64_t)count;
    args.elem_size = (uint32_t)elem_size;
    args.pattern = pattern;
    [encoder setComputePipelineState:gd_metal_fill_pso(backend)];
    [encoder setBuffer:gd_metal_buffer(buffer) offset:0U atIndex:0U];
    [encoder setBytes:&args length:sizeof(args) atIndex:1U];
    grid = MTLSizeMake((NSUInteger)count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(count < GD_METAL_MAX_THREADS_PER_GROUP ? count : GD_METAL_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_rand_uniform(gd_backend *backend,
                                  gd_backend_buffer *buffer,
                                  size_t offset,
                                  size_t count,
                                  uint32_t dtype,
                                  uint64_t seed,
                                  float low,
                                  float high)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_rand_uniform_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t elem_size;
    size_t nbytes;
    gd_status st;
    if (dtype == 1U || dtype == 2U) {
        elem_size = 2U;
    } else if (dtype == 3U) {
        elem_size = 4U;
    } else {
        return GD_ERR_UNSUPPORTED;
    }
    if (backend == NULL || buffer == NULL || count > UINT32_MAX || !(low <= high) ||
        !gd_metal_count_bytes(count, elem_size, &nbytes) ||
        !gd_metal_byte_range_valid(buffer, offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    args.byte_offset = (uint64_t)offset;
    args.count = (uint64_t)count;
    args.dtype = dtype;
    args.pad0 = 0U;
    args.seed = seed;
    args.low = low;
    args.high = high;
    [encoder setComputePipelineState:gd_metal_rand_uniform_pso(backend)];
    [encoder setBuffer:gd_metal_buffer(buffer) offset:0U atIndex:0U];
    [encoder setBytes:&args length:sizeof(args) atIndex:1U];
    grid = MTLSizeMake((NSUInteger)count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(count < GD_METAL_MAX_THREADS_PER_GROUP ? count : GD_METAL_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_accumulate(gd_backend *backend,
                                gd_backend_buffer *dst_buffer,
                                size_t dst_offset,
                                gd_backend_buffer *src_buffer,
                                size_t src_offset,
                                size_t count,
                                uint32_t dtype)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_accumulate_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t elem_size;
    size_t nbytes;
    gd_status st;
    if (dtype == 1U) {
        elem_size = 2U;
    } else if (dtype == 3U) {
        elem_size = 4U;
    } else {
        return GD_ERR_UNSUPPORTED;
    }
    if (backend == NULL || dst_buffer == NULL || src_buffer == NULL || count == 0U ||
        count > UINT32_MAX || !gd_metal_count_bytes(count, elem_size, &nbytes) ||
        !gd_metal_byte_range_valid(dst_buffer, dst_offset, nbytes) ||
        !gd_metal_byte_range_valid(src_buffer, src_offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    args.dst_offset = (uint64_t)dst_offset;
    args.src_offset = (uint64_t)src_offset;
    args.count = (uint64_t)count;
    args.dtype = dtype;
    args.pad0 = 0U;
    [encoder setComputePipelineState:gd_metal_accumulate_pso(backend)];
    [encoder setBuffer:gd_metal_buffer(dst_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_buffer(src_buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(count < GD_METAL_MAX_THREADS_PER_GROUP ? count : GD_METAL_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_scale(gd_backend *backend,
                           gd_backend_buffer *dst_buffer,
                           size_t dst_offset,
                           gd_backend_buffer *src_buffer,
                           size_t src_offset,
                           size_t count,
                           uint32_t dtype,
                           float scale)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_scale_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t elem_size;
    size_t nbytes;
    gd_status st;
    if (dtype == 1U) {
        elem_size = 2U;
    } else if (dtype == 3U) {
        elem_size = 4U;
    } else {
        return GD_ERR_UNSUPPORTED;
    }
    if (backend == NULL || dst_buffer == NULL || src_buffer == NULL || count == 0U ||
        count > UINT32_MAX || scale != scale ||
        !gd_metal_count_bytes(count, elem_size, &nbytes) ||
        !gd_metal_byte_range_valid(dst_buffer, dst_offset, nbytes) ||
        !gd_metal_byte_range_valid(src_buffer, src_offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    memset(&args, 0, sizeof(args));
    args.dst_offset = (uint64_t)dst_offset;
    args.src_offset = (uint64_t)src_offset;
    args.count = (uint64_t)count;
    args.dtype = dtype;
    args.scale = scale;
    [encoder setComputePipelineState:gd_metal_scale_pso(backend)];
    [encoder setBuffer:gd_metal_buffer(dst_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_buffer(src_buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(count < GD_METAL_MAX_THREADS_PER_GROUP ? count : GD_METAL_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_record_fence(gd_backend *backend, gd_backend_fence *out_fence)
{
    id<MTLCommandBuffer> command_buffer;
    if (backend == NULL || out_fence == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    out_fence->handle = NULL;
    if (backend->active_command_buffer != NULL) {
        command_buffer = gd_metal_active_command_buffer(backend);
        [command_buffer commit];
        out_fence->handle = backend->active_command_buffer;
        backend->active_command_buffer = NULL;
        backend->scope_active = false;
        return GD_OK;
    }
    backend->scope_active = false;
    command_buffer = [gd_metal_queue(backend) commandBuffer];
    if (command_buffer == nil) {
        return GD_ERR_INTERNAL;
    }
    [command_buffer commit];
    out_fence->handle = (void *)CFBridgingRetain(command_buffer);
    return GD_OK;
}

void gd_backend_fence_destroy(gd_backend_fence *fence)
{
    if (fence == NULL) {
        return;
    }
    if (fence->handle != NULL) {
        CFRelease(fence->handle);
        fence->handle = NULL;
    }
}

bool gd_backend_fence_is_complete(gd_backend_fence *fence)
{
    id<MTLCommandBuffer> command_buffer;
    if (fence == NULL || fence->handle == NULL) {
        return true;
    }
    command_buffer = gd_metal_command_buffer(fence);
    return command_buffer.status == MTLCommandBufferStatusCompleted;
}

gd_status gd_backend_fence_wait(gd_backend_fence *fence)
{
    id<MTLCommandBuffer> command_buffer;
    if (fence == NULL || fence->handle == NULL) {
        return GD_OK;
    }
    command_buffer = gd_metal_command_buffer(fence);
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
        return GD_ERR_INTERNAL;
    }
    return GD_OK;
}
