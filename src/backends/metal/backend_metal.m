#include "metal_backend_internal.h"
#include "primitives/memory/metal_memory_types.h"
#include "primitives/random/metal_random_types.h"

#import <Foundation/Foundation.h>

#include <stddef.h>
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

typedef struct gd_metal_pipeline_spec {
    const char *kernel_name;
    size_t backend_field_offset;
} gd_metal_pipeline_spec;

#define GD_METAL_PIPELINE(kernel_, field_) \
    { (kernel_), offsetof(gd_backend, field_) }
#define GD_METAL_PIPELINE_INDEX(kernel_, array_, index_) \
    { \
        (kernel_), \
        offsetof(gd_backend, array_) + \
            ((size_t)(index_) * sizeof(((gd_backend *)0)->array_[0])) \
    }

static void **gd_metal_pipeline_slot(gd_backend *backend, size_t field_offset)
{
    if (backend == NULL) {
        return NULL;
    }
    return (void **)((unsigned char *)backend + field_offset);
}

static const gd_metal_pipeline_spec gd_metal_pipeline_specs[] = {
    GD_METAL_PIPELINE("gd_fill_kernel", fill_pso),
    GD_METAL_PIPELINE("gd_rand_uniform_kernel", rand_uniform_pso),
    GD_METAL_PIPELINE("gd_accumulate_kernel", accumulate_pso),
    GD_METAL_PIPELINE("gd_scale_kernel", scale_pso),
    GD_METAL_PIPELINE("gd_amp_begin_step_kernel", amp_begin_step_pso),
    GD_METAL_PIPELINE("gd_amp_finish_step_kernel", amp_finish_step_pso),
    GD_METAL_PIPELINE("gd_amp_fill_scale_kernel", amp_fill_scale_pso),
    GD_METAL_PIPELINE("gd_amp_scale_kernel", amp_scale_pso),
    GD_METAL_PIPELINE("gd_amp_unscale_kernel", amp_unscale_pso),
    GD_METAL_PIPELINE("gd_grad_norm_stage_kernel", grad_norm_stage_pso),
    GD_METAL_PIPELINE("gd_grad_clip_finalize_kernel", grad_clip_finalize_pso),
    GD_METAL_PIPELINE("gd_binary_reduce_broadcast_kernel", binary_reduce_pso),
    GD_METAL_PIPELINE("gd_binary_reduce_broadcast_suffix_kernel", binary_reduce_suffix_pso),
    GD_METAL_PIPELINE("gd_mul_backward_direct_kernel", mul_backward_direct_pso),
    GD_METAL_PIPELINE("gd_mul_reduce_suffix_kernel", mul_reduce_suffix_pso),
    GD_METAL_PIPELINE("gd_mul_reduce_suffix_small_kernel", mul_reduce_suffix_small_pso),
    GD_METAL_PIPELINE("gd_reduce_contiguous_f16_to_f16_kernel", reduce_contiguous_f16_to_f16_pso),
    GD_METAL_PIPELINE("gd_reduce_contiguous_f16_to_f32_kernel", reduce_contiguous_f16_to_f32_pso),
    GD_METAL_PIPELINE("gd_reduce_contiguous_f32_to_f32_kernel", reduce_contiguous_f32_to_f32_pso),
    GD_METAL_PIPELINE("gd_reduce_contiguous_f32_to_f16_kernel", reduce_contiguous_f32_to_f16_pso),
    GD_METAL_PIPELINE("gd_reduce_axis_f16_kernel", reduce_axis_f16_pso),
    GD_METAL_PIPELINE("gd_reduce_axis_f32_kernel", reduce_axis_f32_pso),
    GD_METAL_PIPELINE("gd_reduce_axis_last_f16_kernel", reduce_axis_last_f16_pso),
    GD_METAL_PIPELINE("gd_reduce_axis_last_f32_kernel", reduce_axis_last_f32_pso),
    GD_METAL_PIPELINE("gd_broadcast_axis_f16_kernel", broadcast_axis_f16_pso),
    GD_METAL_PIPELINE("gd_broadcast_axis_f32_kernel", broadcast_axis_f32_pso),
    GD_METAL_PIPELINE("gd_broadcast_axis_last_f16_kernel", broadcast_axis_last_f16_pso),
    GD_METAL_PIPELINE("gd_broadcast_axis_last_f32_kernel", broadcast_axis_last_f32_pso),
    GD_METAL_PIPELINE("gd_broadcast_to_f16_kernel", broadcast_to_f16_pso),
    GD_METAL_PIPELINE("gd_broadcast_to_f32_kernel", broadcast_to_f32_pso),
    GD_METAL_PIPELINE("gd_broadcast_scalar_f16_kernel", broadcast_scalar_f16_pso),
    GD_METAL_PIPELINE("gd_broadcast_scalar_f32_kernel", broadcast_scalar_f32_pso),
    GD_METAL_PIPELINE("gd_broadcast_scalar_f32_to_f16_kernel", broadcast_scalar_f32_to_f16_pso),
    GD_METAL_PIPELINE("gd_cross_entropy_loss_f16_kernel", cross_entropy_loss_f16_pso),
    GD_METAL_PIPELINE("gd_cross_entropy_loss_stats_f16_kernel", cross_entropy_loss_stats_f16_pso),
    GD_METAL_PIPELINE("gd_cross_entropy_backward_f16_kernel", cross_entropy_backward_f16_pso),
    GD_METAL_PIPELINE("gd_cross_entropy_backward_stats_f16_kernel", cross_entropy_backward_stats_f16_pso),
    GD_METAL_PIPELINE("gd_lm_cross_entropy_online_update_f16_kernel", lm_cross_entropy_online_update_f16_pso),
    GD_METAL_PIPELINE("gd_lm_cross_entropy_finalize_f32_kernel", lm_cross_entropy_finalize_f32_pso),
    GD_METAL_PIPELINE("gd_lm_cross_entropy_reduce_normalize_f32_kernel", lm_cross_entropy_reduce_normalize_f32_pso),
    GD_METAL_PIPELINE("gd_lm_cross_entropy_backward_chunk_f16_kernel", lm_cross_entropy_backward_chunk_f16_pso),
    GD_METAL_PIPELINE("gd_mse_forward_f16_kernel", mse_forward_f16_pso),
    GD_METAL_PIPELINE("gd_mse_forward_f32_kernel", mse_forward_f32_pso),
    GD_METAL_PIPELINE("gd_mse_backward_f16_kernel", mse_backward_f16_pso),
    GD_METAL_PIPELINE("gd_mse_backward_f32_kernel", mse_backward_f32_pso),
    GD_METAL_PIPELINE("gd_huber_forward_f16_kernel", huber_forward_f16_pso),
    GD_METAL_PIPELINE("gd_huber_forward_f32_kernel", huber_forward_f32_pso),
    GD_METAL_PIPELINE("gd_huber_backward_f16_kernel", huber_backward_f16_pso),
    GD_METAL_PIPELINE("gd_huber_backward_f32_kernel", huber_backward_f32_pso),
    GD_METAL_PIPELINE("gd_powlu_forward_f16_kernel", powlu_forward_f16_pso),
    GD_METAL_PIPELINE("gd_powlu_backward_f16_kernel", powlu_backward_f16_pso),
    GD_METAL_PIPELINE("gd_powlu_split_forward_f16_kernel", powlu_split_forward_f16_pso),
    GD_METAL_PIPELINE("gd_powlu_split_backward_f16_kernel", powlu_split_backward_f16_pso),
    GD_METAL_PIPELINE("gd_powlu_split_linear_backward_x12_f16_reg_kernel", powlu_split_linear_backward_x12_f16_reg_pso),
    GD_METAL_PIPELINE("gd_embedding_forward_f16_kernel", embedding_forward_f16_pso),
    GD_METAL_PIPELINE("gd_embedding_forward_f32_kernel", embedding_forward_f32_pso),
    GD_METAL_PIPELINE("gd_embedding_forward_vec16_f16_kernel", embedding_forward_vec16_f16_pso),
    GD_METAL_PIPELINE("gd_embedding_forward_vec16_f32_kernel", embedding_forward_vec16_f32_pso),
    GD_METAL_PIPELINE("gd_embedding_zero_f32_kernel", embedding_zero_f32_pso),
    GD_METAL_PIPELINE("gd_embedding_backward_scatter_f16_kernel", embedding_backward_scatter_f16_pso),
    GD_METAL_PIPELINE("gd_embedding_backward_scatter_f32_kernel", embedding_backward_scatter_f32_pso),
    GD_METAL_PIPELINE("gd_embedding_cast_f32_to_f16_kernel", embedding_cast_f32_to_f16_pso),
    GD_METAL_PIPELINE("gd_rms_norm_forward_f16_kernel", rms_norm_forward_f16_pso),
    GD_METAL_PIPELINE("gd_rms_norm_forward_stats_f16_kernel", rms_norm_forward_stats_f16_pso),
    GD_METAL_PIPELINE("gd_rms_norm_forward_f32_kernel", rms_norm_forward_f32_pso),
    GD_METAL_PIPELINE("gd_rms_norm_forward_stats_f32_kernel", rms_norm_forward_stats_f32_pso),
    GD_METAL_PIPELINE("gd_rms_norm_inv_f16_kernel", rms_norm_inv_f16_pso),
    GD_METAL_PIPELINE("gd_rms_norm_inv_f32_kernel", rms_norm_inv_f32_pso),
    GD_METAL_PIPELINE("gd_rms_norm_backward_f16_kernel", rms_norm_backward_f16_pso),
    GD_METAL_PIPELINE("gd_rms_norm_backward_stats_f16_kernel", rms_norm_backward_stats_f16_pso),
    GD_METAL_PIPELINE("gd_rms_norm_backward_f32_kernel", rms_norm_backward_f32_pso),
    GD_METAL_PIPELINE("gd_rms_norm_backward_stats_f32_kernel", rms_norm_backward_stats_f32_pso),
    GD_METAL_PIPELINE("gd_rms_norm_wgrad_stage_stats_f16_kernel", rms_norm_wgrad_stage_stats_f16_pso),
    GD_METAL_PIPELINE("gd_rms_norm_wgrad_stage_stats_f32_kernel", rms_norm_wgrad_stage_stats_f32_pso),
    GD_METAL_PIPELINE("gd_rms_norm_wgrad_stage_stats_f16_rb128_kernel", rms_norm_wgrad_stage_stats_f16_rb128_pso),
    GD_METAL_PIPELINE("gd_rms_norm_wgrad_stage_stats_f32_rb128_kernel", rms_norm_wgrad_stage_stats_f32_rb128_pso),
    GD_METAL_PIPELINE("gd_rms_norm_wgrad_reduce_f16_kernel", rms_norm_wgrad_reduce_f16_pso),
    GD_METAL_PIPELINE("gd_rms_norm_wgrad_reduce_f32_kernel", rms_norm_wgrad_reduce_f32_pso),
    GD_METAL_PIPELINE("gd_concat_to_full_u8_kernel", concat_to_full_u8_pso),
    GD_METAL_PIPELINE("gd_concat_to_full_u16_kernel", concat_to_full_u16_pso),
    GD_METAL_PIPELINE("gd_concat_to_full_u32_kernel", concat_to_full_u32_pso),
    GD_METAL_PIPELINE("gd_concat_from_full_u8_kernel", concat_from_full_u8_pso),
    GD_METAL_PIPELINE("gd_concat_from_full_u16_kernel", concat_from_full_u16_pso),
    GD_METAL_PIPELINE("gd_concat_from_full_u32_kernel", concat_from_full_u32_pso),
    GD_METAL_PIPELINE("gd_split_from_full_u8_kernel", split_from_full_u8_pso),
    GD_METAL_PIPELINE("gd_split_from_full_u16_kernel", split_from_full_u16_pso),
    GD_METAL_PIPELINE("gd_split_from_full_u32_kernel", split_from_full_u32_pso),
    GD_METAL_PIPELINE("gd_split_to_full_u8_kernel", split_to_full_u8_pso),
    GD_METAL_PIPELINE("gd_split_to_full_u16_kernel", split_to_full_u16_pso),
    GD_METAL_PIPELINE("gd_split_to_full_u32_kernel", split_to_full_u32_pso),
    GD_METAL_PIPELINE("gd_split_from_full_vec16_kernel", split_from_full_vec16_pso),
    GD_METAL_PIPELINE("gd_split_to_full_vec16_kernel", split_to_full_vec16_pso),
    GD_METAL_PIPELINE("gd_permute_u8_kernel", permute_u8_pso),
    GD_METAL_PIPELINE("gd_permute_u16_kernel", permute_u16_pso),
    GD_METAL_PIPELINE("gd_permute_u32_kernel", permute_u32_pso),
    GD_METAL_PIPELINE("gd_permute_block_u8_kernel", permute_block_u8_pso),
    GD_METAL_PIPELINE("gd_permute_block_u16_kernel", permute_block_u16_pso),
    GD_METAL_PIPELINE("gd_permute_block_u32_kernel", permute_block_u32_pso),
    GD_METAL_PIPELINE("gd_permute_suffix16_kernel", permute_suffix16_pso),
    GD_METAL_PIPELINE("gd_permute_hwc_to_chw_u8_kernel", permute_hwc_to_chw_u8_pso),
    GD_METAL_PIPELINE("gd_permute_hwc_to_chw_u16_kernel", permute_hwc_to_chw_u16_pso),
    GD_METAL_PIPELINE("gd_permute_hwc_to_chw_u32_kernel", permute_hwc_to_chw_u32_pso),
    GD_METAL_PIPELINE("gd_permute_chw_to_hwc_u8_kernel", permute_chw_to_hwc_u8_pso),
    GD_METAL_PIPELINE("gd_permute_chw_to_hwc_u16_kernel", permute_chw_to_hwc_u16_pso),
    GD_METAL_PIPELINE("gd_permute_chw_to_hwc_u32_kernel", permute_chw_to_hwc_u32_pso),
    GD_METAL_PIPELINE("gd_permute_transpose_u8_kernel", permute_transpose_u8_pso),
    GD_METAL_PIPELINE("gd_permute_transpose_u16_kernel", permute_transpose_u16_pso),
    GD_METAL_PIPELINE("gd_permute_transpose_u32_kernel", permute_transpose_u32_pso),
    GD_METAL_PIPELINE("gd_sdpa_varlen_kernel", sdpa_varlen_pso),
    GD_METAL_PIPELINE("gd_sdpa_varlen_prefix_window_lane8_dh64_f16_kernel", sdpa_varlen_prefix_window_dh64_f16_pso),
    GD_METAL_PIPELINE("gd_sdpa_varlen_bwd_stats_kernel", sdpa_varlen_bwd_stats_pso),
    GD_METAL_PIPELINE("gd_sdpa_varlen_bwd_kernel", sdpa_varlen_bwd_pso),
    GD_METAL_PIPELINE("gd_sdpa_varlen_bwd_dkv_kernel", sdpa_varlen_bwd_dkv_pso),
    GD_METAL_PIPELINE("gd_sdpa_varlen_bwd_stats_dq_prefix_window_lane8_dh64_f16_kernel", sdpa_varlen_bwd_stats_dq_dh64_f16_pso),
    GD_METAL_PIPELINE("gd_sdpa_varlen_bwd_dkv_prefix_window_k16_dh64_f16_kernel", sdpa_varlen_bwd_dkv_dh64_f16_pso),
    GD_METAL_PIPELINE("gd_sdpa_varlen_bwd_dkv_split_prefix_window_k16_dh64_f16_kernel", sdpa_varlen_bwd_dkv_split_dh64_f16_pso),
    GD_METAL_PIPELINE("gd_sdpa_varlen_bwd_dkv_reduce_f16_kernel", sdpa_varlen_bwd_dkv_reduce_f16_pso),
    GD_METAL_PIPELINE("gd_sdpa_decode_kernel", sdpa_decode_pso),
    GD_METAL_PIPELINE("gd_sdpa_decode_tq1_dh64_f16_kernel", sdpa_decode_tq1_dh64_f16_pso),
    GD_METAL_PIPELINE("gd_kv_cache_append_kernel", kv_cache_append_pso),
    GD_METAL_PIPELINE("gd_kv_cache_append_packed_kernel", kv_cache_append_packed_pso),
    GD_METAL_PIPELINE("gd_adamw_prepare_kernel", adamw_prepare_pso),
    GD_METAL_PIPELINE("gd_adamw_kernel", adamw_pso),
    GD_METAL_PIPELINE("gd_adamw_commit_kernel", adamw_commit_pso),
    GD_METAL_PIPELINE("gd_matmul_f16_tiled", matmul_pso),
    GD_METAL_PIPELINE("gd_linear_f16_tiled", linear_pso),
    GD_METAL_PIPELINE("gd_matmul_f16_reg", matmul_reg_pso),
    GD_METAL_PIPELINE("gd_linear_f16_reg", linear_reg_pso),
    GD_METAL_PIPELINE("gd_matmul_f16_nt_tiled", matmul_nt_pso),
    GD_METAL_PIPELINE("gd_matmul_f16_tn_tiled", matmul_tn_pso),
    GD_METAL_PIPELINE("gd_matmul_f16_nt_reg", matmul_nt_reg_pso),
    GD_METAL_PIPELINE("gd_matmul_f16_tn_reg", matmul_tn_reg_pso),
    GD_METAL_PIPELINE("gd_matmul_f16_tn_split8", matmul_tn_split8_pso),
    GD_METAL_PIPELINE("gd_reduce_rows_f16", reduce_rows_pso),
#include "metal_ops_generated.inc"
    GD_METAL_PIPELINE("gd_sigmoid_f32_kernel", sigmoid_f32_pso),
    GD_METAL_PIPELINE("gd_sigmoid_backward_f32_kernel", sigmoid_backward_f32_pso),
    GD_METAL_PIPELINE("gd_sigmoid_backward_saved_f16_kernel", sigmoid_backward_saved_f16_pso),
    GD_METAL_PIPELINE("gd_sigmoid_backward_saved_f32_kernel", sigmoid_backward_saved_f32_pso),
    GD_METAL_PIPELINE("gd_tanh_f32_kernel", tanh_f32_pso),
    GD_METAL_PIPELINE("gd_tanh_backward_f32_kernel", tanh_backward_f32_pso),
    GD_METAL_PIPELINE("gd_tanh_backward_saved_f16_kernel", tanh_backward_saved_f16_pso),
    GD_METAL_PIPELINE("gd_tanh_backward_saved_f32_kernel", tanh_backward_saved_f32_pso),
    GD_METAL_PIPELINE("gd_dropout_forward_f16_kernel", dropout_forward_f16_pso),
    GD_METAL_PIPELINE("gd_dropout_forward_f32_kernel", dropout_forward_f32_pso),
    GD_METAL_PIPELINE("gd_dropout_add_forward_f16_kernel", dropout_add_forward_f16_pso),
    GD_METAL_PIPELINE("gd_dropout_backward_recompute_f16_kernel", dropout_backward_recompute_f16_pso),
    GD_METAL_PIPELINE("gd_dropout_backward_recompute_f32_kernel", dropout_backward_recompute_f32_pso),
    GD_METAL_PIPELINE("gd_dropout_backward_mask_f16_kernel", dropout_backward_mask_f16_pso),
    GD_METAL_PIPELINE("gd_dropout_backward_mask_f32_kernel", dropout_backward_mask_f32_pso),
    GD_METAL_PIPELINE("gd_rope_f16_kernel", rope_f16_pso),
    GD_METAL_PIPELINE("gd_rope_f32_kernel", rope_f32_pso),
    GD_METAL_PIPELINE("gd_rope_full_f16_kernel", rope_full_f16_pso),
    GD_METAL_PIPELINE("gd_rope_full_f32_kernel", rope_full_f32_pso),
    GD_METAL_PIPELINE("gd_rope_backward_f16_kernel", rope_backward_f16_pso),
    GD_METAL_PIPELINE("gd_rope_backward_f32_kernel", rope_backward_f32_pso),
    GD_METAL_PIPELINE("gd_qkv_split_rope_forward_f16_kernel", qkv_split_rope_forward_f16_pso),
    GD_METAL_PIPELINE("gd_qkv_split_rope_backward_f16_kernel", qkv_split_rope_backward_f16_pso),
};

static gd_status gd_metal_make_pipelines(gd_backend *backend)
{
    NSError *error = nil;
    NSURL *url;
    id<MTLLibrary> library;
    size_t i;
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
    for (i = 0U; i < sizeof(gd_metal_pipeline_specs) / sizeof(gd_metal_pipeline_specs[0]); ++i) {
        gd_status st = gd_metal_make_pipeline(
            backend,
            library,
            gd_metal_pipeline_specs[i].kernel_name,
            gd_metal_pipeline_slot(backend, gd_metal_pipeline_specs[i].backend_field_offset));
        if (st != GD_OK) {
            return st;
        }
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

static void gd_metal_release_retained(void **object)
{
    if (object != NULL && *object != NULL) {
        CFRelease(*object);
        *object = NULL;
    }
}

static void gd_metal_release_pipelines(gd_backend *backend)
{
    size_t i;
    if (backend == NULL) {
        return;
    }
    for (i = 0U; i < sizeof(gd_metal_pipeline_specs) / sizeof(gd_metal_pipeline_specs[0]); ++i) {
        gd_metal_release_retained(
            gd_metal_pipeline_slot(backend, gd_metal_pipeline_specs[i].backend_field_offset));
    }
}

void gd_backend_destroy(gd_backend *backend)
{
    if (backend == NULL) {
        return;
    }
    gd_metal_release_retained(&backend->active_command_buffer);
    gd_metal_release_pipelines(backend);
    gd_metal_release_retained(&backend->mps_matmul_kernel);
    gd_metal_release_retained(&backend->queue);
    gd_metal_release_retained(&backend->device);
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
