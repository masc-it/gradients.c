#import "metal_internal.h"

#include <stdlib.h>

static const gd_metal_kernel_entry g_metal_kernels[] = {
    {_GD_OP_ADD, "gd_add"},
    {_GD_OP_MUL, "gd_mul"},
    {_GD_OP_SCALE, "gd_scale"},
    {_GD_OP_RELU, "gd_relu"},
    {_GD_OP_SILU, "gd_silu"},
    {_GD_OP_DROPOUT, "gd_dropout"},
    {_GD_OP_POWLU, "gd_powlu"},
    {_GD_OP_COPY, "gd_copy"},
    {_GD_OP_CAST, "gd_cast"},
    {_GD_OP_MATMUL, "gd_matmul_tiled"},
    {_GD_OP_LINEAR, "gd_linear_tiled"},
    {_GD_OP_SUM, "gd_reduce"},
    {_GD_OP_MEAN, "gd_mean"},
    {_GD_OP_SOFTMAX, "gd_softmax"},
    {_GD_OP_RMS_NORM, "gd_rms_norm"},
    {_GD_OP_CROSS_ENTROPY, "gd_cross_entropy"},
    {_GD_OP_RELU_BWD, "gd_relu_bwd"},
    {_GD_OP_SILU_BWD, "gd_silu_bwd"},
    {_GD_OP_DROPOUT_BWD, "gd_dropout_bwd"},
    {_GD_OP_POWLU_BWD, "gd_powlu_bwd"},
    {_GD_OP_SOFTMAX_BWD, "gd_softmax_bwd"},
    {_GD_OP_SUM_BWD, "gd_sum_bwd"},
    {_GD_OP_MEAN_BWD, "gd_mean_bwd"},
    {_GD_OP_CROSS_ENTROPY_BWD, "gd_cross_entropy_bwd"},
    {_GD_OP_REDUCE_TO, "gd_reduce_to"},
    {_GD_OP_CLIP_GRAD_NORM, "gd_clip_norm_partial"},
    {_GD_OP_STEP_INC, "gd_step_inc"},
    {_GD_OP_ADAMW_STEP, "gd_adamw"},
    {_GD_OP_AMP_UNSCALE_GRAD, "gd_amp_unscale_grad"},
    {_GD_OP_AMP_CLIP_GRAD_NORM, "gd_amp_clip_norm_partial"},
    {_GD_OP_AMP_STEP_INC, "gd_amp_step_inc"},
    {_GD_OP_GELU, "gd_gelu"},
    {_GD_OP_GELU_BWD, "gd_gelu_bwd"},
    {_GD_OP_TRANSPOSE, "gd_transpose"},
    {_GD_OP_EMBEDDING, "gd_embedding"},
    {_GD_OP_EMBEDDING_BWD, "gd_embedding_bwd"},
    {_GD_OP_ROPE, "gd_rope"},
    {_GD_OP_ROPE_BWD, "gd_rope_bwd"},
    {_GD_OP_SDPA, "gd_sdpa_tiled"},
    {_GD_OP_SDPA_BWD, "gd_sdpa_bwd_dq"},
    {_GD_OP_RMS_NORM_BWD, "gd_rms_norm_bwd"},
    {_GD_OP_RMS_NORM_WBWD, "gd_rms_norm_wbwd"},
    {_GD_OP_SDPA_VARLEN, "gd_sdpa_varlen"},
    {_GD_OP_SDPA_VARLEN_BWD, "gd_sdpa_varlen_bwd"},
    {_GD_OP_SLICE, "gd_slice"},
    {_GD_OP_SLICE_BWD, "gd_slice_bwd"},
    {_GD_OP_CONCAT, "gd_concat"},
};

/* Kernels not mapped 1:1 to an op (looked up by name during encode). */
static const char *const g_metal_extra_kernels[] = {
    "gd_sdpa_bwd_dkv",
    "gd_sdpa_bwd_stats",
    "gd_sdpa_varlen_bwd_dkv",
    "gd_sdpa_varlen_bwd_stats",
    "gd_sdpa_varlen_prefix_window_lane8_dh64_f16",
    "gd_sdpa_varlen_bwd_stats_dq_prefix_window_lane8_dh64_f16",
    "gd_sdpa_varlen_bwd_dkv_prefix_window_k16_dh64_f16",
    "gd_sdpa_varlen_bwd_dkv_split_prefix_window_k16_dh64_f16",
    "gd_sdpa_varlen_bwd_dkv_reduce_f16",
    "gd_sdpa", /* reference forward; fallback when head_dim > GD_METAL_SDPA_DHT */
    "gd_sdpa_tiled_causal", /* GPT-style causal/no-bias forward */
    "gd_sdpa_splitk",  /* split-K forward (long context) */
    "gd_sdpa_splitk_causal", /* GPT-style causal/no-bias split-K forward */
    "gd_sdpa_splitk_causal_lane8", /* channel-lane causal split-K forward */
    "gd_sdpa_splitk_causal_window_lane8", /* sliding-window causal split-K forward */
    "gd_sdpa_splitk_causal_window_lane8_f16", /* F16 sliding-window causal split-K forward */
    "gd_sdpa_combine", /* split-K combine pass */
    "gd_sdpa_f16", /* F16 reference forward */
    "gd_sdpa_tiled_f16", /* F16 tiled forward */
    "gd_sdpa_splitk_f16", /* F16 split-K forward */
    "gd_sdpa_combine_f16", /* F16 split-K combine pass */
    "gd_sdpa_bwd_stats_dq_split",
    "gd_sdpa_bwd_stats_dq_split_causal",
    "gd_sdpa_bwd_stats_dq_split_causal_lane8",
    "gd_sdpa_bwd_stats_dq_split_causal_window_lane8",
    "gd_sdpa_bwd_stats_dq_split_causal_window_lane8_f16",
    "gd_sdpa_bwd_stats_dq_split_causal_window_lane8_dh64_f16",
    "gd_sdpa_bwd_stats_dq_split_prefix_window_lane8_dh64_f16",
    "gd_sdpa_bwd_stats_dq_combine",
    "gd_sdpa_bwd_stats_dq_combine_f16",
    "gd_sdpa_bwd_dkv_split",
    "gd_sdpa_bwd_dkv_split_causal",
    "gd_sdpa_bwd_dkv_split_causal_k16",
    "gd_sdpa_bwd_dkv_split_causal_window_k16",
    "gd_sdpa_bwd_dkv_split_causal_window_k16_f16",
    "gd_sdpa_bwd_dkv_split_causal_window_k16_dh64_f16",
    "gd_sdpa_bwd_dkv_split_prefix_window_k16_dh64_f16",
    "gd_sdpa_bwd_dkv_reduce",
    "gd_sdpa_bwd_dkv_reduce_f16",
    "gd_cast_f16_to_f32_x4",
    "gd_cast_f32_to_f16_x4",
    "gd_silu_mul", /* F1: fused SwiGLU activation */
    "gd_add_rms_norm", /* F4: residual add + RMSNorm forward */
    "gd_rms_norm_bwd_f16", /* F16 RMSNorm backward */
    "gd_rms_norm_bwd_add", /* F4: RMSNorm backward + residual grad add */
    "gd_cross_entropy_reduce",
    "gd_cross_entropy_count_valid",
    "gd_clip_norm_reduce",
    "gd_clip_norm_scale",
    "gd_amp_clip_norm_reduce",
    "gd_amp_clip_norm_scale",
    "gd_lmce_fwd_chunk",
    "gd_lmce_loss_rows",
    "gd_lmce_dlogits_chunk",
    "gd_lmce_store_dx_f16",
    "gd_embedding_bwd_scatter",
    "gd_rms_norm_wbwd_reduce",
};

static bool env_flag_enabled(const char *name)
{
    const char *v = getenv(name);
    if (v == NULL || v[0] == '\0') {
        return false;
    }
    return strcmp(v, "0") != 0 && strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0 &&
           strcmp(v, "off") != 0 && strcmp(v, "OFF") != 0;
}

static NSString *resolve_metallib_path(void)
{
    const char *env = getenv("GRADIENTS_METALLIB");

    if (env != NULL && env[0] != '\0') {
        return [NSString stringWithUTF8String:env];
    }
    return @"build/gradients.metallib";
}

static NSUInteger metal_buffer_pool_max_bytes(void)
{
    const char *v = getenv("GD_METAL_BUFFER_POOL_MB");
    char *end = NULL;
    unsigned long mb = 1024UL;

    if (v != NULL && v[0] != '\0') {
        mb = strtoul(v, &end, 10);
        if (end == v || *end != '\0') {
            mb = 1024UL;
        }
    }
    if (mb > (unsigned long)(NSUIntegerMax / (1024UL * 1024UL))) {
        return NSUIntegerMax;
    }
    return (NSUInteger)mb * 1024U * 1024U;
}

static gd_status _gd_metal_init(_gd_backend *self, gd_context *ctx, int device_index)
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = nil;
        id<MTLLibrary> library = nil;
        NSError *error = nil;
        NSString *path = resolve_metallib_path();
        GDMetalState *st = nil;
        size_t k = 0;

        (void)device_index;
        if (device == nil) {
            return _gd_error(GD_ERR_UNSUPPORTED, "no Metal device available");
        }
        queue = [device newCommandQueue];
        if (queue == nil) {
            return _gd_error(GD_ERR_BACKEND, "failed to create Metal command queue");
        }
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
        if (library == nil) {
            return _gd_error(GD_ERR_UNSUPPORTED,
                             "failed to load gradients.metallib (set GRADIENTS_METALLIB)");
        }

        st = [GDMetalState new];
        st.device = device;
        st.queue = queue;
        st.library = library;
        st.pipelines = [NSMutableDictionary dictionary];
        st.pipelinesByName = [NSMutableDictionary dictionary];
        st.inFlight = nil;
        st.inFlightBuffers = [NSMutableArray array];
        st.pendingExes = [NSMutableArray array];
        st.bufferPool = [NSMutableDictionary dictionary];
        st.bufferPoolBytes = 0U;
        st.bufferPoolMaxBytes = metal_buffer_pool_max_bytes();
        st.useMPS = env_flag_enabled("GD_METAL_MPS") ? YES : NO;

        for (k = 0; k < sizeof(g_metal_kernels) / sizeof(g_metal_kernels[0]); ++k) {
            NSString *name = [NSString stringWithUTF8String:g_metal_kernels[k].fn];
            id<MTLFunction> fn = [library newFunctionWithName:name];
            id<MTLComputePipelineState> pso = nil;

            if (fn == nil) {
                return _gd_error(GD_ERR_BACKEND, "metallib is missing a required kernel");
            }
            pso = [device newComputePipelineStateWithFunction:fn error:&error];
            if (pso == nil) {
                return _gd_error(GD_ERR_BACKEND, "failed to create compute pipeline state");
            }
            st.pipelines[@((int)g_metal_kernels[k].op)] = pso;
            st.pipelinesByName[name] = pso;
        }
        for (k = 0; k < sizeof(g_metal_extra_kernels) / sizeof(g_metal_extra_kernels[0]); ++k) {
            NSString *name = [NSString stringWithUTF8String:g_metal_extra_kernels[k]];
            id<MTLFunction> fn = [library newFunctionWithName:name];
            id<MTLComputePipelineState> pso = nil;

            if (fn == nil) {
                return _gd_error(GD_ERR_BACKEND, "metallib is missing a required kernel");
            }
            pso = [device newComputePipelineStateWithFunction:fn error:&error];
            if (pso == nil) {
                return _gd_error(GD_ERR_BACKEND, "failed to create compute pipeline state");
            }
            st.pipelinesByName[name] = pso;
        }

        self->impl = (__bridge_retained void *)st;
        self->caps.host_visible = true;
        self->caps.supports_cpu_ref = false;
        self->caps.default_memory = GD_MEM_UNIFIED;
        (void)ctx;
        return GD_OK;
    }
}

static void _gd_metal_shutdown(_gd_backend *self)
{
    if (self->impl != NULL) {
        GDMetalState *st = (__bridge_transfer GDMetalState *)self->impl; /* ARC releases */
        st.inFlight = nil;
        [st.inFlightBuffers removeAllObjects];
        [st.pendingExes removeAllObjects];
        [st.bufferPool removeAllObjects];
        st.bufferPoolBytes = 0U;
        self->impl = NULL;
    }
}

static const _gd_backend_vtable metal_backend_vtable = {
    .type = GD_DEVICE_METAL,
    .name = "metal",
    .init = _gd_metal_init,
    .shutdown = _gd_metal_shutdown,
    .storage_alloc = _gd_metal_storage_alloc,
    .storage_free = _gd_metal_storage_free,
    .storage_host_ptr = _gd_metal_storage_host_ptr,
    .upload = _gd_metal_upload,
    .download = _gd_metal_download,
    .compile = _gd_metal_compile,
    .execute = _gd_metal_execute,
    .execute_bound = _gd_metal_execute_bound,
    .execute_until = _gd_metal_execute_until,
    .executable_free = _gd_metal_executable_free,
    .value_storage = _gd_metal_value_storage,
    .check_node = _gd_metal_check_node,
    .synchronize = _gd_metal_synchronize,
    .flush_pending = _gd_metal_flush_pending,
};

gd_status _gd_metal_backend_register(gd_context *ctx)
{
    return _gd_context_register_backend(ctx, &metal_backend_vtable);
}
