#include "../../backends/metal/metal_backend_internal.h"
#include "../_shared/loss/metal_pairwise_loss_common.h"
#include "metal_mse_types.h"

static id<MTLComputePipelineState> gd_mse_forward_pso(gd_backend *backend, uint32_t dtype)
{
    if (backend == NULL) {
        return nil;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->mse_forward_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->mse_forward_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_mse_backward_pso(gd_backend *backend, uint32_t dtype)
{
    if (backend == NULL) {
        return nil;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->mse_backward_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->mse_backward_f32_pso;
    }
    return nil;
}

static void gd_mse_init_forward_args(void *dst,
                                     const gd_backend_tensor_view *x,
                                     const gd_backend_tensor_view *y,
                                     const gd_backend_tensor_view *out,
                                     uint64_t chunk_size,
                                     float scale,
                                     uint32_t simdgroups,
                                     const void *attrs)
{
    gd_metal_mse_args *args = (gd_metal_mse_args *)dst;
    (void)attrs;
    args->x_offset = (uint64_t)x->offset;
    args->y_offset = (uint64_t)y->offset;
    args->out_offset = (uint64_t)out->offset;
    args->count = (uint64_t)x->count;
    args->chunk_size = chunk_size;
    args->scale = scale;
    args->simdgroups = simdgroups;
}

static void gd_mse_init_backward_args(void *dst,
                                      const gd_backend_tensor_view *x,
                                      const gd_backend_tensor_view *y,
                                      const gd_backend_tensor_view *grad_out,
                                      const gd_backend_tensor_view *grad_x,
                                      const gd_backend_tensor_view *grad_y,
                                      float scale,
                                      const void *attrs)
{
    gd_metal_mse_args *args = (gd_metal_mse_args *)dst;
    (void)attrs;
    args->x_offset = (uint64_t)x->offset;
    args->y_offset = (uint64_t)y->offset;
    args->grad_out_offset = (uint64_t)grad_out->offset;
    args->out_offset = grad_x != NULL ? (uint64_t)grad_x->offset : 0U;
    args->dy_offset = grad_y != NULL ? (uint64_t)grad_y->offset : 0U;
    args->count = (uint64_t)x->count;
    args->scale = scale;
    args->write_x = grad_x != NULL ? 1U : 0U;
    args->write_y = grad_y != NULL ? 1U : 0U;
}

static const gd_metal_pairwise_loss_config GD_MSE_METAL_LOSS_CONFIG = {
    .args_size = sizeof(gd_metal_mse_args),
    .max_simdgroups = GD_METAL_MSE_MAX_SIMDGROUPS,
    .forward_pso = gd_mse_forward_pso,
    .backward_pso = gd_mse_backward_pso,
    .validate_attrs = NULL,
    .init_forward_args = gd_mse_init_forward_args,
    .init_backward_args = gd_mse_init_backward_args,
};

gd_status gd_backend_mse_forward(gd_backend *backend,
                                 const gd_backend_tensor_view *x,
                                 const gd_backend_tensor_view *y,
                                 const gd_backend_tensor_view *out,
                                 uint64_t chunk_size,
                                 float scale)
{
    return gd_metal_pairwise_loss_forward(backend,
                                          x,
                                          y,
                                          out,
                                          chunk_size,
                                          scale,
                                          NULL,
                                          &GD_MSE_METAL_LOSS_CONFIG);
}

gd_status gd_backend_mse_backward(gd_backend *backend,
                                  const gd_backend_tensor_view *x,
                                  const gd_backend_tensor_view *y,
                                  const gd_backend_tensor_view *grad_out,
                                  const gd_backend_tensor_view *grad_x,
                                  const gd_backend_tensor_view *grad_y,
                                  float scale)
{
    return gd_metal_pairwise_loss_backward(backend,
                                           x,
                                           y,
                                           grad_out,
                                           grad_x,
                                           grad_y,
                                           scale,
                                           NULL,
                                           &GD_MSE_METAL_LOSS_CONFIG);
}
