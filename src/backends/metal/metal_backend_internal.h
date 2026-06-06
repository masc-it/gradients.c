#ifndef GD_METAL_BACKEND_INTERNAL_H
#define GD_METAL_BACKEND_INTERNAL_H

#import <Metal/Metal.h>

#include <stdbool.h>
#include <stddef.h>

#include "../../core/backend.h"
#include "../../ops/op_kind.h"

struct gd_backend {
    void *device;
    void *queue;
    void *fill_pso;
    void *rand_uniform_pso;
    void *matmul_pso;
    void *linear_pso;
    void *matmul_reg_pso;
    void *linear_reg_pso;
    void *matmul_nt_pso;
    void *matmul_tn_pso;
    void *matmul_nt_reg_pso;
    void *matmul_tn_reg_pso;
    void *reduce_rows_pso;
    void *accumulate_pso;
    void *scale_pso;
    void *amp_unscale_pso;
    void *unary_pso[GD_OP_COUNT];
    void *unary_backward_pso[GD_OP_COUNT];
    void *sigmoid_f32_pso;
    void *sigmoid_backward_f32_pso;
    void *sigmoid_backward_saved_f16_pso;
    void *sigmoid_backward_saved_f32_pso;
    void *dropout_forward_f16_pso;
    void *dropout_forward_f32_pso;
    void *dropout_backward_recompute_f16_pso;
    void *dropout_backward_recompute_f32_pso;
    void *dropout_backward_mask_f16_pso;
    void *dropout_backward_mask_f32_pso;
    void *binary_pso[GD_OP_COUNT];
    void *binary_bcast_pso[GD_OP_COUNT];
    void *binary_row_bcast_pso[GD_OP_COUNT];
    void *binary_reduce_pso;
    void *binary_reduce_suffix_pso;
    void *mul_backward_direct_pso;
    void *mul_reduce_suffix_pso;
    void *mul_reduce_suffix_small_pso;
    void *reduce_contiguous_f16_to_f16_pso;
    void *reduce_contiguous_f16_to_f32_pso;
    void *reduce_contiguous_f32_to_f32_pso;
    void *reduce_contiguous_f32_to_f16_pso;
    void *reduce_axis_f16_pso;
    void *reduce_axis_f32_pso;
    void *reduce_axis_last_f16_pso;
    void *reduce_axis_last_f32_pso;
    void *broadcast_axis_f16_pso;
    void *broadcast_axis_f32_pso;
    void *broadcast_axis_last_f16_pso;
    void *broadcast_axis_last_f32_pso;
    void *broadcast_to_f16_pso;
    void *broadcast_to_f32_pso;
    void *broadcast_scalar_f16_pso;
    void *broadcast_scalar_f32_pso;
    void *broadcast_scalar_f32_to_f16_pso;
    void *cross_entropy_loss_f16_pso;
    void *cross_entropy_loss_stats_f16_pso;
    void *cross_entropy_backward_f16_pso;
    void *cross_entropy_backward_stats_f16_pso;
    void *mse_forward_f16_pso;
    void *mse_forward_f32_pso;
    void *mse_backward_f16_pso;
    void *mse_backward_f32_pso;
    void *huber_forward_f16_pso;
    void *huber_forward_f32_pso;
    void *huber_backward_f16_pso;
    void *huber_backward_f32_pso;
    void *concat_to_full_u8_pso;
    void *concat_to_full_u16_pso;
    void *concat_to_full_u32_pso;
    void *concat_from_full_u8_pso;
    void *concat_from_full_u16_pso;
    void *concat_from_full_u32_pso;
    void *sdpa_varlen_pso;
    void *sdpa_varlen_prefix_window_dh64_f16_pso;
    void *sdpa_varlen_bwd_stats_pso;
    void *sdpa_varlen_bwd_pso;
    void *sdpa_varlen_bwd_dkv_pso;
    void *sdpa_varlen_bwd_stats_dq_dh64_f16_pso;
    void *sdpa_varlen_bwd_dkv_dh64_f16_pso;
    void *sdpa_varlen_bwd_dkv_split_dh64_f16_pso;
    void *sdpa_varlen_bwd_dkv_reduce_f16_pso;
    void *sdpa_decode_pso;
    void *adamw_pso;
    void *active_command_buffer;
    bool scope_active;
};

struct gd_backend_buffer {
    void *buffer;
    size_t nbytes;
};

gd_status gd_metal_command_for_op(gd_backend *backend,
                                  id<MTLCommandBuffer> *out_command_buffer,
                                  bool *out_immediate);
gd_status gd_metal_finish_immediate(id<MTLCommandBuffer> command_buffer, bool immediate);

#endif /* GD_METAL_BACKEND_INTERNAL_H */
