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
    void *binary_pso[GD_OP_COUNT];
    void *binary_bcast_pso[GD_OP_COUNT];
    void *binary_row_bcast_pso[GD_OP_COUNT];
    void *binary_reduce_pso;
    void *binary_reduce_suffix_pso;
    void *mul_backward_direct_pso;
    void *mul_reduce_suffix_pso;
    void *mul_reduce_suffix_small_pso;
    void *reduce_contiguous_pso;
    void *reduce_axis_pso;
    void *broadcast_axis_pso;
    void *broadcast_to_pso;
    void *cross_entropy_loss_f16_pso;
    void *cross_entropy_loss_stats_f16_pso;
    void *cross_entropy_backward_f16_pso;
    void *cross_entropy_backward_stats_f16_pso;
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
