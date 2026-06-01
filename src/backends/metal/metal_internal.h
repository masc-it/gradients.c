#ifndef GD_METAL_INTERNAL_H
#define GD_METAL_INTERNAL_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "../backend.h"
#include "../../core/internal.h"
#include "../../core/storage_internal.h"
#include "../../core/tensor_internal.h"
#include "../../graph/graph_internal.h"
#include "metal_kernel_types.h"

@interface GDMetalState : NSObject
@property (strong) id<MTLDevice> device;
@property (strong) id<MTLCommandQueue> queue;
@property (strong) id<MTLLibrary> library;
@property (strong) NSMutableDictionary<NSNumber *, id<MTLComputePipelineState>> *pipelines;
@property (strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelinesByName;
@property (strong) id<MTLCommandBuffer> inFlight;
@property (strong) NSMutableArray<NSValue *> *pendingExes;
@property BOOL useMPS;
@end

@interface GDMPSGemmPlan : NSObject
@property (strong) MPSMatrixMultiplication *kernel;
@property (strong) MPSMatrix *left;
@property (strong) MPSMatrix *right;
@property (strong) MPSMatrix *result;
@end

typedef struct gd_metal_kernel_entry {
    _gd_op_kind op;
    const char *fn;
} gd_metal_kernel_entry;

#define GD_METAL_RMS_TG 256
#define GD_METAL_CLIP_NORM_TG 256
#define GD_METAL_CLIP_NORM_ITEMS 4
#define GD_METAL_CLIP_NORM_CHUNK (GD_METAL_CLIP_NORM_TG * GD_METAL_CLIP_NORM_ITEMS)

typedef struct gd_metal_sdpa_bwd_layout {
    int64_t stats_off;
    int64_t stats_part_off;
    int64_t dq_part_off;
    int64_t dkv_part_off;
    int64_t total;
} gd_metal_sdpa_bwd_layout;

typedef struct gd_metal_lmce_scratch_layout {
    size_t logits_off;
    size_t target_logit_off;
    size_t losses_off;
    size_t total;
} gd_metal_lmce_scratch_layout;

typedef struct gd_metal_value {
    gd_storage *storage;
    gd_tensor *external;
    size_t leaf_offset;
    bool external_alias;
    bool needs_writeback;
    bool has_staged;
    uint64_t staged_version;
} gd_metal_value;

struct _gd_executable {
    const gd_graph *graph;
    int n_values;
    gd_metal_value *values;
    bool needs_stage;
    bool needs_writeback;
    int n_plan;
    void **node_pso;
    void **node_pso2;
    void **node_pso3;
    void **node_mps;
    size_t *node_scratch_bytes;
    gd_storage *scratch_arena;
    size_t scratch_arena_bytes;
    uint8_t *node_absorbed;
    int *node_fused_src;
};

GDMetalState *_gd_metal_state(_gd_backend *self);
id<MTLComputePipelineState> _gd_metal_pipeline_for(GDMetalState *st, _gd_op_kind op);
id<MTLComputePipelineState> _gd_metal_pipeline_named(GDMetalState *st, const char *name);
int64_t _gd_metal_desc_numel(const gd_tensor_desc *desc);
bool _gd_metal_desc_same_shape(const gd_tensor_desc *a, const gd_tensor_desc *b);
gd_status _gd_metal_dtype_code(gd_dtype dtype, int *out);

gd_metal_sdpa_bwd_layout _gd_metal_sdpa_bwd_scratch_layout(int B, int Hq, int Hkv,
                                                           int Tq, int Tk, int Dh, int S);
int _gd_metal_sdpa_num_splits(int Tk);
int _gd_metal_sdpa_split_len(int Tk, int n_splits);
gd_metal_lmce_scratch_layout _gd_metal_lmce_fwd_scratch_layout_for(int rows, int chunk);
size_t _gd_metal_lmce_bwd_scratch_bytes_for(int rows, int chunk);

void _gd_metal_build_ew_params(gd_metal_ew_params *p,
                               const gd_tensor_desc *out_desc,
                               const gd_tensor_desc *a_desc,
                               const gd_tensor_desc *b_desc);
id<MTLBuffer> _gd_metal_value_buffer(_gd_executable *exe, int value_id);
void _gd_metal_dispatch_1d(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso,
                           NSUInteger numel);
void _gd_metal_dispatch_gemm_tiles(id<MTLComputeCommandEncoder> enc,
                                   NSUInteger cols,
                                   NSUInteger rows,
                                   NSUInteger batch);
void _gd_metal_dispatch_reduce_groups(id<MTLComputeCommandEncoder> enc, NSUInteger groups);

MPSMatrix *_gd_metal_mps_matrix(id<MTLBuffer> buffer,
                                NSUInteger offset,
                                NSUInteger rows,
                                NSUInteger cols,
                                NSUInteger row_bytes);
gd_status _gd_metal_encode_mps_mm(id<MTLCommandBuffer> cmd,
                                  id<MTLComputeCommandEncoder> *enc,
                                  id<MTLDevice> device,
                                  MPSMatrix *left,
                                  MPSMatrix *right,
                                  MPSMatrix *result,
                                  BOOL trans_left,
                                  BOOL trans_right,
                                  NSUInteger rows,
                                  NSUInteger cols,
                                  NSUInteger inner,
                                  double beta);
gd_status _gd_metal_encode_mps_gemm(id<MTLCommandBuffer> cmd,
                                    id<MTLComputeCommandEncoder> *enc,
                                    GDMPSGemmPlan *plan);

gd_status _gd_metal_storage_alloc(_gd_backend *self, const gd_storage_desc *desc,
                                  void **handle_out);
void _gd_metal_storage_free(_gd_backend *self, void *handle);
gd_status _gd_metal_storage_host_ptr(_gd_backend *self, void *handle, void **ptr_out);
gd_status _gd_metal_upload(_gd_backend *self, void *dst_handle, size_t dst_off,
                           const void *src, size_t nbytes);
gd_status _gd_metal_download(_gd_backend *self, void *src_handle, size_t src_off,
                             void *dst, size_t nbytes);
gd_status _gd_metal_synchronize(_gd_backend *self);
gd_status _gd_metal_flush_pending(_gd_backend *self, void *cookie);
gd_status _gd_metal_writeback_externals(_gd_backend *self, _gd_executable *exe);
gd_status _gd_metal_flush_pending_writebacks(_gd_backend *self);
void _gd_metal_register_pending_writeback(_gd_backend *self, _gd_executable *exe);

gd_status _gd_metal_compile(_gd_backend *self, gd_graph *graph, _gd_executable **out);
void _gd_metal_executable_free(_gd_backend *self, _gd_executable *exe);

gd_status _gd_metal_execute(_gd_backend *self, _gd_executable *exe);
gd_status _gd_metal_execute_until(_gd_backend *self, _gd_executable *exe, int node_id);
gd_status _gd_metal_value_storage(_gd_backend *self, _gd_executable *exe, int value_id,
                                  gd_storage **storage_out, size_t *offset_out);
bool _gd_metal_supports_node(_gd_backend *self, const _gd_node *node);

void _gd_metal_plan_fusions(_gd_backend *self, const gd_graph *graph, _gd_executable *exe);
gd_status _gd_metal_encode_fused_head(id<MTLComputeCommandEncoder> enc,
                                      id<MTLComputePipelineState> pso,
                                      _gd_executable *exe,
                                      const _gd_node *head,
                                      const _gd_node *src);

gd_status _gd_metal_encode_node(_gd_backend *self,
                                id<MTLComputeCommandEncoder> enc,
                                _gd_executable *exe,
                                const _gd_node *node,
                                id<MTLComputePipelineState> pso,
                                id<MTLComputePipelineState> pso2,
                                id<MTLComputePipelineState> pso3,
                                id<MTLBuffer> scratch);
bool _gd_metal_encode_core_node(_gd_backend *self,
                                id<MTLComputeCommandEncoder> enc,
                                _gd_executable *exe,
                                const _gd_node *node,
                                id<MTLComputePipelineState> pso,
                                id<MTLComputePipelineState> pso2,
                                id<MTLBuffer> scratch,
                                gd_status *status_out);
bool _gd_metal_encode_misc_node(_gd_backend *self,
                                id<MTLComputeCommandEncoder> enc,
                                _gd_executable *exe,
                                const _gd_node *node,
                                id<MTLComputePipelineState> pso,
                                id<MTLComputePipelineState> pso2,
                                id<MTLComputePipelineState> pso3,
                                id<MTLBuffer> scratch,
                                gd_status *status_out);
gd_status _gd_metal_encode_lm_cross_entropy(_gd_backend *self,
                                            id<MTLCommandBuffer> cmd,
                                            id<MTLComputeCommandEncoder> *enc,
                                            id<MTLBuffer> scratch,
                                            _gd_executable *exe,
                                            const _gd_node *node);
gd_status _gd_metal_encode_lm_cross_entropy_bwd(_gd_backend *self,
                                                id<MTLCommandBuffer> cmd,
                                                id<MTLComputeCommandEncoder> *enc,
                                                id<MTLBuffer> scratch,
                                                _gd_executable *exe,
                                                const _gd_node *node);
void _gd_metal_encode_sdpa(_gd_backend *self,
                           id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso,
                           id<MTLComputePipelineState> splitk_pso,
                           id<MTLComputePipelineState> combine_pso,
                           id<MTLBuffer> scratch,
                           _gd_executable *exe,
                           const _gd_node *node);
gd_status _gd_metal_encode_sdpa_bwd(_gd_backend *self,
                                    id<MTLComputeCommandEncoder> enc,
                                    id<MTLComputePipelineState> dq_pso,
                                    id<MTLComputePipelineState> dkv_pso,
                                    id<MTLComputePipelineState> stats_pso,
                                    id<MTLBuffer> stats_buf,
                                    _gd_executable *exe,
                                    const _gd_node *node);

#endif /* GD_METAL_INTERNAL_H */
