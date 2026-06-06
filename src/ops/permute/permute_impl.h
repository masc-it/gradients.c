#ifndef GD_OP_PERMUTE_IMPL_H
#define GD_OP_PERMUTE_IMPL_H

#include <gradients/ops.h>

#include <stdint.h>

#define GD_PERMUTE_MAX_DIMS GD_MAX_DIMS

typedef struct gd_permute_attrs {
    uint32_t n_axes;
    int32_t axes[GD_PERMUTE_MAX_DIMS];
} gd_permute_attrs;

gd_status gd_permute_resolve_axes(gd_context *ctx,
                                  uint32_t rank,
                                  const int32_t *axes,
                                  uint32_t n_axes,
                                  uint32_t normalized[GD_PERMUTE_MAX_DIMS]);

#endif /* GD_OP_PERMUTE_IMPL_H */
