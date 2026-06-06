#ifndef GD_OP_SDPA_VARLEN_IMPL_H
#define GD_OP_SDPA_VARLEN_IMPL_H

#include <stdint.h>

#define GD_SDPA_VARLEN_MAX_HEAD_DIM 64

typedef struct gd_sdpa_varlen_attrs {
    float scale;
    int32_t causal;
    int32_t sliding_window;
    int32_t prefix_len;
    int32_t max_seqlen;
} gd_sdpa_varlen_attrs;

#endif /* GD_OP_SDPA_VARLEN_IMPL_H */
