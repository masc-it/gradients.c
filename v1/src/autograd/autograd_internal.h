#ifndef GRADIENTS_AUTOGRAD_INTERNAL_H
#define GRADIENTS_AUTOGRAD_INTERNAL_H

#include "gradients/context.h"
#include "gradients/status.h"
#include "gradients/tensor.h"

#include "../graph/graph_internal.h"
#include "../ops/op_impl.h"

typedef struct _gd_bwd_ctx {
    gd_context *ctx;
    gd_graph *graph;
    int n_values;
    gd_tensor **fwd;   /* cached forward value handles (virtual owned or external borrowed) */
    gd_tensor **grad;  /* accumulated gradient handles (owned) */
    gd_tensor *ones;   /* scalar seed (owned) */
} _gd_bwd_ctx;

typedef gd_status (*_gd_bwd_rule_fn)(_gd_bwd_ctx *b, const _gd_node *node);

typedef struct _gd_bwd_rule {
    _gd_op_kind op;
    _gd_bwd_rule_fn fn;
    const char *unsupported_reason;
} _gd_bwd_rule;

const _gd_bwd_rule *_gd_bwd_rule_for(_gd_op_kind op);

gd_context *_gd_bwd_context(_gd_bwd_ctx *b);
gd_graph *_gd_bwd_graph(_gd_bwd_ctx *b);
const gd_tensor_desc *_gd_bwd_value_desc(_gd_bwd_ctx *b, int value_id);

gd_status _gd_bwd_fwd(_gd_bwd_ctx *b, int value_id, gd_tensor **out);
gd_tensor *_gd_bwd_grad(_gd_bwd_ctx *b, int value_id);

gd_status _gd_bwd_accumulate(_gd_bwd_ctx *b, int value_id, gd_tensor *contrib);
gd_status _gd_bwd_accumulate_broadcast(_gd_bwd_ctx *b, int value_id, gd_tensor *grad);

gd_status _gd_bwd_emit(_gd_bwd_ctx *b,
                       _gd_op_kind op,
                       gd_tensor **inputs,
                       int n_inputs,
                       const _gd_op_attrs *attrs,
                       const gd_tensor_desc *out_desc,
                       gd_tensor **out);

gd_status _gd_bwd_emit_multi(_gd_bwd_ctx *b,
                             _gd_op_kind op,
                             gd_tensor **inputs,
                             int n_inputs,
                             const _gd_op_attrs *attrs,
                             const gd_tensor_desc *out_descs,
                             int n_outputs,
                             gd_tensor **outs);

#endif /* GRADIENTS_AUTOGRAD_INTERNAL_H */
