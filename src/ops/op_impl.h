#ifndef GRADIENTS_OP_IMPL_H
#define GRADIENTS_OP_IMPL_H

#include <stdbool.h>

#include "gradients/context.h"
#include "gradients/status.h"
#include "gradients/tensor.h"
#include "op_kind.h"

typedef struct _gd_op_attrs _gd_op_attrs;

typedef enum _gd_op_flags {
    GD_OPF_PUBLIC      = 1u << 0,
    GD_OPF_INTERNAL    = 1u << 1,
    GD_OPF_DIFF        = 1u << 2,
    GD_OPF_MUTATES     = 1u << 3,
    GD_OPF_SIDE_EFFECT = 1u << 4,
    GD_OPF_DEBUG       = 1u << 5,
    GD_OPF_BROADCAST   = 1u << 6,
    GD_OPF_AUX_OUTS    = 1u << 7
} _gd_op_flags;

typedef gd_status (*_gd_meta_fn)(const gd_tensor_desc *const *inputs,
                                 int n_inputs,
                                 _gd_op_attrs *attrs,
                                 gd_tensor_desc *outputs,
                                 int *n_outputs);

typedef struct _gd_op_def {
    _gd_op_kind kind;
    const char *name;
    int min_inputs;
    int max_inputs;
    int n_outputs;
    unsigned flags;
    _gd_meta_fn meta;
} _gd_op_def;

static inline gd_status _gd_meta_not_implemented(const gd_tensor_desc *const *inputs,
                                                 int n_inputs,
                                                 _gd_op_attrs *attrs,
                                                 gd_tensor_desc *outputs,
                                                 int *n_outputs)
{
    (void)inputs;
    (void)n_inputs;
    (void)attrs;
    (void)outputs;
    (void)n_outputs;
    return GD_ERR_UNSUPPORTED;
}

const _gd_op_def *_gd_op_def_for(_gd_op_kind kind);
const char *_gd_op_kind_name(_gd_op_kind kind);
gd_status _gd_op_validate_arity(_gd_op_kind kind, int n_inputs, int n_outputs);
bool _gd_op_is_differentiable(_gd_op_kind kind);

gd_status _gd_emit_checked(gd_context *ctx,
                           _gd_op_kind op,
                           gd_tensor **inputs,
                           int n_inputs,
                           _gd_op_attrs *attrs,
                           gd_tensor **outputs,
                           int n_outputs);

#endif /* GRADIENTS_OP_IMPL_H */
