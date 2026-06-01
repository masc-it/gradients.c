#include "../op_impl.h"

const _gd_op_def _gd_opdef_clip_grad_norm = {
    .kind = _GD_OP_CLIP_GRAD_NORM,
    .name = "clip_grad_norm",
    .min_inputs = 1,
    .max_inputs = 256,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_MUTATES,
    .meta = _gd_meta_not_implemented,
};
