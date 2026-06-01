#include "../op_impl.h"

const _gd_op_def _gd_opdef_sdpa = {
    .kind = _GD_OP_SDPA,
    .name = "sdpa",
    .min_inputs = 3,
    .max_inputs = 4,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = _gd_meta_not_implemented,
};
