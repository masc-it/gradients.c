#include "../op_impl.h"

const _gd_op_def _gd_opdef_sdpa_bwd = {
    .kind = _GD_OP_SDPA_BWD,
    .name = "sdpa_bwd",
    .min_inputs = 4,
    .max_inputs = 5,
    .n_outputs = 3,
    .flags = GD_OPF_INTERNAL | GD_OPF_AUX_OUTS,
    .meta = _gd_meta_not_implemented,
};
