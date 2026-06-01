#include "../op_impl.h"

const _gd_op_def _gd_opdef_assert_close = {
    .kind = _GD_OP_ASSERT_CLOSE,
    .name = "assert_close",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 0,
    .flags = GD_OPF_PUBLIC | GD_OPF_SIDE_EFFECT | GD_OPF_DEBUG,
    .meta = _gd_meta_not_implemented,
};
