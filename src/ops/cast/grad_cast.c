#include "../grad_impl.h"

const _gd_bwd_rule _gd_bwd_rule_cast = {
    .op = _GD_OP_CAST,
    .fn = NULL,
    .unsupported_reason = "cast backward is not supported in v1",
};
