#ifndef GD_OPS_AUTOGRAD_IMPL_H
#define GD_OPS_AUTOGRAD_IMPL_H

#include "../autograd/autograd_internal.h"

#define GD_TRY(expr)                     \
    do {                                 \
        gd_status gd_try_status = (expr); \
        if (gd_try_status != GD_OK) {    \
            return gd_try_status;        \
        }                                \
    } while (0)

#endif /* GD_OPS_AUTOGRAD_IMPL_H */
