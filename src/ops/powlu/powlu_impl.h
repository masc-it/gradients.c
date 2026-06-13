#ifndef GD_OP_POWLU_IMPL_H
#define GD_OP_POWLU_IMPL_H

#include <stdint.h>

typedef struct gd_powlu_attrs {
    float m;
} gd_powlu_attrs;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_powlu_attrs) == 4U, "gd_powlu_attrs ABI mismatch");
#endif

#endif /* GD_OP_POWLU_IMPL_H */
