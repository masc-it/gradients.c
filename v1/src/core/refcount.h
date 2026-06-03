#ifndef GRADIENTS_REFCOUNT_H
#define GRADIENTS_REFCOUNT_H

#include <limits.h>
#include <stdatomic.h>

#include "gradients/status.h"

#define GD_REFCOUNT_INIT 1

typedef atomic_int gd_refcount;

static inline void _gd_refcount_init(gd_refcount *refcount)
{
    atomic_init(refcount, GD_REFCOUNT_INIT);
}

static inline gd_status _gd_refcount_retain(gd_refcount *refcount)
{
    int old = atomic_load_explicit(refcount, memory_order_relaxed);

    while (old > 0 && old < INT_MAX) {
        if (atomic_compare_exchange_weak_explicit(refcount,
                                                  &old,
                                                  old + 1,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) {
            return GD_OK;
        }
    }

    return GD_ERR_INVALID_STATE;
}

static inline int _gd_refcount_release(gd_refcount *refcount)
{
    return atomic_fetch_sub_explicit(refcount, 1, memory_order_acq_rel) == 1;
}

#endif /* GRADIENTS_REFCOUNT_H */
