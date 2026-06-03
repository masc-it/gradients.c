#import "metal_internal.h"

#include <stdlib.h>

void _gd_metal_scratch_slot_retain(gd_metal_scratch_slot *slot)
{
    if (slot != NULL) {
        (void)atomic_fetch_add_explicit(&slot->refcount, 1U, memory_order_relaxed);
    }
}

void _gd_metal_scratch_slot_release(gd_metal_scratch_slot *slot)
{
    if (slot == NULL) {
        return;
    }
    if (atomic_fetch_sub_explicit(&slot->refcount, 1U, memory_order_acq_rel) == 1U) {
        gd_storage_release(slot->storage);
        free(slot);
    }
}
