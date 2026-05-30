#ifndef GRADIENTS_STORAGE_INTERNAL_H
#define GRADIENTS_STORAGE_INTERNAL_H

#include <stddef.h>

#include "gradients/tensor.h"

void *_gd_storage_data_mut(gd_storage *storage);
const void *_gd_storage_data(const gd_storage *storage);
size_t _gd_storage_nbytes(const gd_storage *storage);
const gd_storage_desc *_gd_storage_desc(const gd_storage *storage);

/* Raw backend allocation handle (e.g. an id<MTLBuffer>). Only meaningful to the
 * backend that allocated the storage; opaque to core. */
void *_gd_storage_handle(const gd_storage *storage);

#endif /* GRADIENTS_STORAGE_INTERNAL_H */
