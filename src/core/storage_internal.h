#ifndef GRADIENTS_STORAGE_INTERNAL_H
#define GRADIENTS_STORAGE_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "gradients/tensor.h"

typedef struct _gd_backend _gd_backend;

/* Lazy device->host writeback markers (P4). A backend that has computed newer
 * bytes for a host storage but deferred copying them back registers itself; the
 * next host read resolves it via the backend's flush_pending hook. */
void _gd_storage_set_pending_flush(gd_storage *storage, _gd_backend *backend, void *cookie);
void _gd_storage_clear_pending_flush(gd_storage *storage);

void *_gd_storage_data_mut(gd_storage *storage);
const void *_gd_storage_data(const gd_storage *storage);
size_t _gd_storage_nbytes(const gd_storage *storage);
uint64_t _gd_storage_version(const gd_storage *storage);
const gd_storage_desc *_gd_storage_desc(const gd_storage *storage);

/* Raw backend allocation handle (e.g. an id<MTLBuffer>). Only meaningful to the
 * backend that allocated the storage; opaque to core. */
void *_gd_storage_handle(const gd_storage *storage);

#endif /* GRADIENTS_STORAGE_INTERNAL_H */
