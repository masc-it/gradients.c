#import "metal_internal.h"

/* Reuse transient Metal buffers across dynamic graphs. MTLBuffer allocation churn
 * can grow process footprint enough for MPS F16 GEMMs to return NaNs. Size
 * classes keep ragged batches reusable without global padding. */
#define GD_METAL_BUFFER_POOL_MIN_CLASS ((NSUInteger)4096U)
#define GD_METAL_BUFFER_POOL_SMALL_MAX ((NSUInteger)1048576U)
#define GD_METAL_BUFFER_POOL_LARGE_GRAN ((NSUInteger)1048576U)

static NSUInteger metal_buffer_pool_size_class(size_t nbytes)
{
    NSUInteger n = (NSUInteger)nbytes;
    NSUInteger c = GD_METAL_BUFFER_POOL_MIN_CLASS;

    if (n <= c) {
        return c;
    }
    if (n <= GD_METAL_BUFFER_POOL_SMALL_MAX) {
        while (c < n && c <= (NSUIntegerMax / (NSUInteger)2U)) {
            c *= (NSUInteger)2U;
        }
        return c;
    }
    if (n > NSUIntegerMax - (GD_METAL_BUFFER_POOL_LARGE_GRAN - (NSUInteger)1U)) {
        return n;
    }
    return ((n + GD_METAL_BUFFER_POOL_LARGE_GRAN - (NSUInteger)1U) /
            GD_METAL_BUFFER_POOL_LARGE_GRAN) * GD_METAL_BUFFER_POOL_LARGE_GRAN;
}

static bool metal_buffer_pool_has_inflight(GDMetalState *st)
{
    return st.inFlight != nil || st.inFlightBuffers.count > 0U;
}

static void metal_buffer_pool_evict_one_locked(GDMetalState *st)
{
    NSNumber *evict_key = nil;
    NSMutableArray<id<MTLBuffer>> *evict_bucket = nil;
    id<MTLBuffer> evict = nil;

    for (NSNumber *key in st.bufferPool) {
        NSMutableArray<id<MTLBuffer>> *bucket = st.bufferPool[key];
        if (bucket.count > 0U) {
            evict_key = key;
            evict_bucket = bucket;
            break;
        }
    }
    if (evict_bucket == nil) {
        st.bufferPoolBytes = 0U;
        return;
    }
    evict = evict_bucket.lastObject;
    [evict_bucket removeLastObject];
    if (evict.length <= st.bufferPoolBytes) {
        st.bufferPoolBytes -= evict.length;
    } else {
        st.bufferPoolBytes = 0U;
    }
    if (evict_bucket.count == 0U && evict_key != nil) {
        [st.bufferPool removeObjectForKey:evict_key];
    }
}

static void metal_buffer_pool_trim_for_locked(GDMetalState *st, NSUInteger needed)
{
    while (st.bufferPoolBytes > 0U && needed <= st.bufferPoolMaxBytes &&
           st.bufferPoolBytes > st.bufferPoolMaxBytes - needed) {
        metal_buffer_pool_evict_one_locked(st);
    }
}

gd_status _gd_metal_storage_alloc(_gd_backend *self, const gd_storage_desc *desc,
                                     void **handle_out)
{
    GDMetalState *st = _gd_metal_state(self);

    *handle_out = NULL;
    if (desc->device.type != GD_DEVICE_METAL) {
        return _gd_error(GD_ERR_DEVICE, "metal storage requires a Metal device");
    }
    if (desc->memory != GD_MEM_UNIFIED && desc->memory != GD_MEM_HOST) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal storage supports unified/host memory in v1");
    }

    {
        NSUInteger class_nbytes = metal_buffer_pool_size_class(desc->nbytes);
        id<MTLBuffer> buffer = nil;

        if (st.bufferPoolMaxBytes > 0U && class_nbytes <= st.bufferPoolMaxBytes) {
            NSNumber *key = @(class_nbytes);
            @synchronized(st) {
                NSMutableArray<id<MTLBuffer>> *bucket = st.bufferPool[key];
                if (bucket.count > 0U) {
                    buffer = bucket.lastObject;
                    [bucket removeLastObject];
                    if (buffer.length <= st.bufferPoolBytes) {
                        st.bufferPoolBytes -= buffer.length;
                    } else {
                        st.bufferPoolBytes = 0U;
                    }
                    if (bucket.count == 0U) {
                        [st.bufferPool removeObjectForKey:key];
                    }
                }
            }
        }
        if (buffer == nil) {
            buffer = [st.device newBufferWithLength:class_nbytes
                                            options:MTLResourceStorageModeShared];
            if (buffer == nil) {
                return _gd_error(GD_ERR_OUT_OF_MEMORY, "MTLBuffer allocation failed");
            }
        }
        memset(buffer.contents, 0, desc->nbytes);
        *handle_out = (__bridge_retained void *)buffer;
    }
    return GD_OK;
}

void _gd_metal_storage_free(_gd_backend *self, void *handle)
{
    if (handle != NULL) {
        id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)handle; /* ARC releases */
        GDMetalState *st = nil;

        if (self == NULL || self->impl == NULL) {
            return;
        }
        st = _gd_metal_state(self);
        if (st.bufferPoolMaxBytes == 0U || buffer.length > st.bufferPoolMaxBytes) {
            return;
        }
        @synchronized(st) {
            /* Only recycle buffers when no queued command buffer can still read
             * them. Otherwise let ARC release; Metal retains resources needed by
             * in-flight command buffers. */
            if (metal_buffer_pool_has_inflight(st)) {
                return;
            }
            metal_buffer_pool_trim_for_locked(st, buffer.length);
            if (buffer.length <= st.bufferPoolMaxBytes &&
                st.bufferPoolBytes <= st.bufferPoolMaxBytes - buffer.length) {
                NSNumber *key = @(buffer.length);
                NSMutableArray<id<MTLBuffer>> *bucket = st.bufferPool[key];
                if (bucket == nil) {
                    bucket = [NSMutableArray array];
                    st.bufferPool[key] = bucket;
                }
                [bucket addObject:buffer];
                st.bufferPoolBytes += buffer.length;
            }
        }
    }
}

gd_status _gd_metal_storage_host_ptr(_gd_backend *self, void *handle, void **ptr_out)
{
    (void)self;
    if (handle == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "metal storage handle is NULL");
    }
    {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
        *ptr_out = buffer.contents;
    }
    return GD_OK;
}

/* Persist any external-leaf mutations (optimizer in-place updates, gradient
 * slots written via emit_to) from their Metal buffers back to the owning CPU
 * tensor storage. CPU_REF gets this for free by borrowing leaf storage; Metal
 * stages copies, so it must copy back after the work completes. */
gd_status _gd_metal_writeback_externals(_gd_backend *self, _gd_executable *exe)
{
    int i = 0;
    uint64_t start = _gd_profile_enabled(self->ctx) ? _gd_profile_now_ns() : 0U;
    size_t bytes = 0U;
    uint64_t count = 0U;

    for (i = 0; i < exe->n_values; ++i) {
        gd_metal_value *v = &exe->values[i];
        gd_storage *dst = NULL;
        void *src = NULL;
        size_t nbytes = 0U;
        gd_status status = GD_OK;

        if (v->external == NULL || v->external_alias || !v->needs_writeback) {
            continue;
        }
        dst = gd_tensor_storage(v->external);
        if (dst == NULL) {
            continue;
        }
        nbytes = gd_storage_nbytes(v->storage);
        bytes += nbytes;
        count += 1U;
        status = gd_storage_data_cpu(v->storage, &src);
        if (status != GD_OK) {
            return status;
        }
        status = gd_storage_copy_from_cpu(self->ctx, dst, v->leaf_offset, src, nbytes);
        if (status != GD_OK) {
            return status;
        }
    }
    if (start != 0U) {
        _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_WRITEBACK,
                                 _gd_profile_now_ns() - start, bytes, count);
    }
    return GD_OK;
}

/* Writes back every executable with deferred CPU-backed external mutations and
 * clears the pending set. Called after the in-flight command buffer completes. */
gd_status _gd_metal_flush_pending_writebacks(_gd_backend *self)
{
    GDMetalState *st = _gd_metal_state(self);
    gd_status status = GD_OK;

    if (st.pendingExes.count == 0) {
        return GD_OK;
    }
    for (NSValue *boxed in st.pendingExes) {
        _gd_executable *exe = (_gd_executable *)boxed.pointerValue;
        gd_status s = _gd_metal_writeback_externals(self, exe);
        if (s != GD_OK && status == GD_OK) {
            status = s;
        }
    }
    [st.pendingExes removeAllObjects];
    return status;
}

/* Registers an executable's CPU-backed mutated externals for deferred writeback.
 * The host storages are marked pending so a later read resolves the flush; the
 * executable is added to the pending set (deduped) for the next synchronize. */
void _gd_metal_register_pending_writeback(_gd_backend *self, _gd_executable *exe)
{
    GDMetalState *st = _gd_metal_state(self);
    NSValue *boxed = [NSValue valueWithPointer:exe];
    int i = 0;

    if (![st.pendingExes containsObject:boxed]) {
        [st.pendingExes addObject:boxed];
    }
    for (i = 0; i < exe->n_values; ++i) {
        gd_metal_value *v = &exe->values[i];
        gd_storage *dst = NULL;

        if (v->external == NULL || v->external_alias || !v->needs_writeback) {
            continue;
        }
        dst = gd_tensor_storage(v->external);
        if (dst != NULL) {
            _gd_storage_set_pending_flush(dst, self, exe);
        }
    }
}

gd_status _gd_metal_synchronize(_gd_backend *self)
{
    GDMetalState *st = _gd_metal_state(self);
    id<MTLCommandBuffer> failed = nil;
    NSUInteger count = st.inFlightBuffers.count;

    if (count == 0U && st.inFlight != nil) {
        [st.inFlightBuffers addObject:st.inFlight];
        count = 1U;
    }
    if (count == 0U) {
        /* No GPU work in flight, but a prior sync may have left writebacks queued
         * (it never happens today, but keep the set authoritative). */
        return _gd_metal_flush_pending_writebacks(self);
    }
    {
        uint64_t start = _gd_profile_enabled(self->ctx) ? _gd_profile_now_ns() : 0U;
        for (id<MTLCommandBuffer> cmd in st.inFlightBuffers) {
            [cmd waitUntilCompleted];
            if (failed == nil && cmd.status == MTLCommandBufferStatusError) {
                failed = cmd;
            }
        }
        if (start != 0U) {
            _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_WAIT,
                                     _gd_profile_now_ns() - start, 0U, (uint64_t)count);
        }
    }
    st.inFlight = nil;
    [st.inFlightBuffers removeAllObjects];
    if (failed != nil) {
        const char *reason = failed.error != nil
                                 ? failed.error.localizedDescription.UTF8String
                                 : "command buffer failed";
        [st.pendingExes removeAllObjects];
        return _gd_error(GD_ERR_BACKEND, reason);
    }
    return _gd_metal_flush_pending_writebacks(self);
}

/* P4 flush hook: a host read hit a storage this backend marked pending. The
 * command buffer is serial, so syncing the latest in-flight buffer completes all
 * earlier ones; we then write back every pending executable. `cookie` is
 * advisory (the registering executable); we flush the whole pending set. */
gd_status _gd_metal_flush_pending(_gd_backend *self, void *cookie)
{
    (void)cookie;
    return _gd_metal_synchronize(self);
}

gd_status _gd_metal_upload(_gd_backend *self, void *dst_handle, size_t dst_off,
                              const void *src, size_t nbytes)
{
    (void)self;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)dst_handle;
    memcpy((unsigned char *)buffer.contents + dst_off, src, nbytes);
    return GD_OK;
}

gd_status _gd_metal_download(_gd_backend *self, void *src_handle, size_t src_off,
                                void *dst, size_t nbytes)
{
    /* Blocking read (P4): the CPU only observes GPU writes after completion. */
    gd_status status = _gd_metal_synchronize(self);
    if (status != GD_OK) {
        return status;
    }
    {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)src_handle;
        memcpy(dst, (const unsigned char *)buffer.contents + src_off, nbytes);
    }
    return GD_OK;
}
