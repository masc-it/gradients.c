#ifndef GD_METAL_BACKEND_INTERNAL_H
#define GD_METAL_BACKEND_INTERNAL_H

#import <Metal/Metal.h>

#include <stdbool.h>
#include <stddef.h>

#include "../../core/backend.h"

struct gd_backend {
    void *device;
    void *queue;
    void *fill_pso;
    void *rand_uniform_pso;
    void *linear_bias_pso;
    void *active_command_buffer;
    bool scope_active;
};

struct gd_backend_buffer {
    void *buffer;
    size_t nbytes;
};

gd_status gd_metal_command_for_op(gd_backend *backend,
                                  id<MTLCommandBuffer> *out_command_buffer,
                                  bool *out_immediate);
gd_status gd_metal_finish_immediate(id<MTLCommandBuffer> command_buffer, bool immediate);

#endif /* GD_METAL_BACKEND_INTERNAL_H */
