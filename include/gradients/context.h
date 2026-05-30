#ifndef GRADIENTS_CONTEXT_H
#define GRADIENTS_CONTEXT_H

#include "gradients/device.h"
#include "gradients/dtype.h"
#include "gradients/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_context gd_context;

typedef enum gd_fallback_policy {
    GD_FALLBACK_NONE = 0,
    GD_FALLBACK_CPU_REF
} gd_fallback_policy;

gd_status gd_context_create(gd_context **out);
void gd_context_destroy(gd_context *ctx);

gd_status gd_context_set_default_device(gd_context *ctx, gd_device device);
gd_device gd_context_default_device(const gd_context *ctx);

gd_status gd_context_set_fallback_policy(gd_context *ctx,
                                         gd_fallback_policy policy);
gd_fallback_policy gd_context_fallback_policy(const gd_context *ctx);

gd_status gd_context_set_compute_policy(gd_context *ctx,
                                        gd_compute_policy policy);
gd_compute_policy gd_context_compute_policy(const gd_context *ctx);

gd_status gd_synchronize(gd_context *ctx, gd_device device);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_CONTEXT_H */
