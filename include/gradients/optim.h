#ifndef GRADIENTS_OPTIM_H
#define GRADIENTS_OPTIM_H

#include "gradients/context.h"
#include "gradients/dtype.h"
#include "gradients/device.h"
#include "gradients/status.h"
#include "gradients/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_optimizer gd_optimizer;

typedef struct gd_adamw_config {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    gd_dtype state_dtype;
    int use_state_device;
    gd_device state_device;
} gd_adamw_config;

gd_status gd_adamw_create(gd_context *ctx,
                          gd_tensor **params,
                          int n_params,
                          const gd_adamw_config *config,
                          gd_optimizer **out);
void gd_optimizer_destroy(gd_optimizer *optimizer);

gd_status gd_optimizer_step(gd_context *ctx, gd_optimizer *optimizer);
gd_status gd_optimizer_zero_grad(gd_context *ctx, gd_optimizer *optimizer);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_OPTIM_H */
