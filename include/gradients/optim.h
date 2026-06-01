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

typedef enum gd_master_param_policy {
    GD_MASTER_PARAM_AUTO = 0,
    GD_MASTER_PARAM_DISABLED = 1,
    GD_MASTER_PARAM_ALWAYS = 2
} gd_master_param_policy;

typedef struct gd_adamw_config {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    gd_dtype state_dtype;
    gd_master_param_policy master_param_policy;
    int use_state_device;
    gd_device state_device;
} gd_adamw_config;

typedef struct gd_lr_scheduler_config {
    float max_lr;
    float min_lr;
    int warmup_steps;
    int total_steps;
} gd_lr_scheduler_config;

typedef struct gd_param_group {
    gd_tensor **params;
    int n_params;
    float weight_decay;
    float lr_scale;
} gd_param_group;

gd_status gd_adamw_create(gd_context *ctx,
                          gd_tensor **params,
                          int n_params,
                          const gd_adamw_config *config,
                          gd_optimizer **out);
gd_status gd_adamw_create_groups(gd_context *ctx,
                                 const gd_param_group *groups,
                                 int n_groups,
                                 const gd_adamw_config *config,
                                 gd_optimizer **out);
void gd_optimizer_destroy(gd_optimizer *optimizer);

gd_status gd_optimizer_step(gd_context *ctx, gd_optimizer *optimizer);
gd_status gd_optimizer_step_lr(gd_context *ctx,
                               gd_optimizer *optimizer,
                               gd_tensor *lr_scalar);
gd_status gd_optimizer_zero_grad(gd_context *ctx, gd_optimizer *optimizer);
gd_status gd_optimizer_save(gd_optimizer *optimizer, const char *path);
gd_status gd_optimizer_load(gd_optimizer *optimizer, const char *path);

gd_status gd_lr_scheduler_value(const gd_lr_scheduler_config *config,
                                int step,
                                float *lr_out);
gd_status gd_lr_scheduler_write(gd_context *ctx,
                                const gd_lr_scheduler_config *config,
                                int step,
                                gd_tensor *lr_scalar,
                                float *lr_out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_OPTIM_H */
