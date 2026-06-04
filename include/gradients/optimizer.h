#ifndef GRADIENTS_OPTIMIZER_H
#define GRADIENTS_OPTIMIZER_H

#include <stdbool.h>
#include <stdint.h>

#include <gradients/module.h>
#include <gradients/status.h>

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
    bool bias_correction;
} gd_adamw_config;

gd_adamw_config gd_adamw_config_default(void);

gd_status gd_adamw_create(gd_context *ctx,
                          const gd_param_set *params,
                          const gd_adamw_config *config,
                          gd_optimizer **out);
void gd_optimizer_destroy(gd_optimizer *optimizer);

gd_status gd_optimizer_zero_grad(gd_context *ctx, gd_optimizer *optimizer);
gd_status gd_optimizer_step(gd_context *ctx, gd_optimizer *optimizer);
uint64_t gd_optimizer_step_count(const gd_optimizer *optimizer);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_OPTIMIZER_H */
