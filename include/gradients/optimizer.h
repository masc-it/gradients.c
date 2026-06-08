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
typedef struct gd_amp_scaler gd_amp_scaler;

typedef struct gd_adamw_config {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    bool bias_correction;
} gd_adamw_config;

typedef struct gd_lr_scheduler_config {
    float max_lr;
    float min_lr;
    uint64_t warmup_steps;
    uint64_t total_steps;
} gd_lr_scheduler_config;

typedef struct gd_amp_config {
    bool enabled;
    /* When true, optimizer steps keep found-inf skip logic on the backend and
       optimistically update CPU-side scaler state without a mid-step readback.
       This removes a per-step synchronization for workloads that expect finite
       gradients, at the cost of delayed CPU visibility for rare overflows. */
    bool defer_found_inf;
    float init_scale;
    float growth_factor;
    float backoff_factor;
    float min_scale;
    float max_scale;
    uint32_t growth_interval;
} gd_amp_config;

gd_adamw_config gd_adamw_config_default(void);
gd_lr_scheduler_config gd_lr_scheduler_config_default(void);
gd_amp_config gd_amp_config_default(void);

gd_status gd_lr_scheduler_value(const gd_lr_scheduler_config *config,
                                uint64_t step,
                                float *lr_out);

gd_status gd_amp_scaler_create(const gd_amp_config *config, gd_amp_scaler **out);
void gd_amp_scaler_destroy(gd_amp_scaler *scaler);
float gd_amp_scaler_scale(const gd_amp_scaler *scaler);
bool gd_amp_scaler_enabled(const gd_amp_scaler *scaler);
bool gd_amp_scaler_last_found_inf(const gd_amp_scaler *scaler);
uint32_t gd_amp_scaler_growth_tracker(const gd_amp_scaler *scaler);

gd_status gd_adamw_create(gd_context *ctx,
                          const gd_param_set *params,
                          const gd_adamw_config *config,
                          gd_optimizer **out);
void gd_optimizer_destroy(gd_optimizer *optimizer);

gd_status gd_optimizer_zero_grad(gd_context *ctx, gd_optimizer *optimizer);
gd_status gd_optimizer_step(gd_context *ctx, gd_optimizer *optimizer);
gd_status gd_optimizer_step_lr(gd_context *ctx, gd_optimizer *optimizer, float lr);
gd_status gd_optimizer_step_clip(gd_context *ctx,
                                 gd_optimizer *optimizer,
                                 float max_grad_norm);
gd_status gd_optimizer_step_clip_lr(gd_context *ctx,
                                    gd_optimizer *optimizer,
                                    float lr,
                                    float max_grad_norm);
gd_status gd_optimizer_step_amp(gd_context *ctx, gd_optimizer *optimizer, gd_amp_scaler *scaler);
gd_status gd_optimizer_step_amp_lr(gd_context *ctx,
                                   gd_optimizer *optimizer,
                                   gd_amp_scaler *scaler,
                                   float lr);
gd_status gd_optimizer_step_amp_clip(gd_context *ctx,
                                     gd_optimizer *optimizer,
                                     gd_amp_scaler *scaler,
                                     float max_grad_norm);
gd_status gd_optimizer_step_amp_clip_lr(gd_context *ctx,
                                        gd_optimizer *optimizer,
                                        gd_amp_scaler *scaler,
                                        float lr,
                                        float max_grad_norm);
gd_status gd_optimizer_last_grad_norm(gd_context *ctx,
                                      const gd_optimizer *optimizer,
                                      float *out);
uint64_t gd_optimizer_step_count(const gd_optimizer *optimizer);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_OPTIMIZER_H */
