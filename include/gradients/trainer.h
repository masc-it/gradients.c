#ifndef GRADIENTS_TRAINER_H
#define GRADIENTS_TRAINER_H

#include <stdbool.h>
#include <stdint.h>

#include <gradients/dataloader.h>
#include <gradients/optimizer.h>
#include <gradients/status.h>
#include <gradients/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Minimal single-batch training transaction helper. It owns the fragile
 * dataloader/scope/backward/optimizer/release/prefetch ordering while the caller
 * still owns epochs, reporting policy, validation, checkpointing, and custom
 * model forward code. */
typedef struct gd_train_batch_step {
    gd_context *ctx;
    gd_batch *batch;
    uint64_t global_step;
    float lr;
    bool has_lr;
} gd_train_batch_step;

typedef gd_status (*gd_train_batch_loss_fn)(const gd_train_batch_step *step,
                                            void *user_data,
                                            gd_tensor *loss_out);

typedef struct gd_train_batch_config {
    gd_dataloader *loader;
    gd_optimizer *optimizer;
    gd_amp_scaler *scaler; /* Optional. NULL disables AMP. */
    const gd_lr_scheduler_config *lr_schedule; /* Optional LR override. */
    uint64_t global_step;
    float fixed_lr;
    bool use_fixed_lr;
    float grad_clip_norm; /* <= 0 disables clipping. */
    bool prefetch_next;
    bool read_loss_value;
    bool read_grad_norm;
    bool sync_scaler;
} gd_train_batch_config;

typedef struct gd_train_batch_result {
    float lr;
    bool has_lr;
    float loss_value;
    bool has_loss_value;
    float grad_norm;
    bool has_grad_norm;
    float amp_scale;
    bool found_inf;
    bool has_amp_state;
} gd_train_batch_result;

gd_train_batch_config gd_train_batch_config_default(gd_dataloader *loader,
                                                    gd_optimizer *optimizer);

gd_status gd_train_batch(gd_context *ctx,
                         const gd_train_batch_config *config,
                         gd_train_batch_loss_fn loss_fn,
                         void *user_data,
                         gd_train_batch_result *out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_TRAINER_H */
