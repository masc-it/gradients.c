#ifndef GRADIENTS_TRAINER_H
#define GRADIENTS_TRAINER_H

#include <stdbool.h>
#include <stdint.h>

#include <gradients/dataloader.h>
#include <gradients/module.h>
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

/* Mean-loss evaluation helper over a dataset. It owns eval dataloader creation,
 * scope begin/end, loss readback, batch release/prefetch, and optional module
 * training-mode restore. */
typedef struct gd_eval_batch_step {
    gd_context *ctx;
    gd_batch *batch;
    uint64_t step_index;
} gd_eval_batch_step;

typedef gd_status (*gd_eval_loss_fn)(const gd_eval_batch_step *step,
                                     void *user_data,
                                     gd_tensor *loss_out);

typedef struct gd_eval_config {
    gd_dataset *dataset;
    gd_module *module; /* Optional: set eval mode for the run, then restore. */
    int batch_size;
    int num_workers;     /* 0 => dataloader default. */
    int prefetch_factor; /* 0 => dataloader default. */
    gd_scope_mode scope; /* GD_SCOPE_NONE => GD_SCOPE_EVAL. */
    uint64_t max_steps;  /* 0 => all full batches. */
    bool prefetch_next;
} gd_eval_config;

gd_eval_config gd_eval_config_default(gd_dataset *dataset, int batch_size);

gd_status gd_eval_mean_loss(gd_context *ctx,
                            const gd_eval_config *config,
                            gd_eval_loss_fn loss_fn,
                            void *user_data,
                            float *mean_loss_out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_TRAINER_H */
