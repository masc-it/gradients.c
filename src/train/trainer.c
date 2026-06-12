#include <gradients/trainer.h>

#include <gradients/autograd.h>

#include "../core/memory_internal.h"

#include <string.h>

static gd_status gd_train_batch_error(gd_context *ctx, gd_status st, const char *message)
{
    if (st != GD_OK && ctx != NULL && gd_context_status(ctx) == GD_OK) {
        return gd_context_set_error(ctx, st, message);
    }
    return st;
}

static int gd_train_batch_grad_clip_enabled(float grad_clip_norm)
{
    return grad_clip_norm > 0.0f;
}

static int gd_train_batch_grad_clip_invalid(float grad_clip_norm)
{
    return grad_clip_norm != grad_clip_norm || grad_clip_norm < 0.0f;
}

static gd_status gd_train_batch_optimizer_step(gd_context *ctx,
                                               const gd_train_batch_config *config,
                                               bool has_lr,
                                               float lr)
{
    const int clip = gd_train_batch_grad_clip_enabled(config->grad_clip_norm);
    if (config->scaler != NULL) {
        if (clip != 0) {
            if (has_lr) {
                return gd_optimizer_step_amp_clip_lr(ctx,
                                                     config->optimizer,
                                                     config->scaler,
                                                     lr,
                                                     config->grad_clip_norm);
            }
            return gd_optimizer_step_amp_clip(ctx,
                                              config->optimizer,
                                              config->scaler,
                                              config->grad_clip_norm);
        }
        if (has_lr) {
            return gd_optimizer_step_amp_lr(ctx, config->optimizer, config->scaler, lr);
        }
        return gd_optimizer_step_amp(ctx, config->optimizer, config->scaler);
    }
    if (clip != 0) {
        if (has_lr) {
            return gd_optimizer_step_clip_lr(ctx, config->optimizer, lr, config->grad_clip_norm);
        }
        return gd_optimizer_step_clip(ctx, config->optimizer, config->grad_clip_norm);
    }
    if (has_lr) {
        return gd_optimizer_step_lr(ctx, config->optimizer, lr);
    }
    return gd_optimizer_step(ctx, config->optimizer);
}

gd_train_batch_config gd_train_batch_config_default(gd_dataloader *loader,
                                                    gd_optimizer *optimizer)
{
    gd_train_batch_config config;
    memset(&config, 0, sizeof(config));
    config.loader = loader;
    config.optimizer = optimizer;
    config.prefetch_next = true;
    return config;
}

gd_status gd_train_batch(gd_context *ctx,
                         const gd_train_batch_config *config,
                         gd_train_batch_loss_fn loss_fn,
                         void *user_data,
                         gd_train_batch_result *out)
{
    gd_train_batch_result result;
    gd_train_batch_step step;
    gd_batch *batch = NULL;
    gd_tensor loss;
    gd_status st;
    bool has_lr = false;
    bool in_step = false;
    float lr = 0.0f;

    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    memset(&result, 0, sizeof(result));
    memset(&step, 0, sizeof(step));
    memset(&loss, 0, sizeof(loss));

    if (ctx == NULL || config == NULL || config->loader == NULL ||
        config->optimizer == NULL || loss_fn == NULL) {
        return gd_train_batch_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid train batch config");
    }
    if (gd_train_batch_grad_clip_invalid(config->grad_clip_norm) != 0) {
        return gd_train_batch_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid train batch grad clip norm");
    }
    if (config->lr_schedule != NULL && config->use_fixed_lr) {
        return gd_train_batch_error(ctx, GD_ERR_INVALID_ARGUMENT, "train batch lr configured twice");
    }

    if (config->lr_schedule != NULL) {
        st = gd_lr_scheduler_value(config->lr_schedule, config->global_step, &lr);
        if (st != GD_OK) {
            return gd_train_batch_error(ctx, st, "train batch lr schedule failed");
        }
        has_lr = true;
    } else if (config->use_fixed_lr) {
        lr = config->fixed_lr;
        has_lr = true;
    }
    result.lr = lr;
    result.has_lr = has_lr;
    if (config->scaler != NULL) {
        result.amp_scale = gd_amp_scaler_scale(config->scaler);
        result.found_inf = gd_amp_scaler_last_found_inf(config->scaler);
        result.has_amp_state = true;
    }

    st = gd_dataloader_next(config->loader, &batch);
    if (st != GD_OK) {
        return gd_train_batch_error(ctx, st, "train batch dataloader next failed");
    }

    st = gd_begin_step(ctx, GD_SCOPE_TRAIN, batch);
    if (st != GD_OK) {
        (void)gd_dataloader_release(config->loader, batch);
        return gd_train_batch_error(ctx, st, "train batch begin step failed");
    }
    in_step = true;

    step.ctx = ctx;
    step.batch = batch;
    step.global_step = config->global_step;
    step.lr = lr;
    step.has_lr = has_lr;

    st = loss_fn(&step, user_data, &loss);
    if (st != GD_OK) {
        st = gd_train_batch_error(ctx, st, "train batch loss callback failed");
    }
    if (st == GD_OK) {
        if (config->scaler != NULL) {
            st = gd_backward_amp(ctx, &loss, NULL, config->scaler);
        } else {
            st = gd_backward(ctx, &loss, NULL);
        }
        if (st != GD_OK) {
            st = gd_train_batch_error(ctx, st, "train batch backward failed");
        }
    }
    if (st == GD_OK) {
        st = gd_train_batch_optimizer_step(ctx, config, has_lr, lr);
        if (st != GD_OK) {
            st = gd_train_batch_error(ctx, st, "train batch optimizer step failed");
        }
    }

    if (st != GD_OK) {
        const gd_status original = st;
        if (in_step) {
            (void)gd_end_step(ctx);
            in_step = false;
        }
        (void)gd_dataloader_release(config->loader, batch);
        return original;
    }

    st = gd_end_step(ctx);
    in_step = false;
    if (st != GD_OK) {
        return gd_train_batch_error(ctx, st, "train batch end step failed");
    }

    st = gd_dataloader_release(config->loader, batch);
    if (st != GD_OK) {
        return gd_train_batch_error(ctx, st, "train batch dataloader release failed");
    }

    if (config->prefetch_next) {
        st = gd_dataloader_prefetch(config->loader);
        if (st != GD_OK) {
            return gd_train_batch_error(ctx, st, "train batch dataloader prefetch failed");
        }
    }

    if (config->read_loss_value) {
        st = gd_tensor_item(ctx, &loss, &result.loss_value);
        if (st != GD_OK) {
            return gd_train_batch_error(ctx, st, "train batch loss read failed");
        }
        result.has_loss_value = true;
    }

    if (config->sync_scaler && config->scaler != NULL) {
        st = gd_amp_scaler_sync(ctx, config->scaler);
        if (st != GD_OK) {
            return gd_train_batch_error(ctx, st, "train batch amp scaler sync failed");
        }
        result.amp_scale = gd_amp_scaler_scale(config->scaler);
        result.found_inf = gd_amp_scaler_last_found_inf(config->scaler);
        result.has_amp_state = true;
    }

    if (config->read_grad_norm) {
        st = gd_optimizer_last_grad_norm(ctx, config->optimizer, &result.grad_norm);
        if (st != GD_OK) {
            return gd_train_batch_error(ctx, st, "train batch grad norm read failed");
        }
        result.has_grad_norm = true;
    }

    if (out != NULL) {
        *out = result;
    }
    return GD_OK;
}
