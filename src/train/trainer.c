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

gd_eval_config gd_eval_config_default(gd_dataset *dataset, int batch_size)
{
    gd_eval_config config;
    memset(&config, 0, sizeof(config));
    config.dataset = dataset;
    config.batch_size = batch_size;
    config.scope = GD_SCOPE_EVAL;
    config.prefetch_next = true;
    return config;
}

gd_status gd_eval_mean_loss(gd_context *ctx,
                            const gd_eval_config *config,
                            gd_eval_loss_fn loss_fn,
                            void *user_data,
                            float *mean_loss_out)
{
    gd_dataloader *loader = NULL;
    gd_batch *batch = NULL;
    gd_scope_mode scope;
    gd_dataloader_config loader_config;
    gd_status st = GD_OK;
    uint64_t samples;
    uint64_t steps;
    uint64_t step_index;
    double total_loss = 0.0;
    int batch_size;
    bool in_step = false;
    bool restore_training = false;
    bool was_training = false;

    if (mean_loss_out != NULL) {
        *mean_loss_out = 0.0f;
    }
    if (ctx == NULL || config == NULL || config->dataset == NULL ||
        config->batch_size <= 0 || config->num_workers < 0 || config->prefetch_factor < 0 ||
        loss_fn == NULL || mean_loss_out == NULL) {
        return gd_train_batch_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid eval config");
    }
    scope = config->scope == GD_SCOPE_NONE ? GD_SCOPE_EVAL : config->scope;
    if (scope != GD_SCOPE_EVAL && scope != GD_SCOPE_INFER) {
        return gd_train_batch_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid eval scope");
    }
    samples = gd_dataset_num_samples(config->dataset);
    if (samples == 0U) {
        return gd_train_batch_error(ctx, GD_ERR_INVALID_ARGUMENT, "empty eval dataset");
    }
    batch_size = config->batch_size;
    if ((uint64_t)batch_size > samples) {
        batch_size = (int)samples;
    }
    if (batch_size <= 0) {
        return gd_train_batch_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid eval batch size");
    }
    steps = samples / (uint64_t)batch_size;
    if (config->max_steps > 0U && config->max_steps < steps) {
        steps = config->max_steps;
    }
    if (steps == 0U) {
        return gd_train_batch_error(ctx, GD_ERR_INVALID_ARGUMENT, "eval dataset has no full batches");
    }

    if (config->module != NULL) {
        was_training = config->module->training;
        gd_module_set_training(config->module, false);
        restore_training = true;
    }

    loader_config = gd_dataloader_config_default(batch_size);
    if (config->num_workers > 0) {
        loader_config.num_workers = config->num_workers;
    }
    if (config->prefetch_factor > 0) {
        loader_config.prefetch_factor = config->prefetch_factor;
    }
    st = gd_dataloader_create(ctx, config->dataset, NULL, &loader_config, &loader);
    if (st != GD_OK) {
        st = gd_train_batch_error(ctx, st, "eval dataloader create failed");
        goto done;
    }
    st = gd_dataloader_prefetch(loader);
    if (st != GD_OK) {
        st = gd_train_batch_error(ctx, st, "eval dataloader prefetch failed");
        goto done;
    }

    for (step_index = 0U; step_index < steps; ++step_index) {
        gd_eval_batch_step step;
        gd_tensor loss;
        float loss_value = 0.0f;
        memset(&step, 0, sizeof(step));
        memset(&loss, 0, sizeof(loss));

        st = gd_dataloader_next(loader, &batch);
        if (st != GD_OK) {
            st = gd_train_batch_error(ctx, st, "eval dataloader next failed");
            goto done;
        }
        st = gd_begin_step(ctx, scope, batch);
        if (st != GD_OK) {
            st = gd_train_batch_error(ctx, st, "eval begin step failed");
            goto done;
        }
        in_step = true;

        step.ctx = ctx;
        step.batch = batch;
        step.step_index = step_index;
        st = loss_fn(&step, user_data, &loss);
        if (st != GD_OK) {
            st = gd_train_batch_error(ctx, st, "eval loss callback failed");
            goto done;
        }
        st = gd_tensor_item(ctx, &loss, &loss_value);
        if (st != GD_OK) {
            st = gd_train_batch_error(ctx, st, "eval loss read failed");
            goto done;
        }
        st = gd_end_step(ctx);
        in_step = false;
        if (st != GD_OK) {
            st = gd_train_batch_error(ctx, st, "eval end step failed");
            goto done;
        }
        st = gd_dataloader_release(loader, batch);
        batch = NULL;
        if (st != GD_OK) {
            st = gd_train_batch_error(ctx, st, "eval dataloader release failed");
            goto done;
        }
        if (config->prefetch_next && step_index + 1U < steps) {
            st = gd_dataloader_prefetch(loader);
            if (st != GD_OK) {
                st = gd_train_batch_error(ctx, st, "eval dataloader prefetch failed");
                goto done;
            }
        }
        total_loss += (double)loss_value;
    }

    *mean_loss_out = (float)(total_loss / (double)steps);

done:
    if (in_step) {
        (void)gd_end_step(ctx);
    }
    if (batch != NULL && loader != NULL) {
        (void)gd_dataloader_release(loader, batch);
    }
    gd_dataloader_destroy(loader);
    if (restore_training) {
        gd_module_set_training(config->module, was_training);
    }
    return st;
}
