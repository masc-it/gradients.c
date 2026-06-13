#include <gradients/optimizer.h>

#include <math.h>

static int gd_lr_finite_nonnegative(float value)
{
    return isfinite(value) && value >= 0.0f;
}

gd_lr_scheduler_config gd_lr_scheduler_config_default(void)
{
    gd_lr_scheduler_config config;
    config.max_lr = 1.0e-3f;
    config.min_lr = 1.0e-4f;
    config.warmup_steps = 0U;
    config.total_steps = 1U;
    return config;
}

gd_status gd_lr_scheduler_value(const gd_lr_scheduler_config *config,
                                uint64_t step,
                                float *lr_out)
{
    static const double pi = 3.14159265358979323846264338327950288;
    double lr;
    if (config == NULL || lr_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_lr_finite_nonnegative(config->max_lr) ||
        !gd_lr_finite_nonnegative(config->min_lr) ||
        config->min_lr > config->max_lr) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (config->total_steps == 0U || config->warmup_steps > config->total_steps) {
        return GD_ERR_INVALID_ARGUMENT;
    }

    if (config->warmup_steps > 0U && step < config->warmup_steps) {
        lr = (double)config->max_lr * (double)step / (double)config->warmup_steps;
    } else if (config->total_steps == config->warmup_steps) {
        lr = (double)config->min_lr;
    } else {
        const uint64_t decay_steps = config->total_steps - config->warmup_steps;
        const uint64_t elapsed = step > config->warmup_steps ?
                                     step - config->warmup_steps :
                                     0U;
        double progress = (double)elapsed / (double)decay_steps;
        if (progress > 1.0) {
            progress = 1.0;
        }
        lr = (double)config->min_lr +
             0.5 * ((double)config->max_lr - (double)config->min_lr) *
                 (1.0 + cos(pi * progress));
    }
    *lr_out = (float)lr;
    return GD_OK;
}
