#include "gradients/optim.h"

#include <math.h>
#include <stdlib.h>

#include "gradients/ops.h"

#include "../core/internal.h"
#include "../core/tensor_internal.h"
#include "amp_internal.h"

struct gd_amp_scaler {
    gd_amp_scaler_config config;
    gd_tensor *scale;
    gd_tensor *found_inf;
    float current_scale;
    int growth_tracker;
};

static gd_status normalize_scaler_config(const gd_amp_scaler_config *in,
                                         gd_amp_scaler_config *out)
{
    gd_amp_scaler_config cfg = {0};

    if (in != NULL) {
        cfg = *in;
    }
    if (cfg.init_scale == 0.0F) { cfg.init_scale = 65536.0F; }
    if (cfg.growth_factor == 0.0F) { cfg.growth_factor = 2.0F; }
    if (cfg.backoff_factor == 0.0F) { cfg.backoff_factor = 0.5F; }
    if (cfg.growth_interval == 0) { cfg.growth_interval = 2000; }
    if (cfg.min_scale == 0.0F) { cfg.min_scale = 1.0F; }
    if (cfg.max_scale == 0.0F) { cfg.max_scale = 16777216.0F; }
    if (!isfinite(cfg.init_scale) || cfg.init_scale <= 0.0F ||
        !isfinite(cfg.growth_factor) || cfg.growth_factor <= 1.0F ||
        !isfinite(cfg.backoff_factor) || cfg.backoff_factor <= 0.0F ||
        cfg.backoff_factor >= 1.0F || cfg.growth_interval <= 0 ||
        !isfinite(cfg.min_scale) || cfg.min_scale <= 0.0F ||
        !isfinite(cfg.max_scale) || cfg.max_scale < cfg.min_scale) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid AMP scaler config");
    }
    if (cfg.init_scale < cfg.min_scale) { cfg.init_scale = cfg.min_scale; }
    if (cfg.init_scale > cfg.max_scale) { cfg.init_scale = cfg.max_scale; }
    *out = cfg;
    return GD_OK;
}

static gd_status make_scalar_tensor(gd_context *ctx, gd_dtype dtype, gd_tensor **out)
{
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(dtype, gd_context_default_device(ctx), 0,
                                                 NULL, &desc);
    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_empty(ctx, &desc, out);
}

gd_status gd_amp_scaler_create(gd_context *ctx,
                               const gd_amp_scaler_config *config,
                               gd_amp_scaler **out)
{
    gd_status status = GD_OK;
    gd_amp_scaler *scaler = NULL;
    int zero = 0;

    if (ctx == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_amp_scaler_create argument is NULL");
    }
    *out = NULL;
    scaler = calloc(1U, sizeof(*scaler));
    if (scaler == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate AMP scaler");
    }
    status = normalize_scaler_config(config, &scaler->config);
    if (status != GD_OK) {
        free(scaler);
        return status;
    }
    scaler->current_scale = scaler->config.init_scale;
    status = make_scalar_tensor(ctx, GD_DTYPE_F32, &scaler->scale);
    if (status == GD_OK) {
        status = make_scalar_tensor(ctx, GD_DTYPE_I32, &scaler->found_inf);
    }
    if (status == GD_OK) {
        status = gd_tensor_copy_from_cpu(ctx, scaler->scale, &scaler->current_scale,
                                         sizeof(scaler->current_scale));
    }
    if (status == GD_OK) {
        status = gd_tensor_copy_from_cpu(ctx, scaler->found_inf, &zero, sizeof(zero));
    }
    if (status != GD_OK) {
        gd_amp_scaler_destroy(scaler);
        return status;
    }
    *out = scaler;
    return GD_OK;
}

void gd_amp_scaler_destroy(gd_amp_scaler *scaler)
{
    if (scaler == NULL) {
        return;
    }
    gd_tensor_release(scaler->scale);
    gd_tensor_release(scaler->found_inf);
    free(scaler);
    _gd_set_last_error(GD_OK, NULL);
}

float gd_amp_scaler_scale(const gd_amp_scaler *scaler)
{
    return scaler != NULL ? scaler->current_scale : 0.0F;
}

gd_status _gd_amp_scaler_validate(gd_amp_scaler *scaler)
{
    if (scaler == NULL || scaler->scale == NULL || scaler->found_inf == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "AMP scaler is NULL");
    }
    return GD_OK;
}

gd_tensor *_gd_amp_scaler_scale_tensor(gd_amp_scaler *scaler)
{
    return scaler != NULL ? scaler->scale : NULL;
}

gd_tensor *_gd_amp_scaler_found_inf_tensor(gd_amp_scaler *scaler)
{
    return scaler != NULL ? scaler->found_inf : NULL;
}

gd_status gd_amp_scaler_scale_loss(gd_context *ctx,
                                   gd_amp_scaler *scaler,
                                   gd_tensor *loss,
                                   gd_tensor **scaled_loss)
{
    const gd_tensor_desc *desc = NULL;
    gd_status status = _gd_amp_scaler_validate(scaler);

    if (status != GD_OK) {
        return status;
    }
    if (ctx == NULL || loss == NULL || scaled_loss == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_amp_scaler_scale_loss argument is NULL");
    }
    desc = _gd_tensor_desc_ptr(loss);
    if (desc->dtype != GD_DTYPE_F32 || desc->ndim != 0) {
        return _gd_error(GD_ERR_DTYPE, "AMP scaler requires F32 scalar loss");
    }
    return gd_mul(ctx, loss, scaler->scale, scaled_loss);
}

gd_status gd_amp_scaler_found_inf(gd_context *ctx,
                                  gd_amp_scaler *scaler,
                                  bool *found_inf_out)
{
    int found = 0;
    gd_status status = _gd_amp_scaler_validate(scaler);

    if (status != GD_OK) {
        return status;
    }
    if (ctx == NULL || found_inf_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_amp_scaler_found_inf argument is NULL");
    }
    status = gd_tensor_copy_to_cpu(ctx, scaler->found_inf, &found, sizeof(found));
    if (status != GD_OK) {
        return status;
    }
    *found_inf_out = found != 0;
    return GD_OK;
}

gd_status gd_amp_scaler_update(gd_context *ctx,
                               gd_amp_scaler *scaler,
                               bool *stepped_out)
{
    bool found = false;
    int zero = 0;
    gd_status status = gd_amp_scaler_found_inf(ctx, scaler, &found);

    if (status != GD_OK) {
        return status;
    }
    if (found) {
        scaler->current_scale *= scaler->config.backoff_factor;
        if (scaler->current_scale < scaler->config.min_scale) {
            scaler->current_scale = scaler->config.min_scale;
        }
        scaler->growth_tracker = 0;
    } else {
        scaler->growth_tracker += 1;
        if (scaler->growth_tracker >= scaler->config.growth_interval) {
            scaler->current_scale *= scaler->config.growth_factor;
            if (scaler->current_scale > scaler->config.max_scale) {
                scaler->current_scale = scaler->config.max_scale;
            }
            scaler->growth_tracker = 0;
        }
    }
    status = gd_tensor_copy_from_cpu(ctx, scaler->scale, &scaler->current_scale,
                                     sizeof(scaler->current_scale));
    if (status == GD_OK) {
        status = gd_tensor_copy_from_cpu(ctx, scaler->found_inf, &zero, sizeof(zero));
    }
    if (status != GD_OK) {
        return status;
    }
    if (stepped_out != NULL) {
        *stepped_out = !found;
    }
    return GD_OK;
}
