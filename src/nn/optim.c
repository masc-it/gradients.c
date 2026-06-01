#include "gradients/optim.h"

#include <math.h>
#include <stdlib.h>

#include "gradients/ops.h"

#include "../core/internal.h"
#include "../core/tensor_internal.h"
#include "../graph/graph_internal.h"

typedef struct adamw_slot {
    gd_tensor *param;  /* retained */
    gd_tensor *m;      /* owned */
    gd_tensor *v;      /* owned */
} adamw_slot;

struct gd_optimizer {
    gd_context *ctx;
    gd_adamw_config config;
    adamw_slot *slots;
    int n_slots;
    gd_tensor *step;   /* owned F32 scalar, shared across params */
};

static int same_parameter(gd_tensor *a, gd_tensor *b)
{
    const gd_tensor_desc *da = _gd_tensor_desc_ptr(a);
    const gd_tensor_desc *db = _gd_tensor_desc_ptr(b);
    size_t na = 0U;
    size_t nb = 0U;

    if (gd_tensor_storage(a) != gd_tensor_storage(b)) {
        return 0;
    }
    if (da->storage_offset_bytes != db->storage_offset_bytes) {
        return 0;
    }
    if (gd_tensor_desc_nbytes(da, &na, NULL) != GD_OK ||
        gd_tensor_desc_nbytes(db, &nb, NULL) != GD_OK) {
        return 0;
    }
    return na == nb;
}

static gd_status make_state_like(gd_context *ctx, gd_tensor *param, gd_tensor **out)
{
    const gd_tensor_desc *pd = _gd_tensor_desc_ptr(param);
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, pd->device, pd->ndim,
                                                 pd->sizes, &desc);
    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_empty(ctx, &desc, out);
}

static gd_status validate_config(const gd_adamw_config *cfg)
{
    if (!isfinite(cfg->lr) || cfg->lr < 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "adamw lr must be finite and nonnegative");
    }
    if (!isfinite(cfg->beta1) || !isfinite(cfg->beta2) ||
        cfg->beta1 <= 0.0F || cfg->beta1 >= 1.0F ||
        cfg->beta2 <= 0.0F || cfg->beta2 >= 1.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "adamw betas must be finite and in (0,1)");
    }
    if (!isfinite(cfg->eps) || cfg->eps <= 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "adamw eps must be finite and positive");
    }
    if (!isfinite(cfg->weight_decay) || cfg->weight_decay < 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "adamw weight_decay must be finite and nonnegative");
    }
    if (cfg->state_dtype != GD_DTYPE_INVALID && cfg->state_dtype != GD_DTYPE_F32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "adamw v1 supports F32 optimizer state only");
    }
    return GD_OK;
}

gd_status gd_adamw_create(gd_context *ctx,
                          gd_tensor **params,
                          int n_params,
                          const gd_adamw_config *config,
                          gd_optimizer **out)
{
    gd_status status = GD_OK;
    gd_optimizer *opt = NULL;
    gd_tensor_desc step_desc;
    int i = 0;

    if (ctx == NULL || (params == NULL && n_params > 0) || n_params < 0 ||
        config == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_adamw_create argument is invalid");
    }
    *out = NULL;
    status = validate_config(config);
    if (status != GD_OK) {
        return status;
    }

    opt = calloc(1U, sizeof(*opt));
    if (opt == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate optimizer");
    }
    opt->ctx = ctx;
    opt->config = *config;
    if (opt->config.state_dtype == GD_DTYPE_INVALID) {
        opt->config.state_dtype = GD_DTYPE_F32;
    }
    if (n_params > 0) {
        opt->slots = calloc((size_t)n_params, sizeof(*opt->slots));
        if (opt->slots == NULL) {
            free(opt);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate optimizer slots");
        }
    }

    for (i = 0; i < n_params; ++i) {
        gd_tensor *p = params[i];
        int duplicate = 0;
        int j = 0;

        if (p == NULL) {
            status = _gd_error(GD_ERR_INVALID_ARGUMENT, "optimizer parameter is NULL");
            goto fail;
        }
        if (gd_tensor_dtype(p) != GD_DTYPE_F32) {
            status = _gd_error(GD_ERR_UNSUPPORTED, "adamw v1 supports F32 parameters only");
            goto fail;
        }
        if (!gd_tensor_requires_grad(p)) {
            status = _gd_error(GD_ERR_INVALID_ARGUMENT, "optimizer parameter must require grad");
            goto fail;
        }
        for (j = 0; j < opt->n_slots; ++j) {
            if (same_parameter(opt->slots[j].param, p)) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) {
            continue; /* tied weights share a single optimizer state */
        }

        status = gd_tensor_retain(p);
        if (status != GD_OK) {
            goto fail;
        }
        opt->slots[opt->n_slots].param = p;
        status = make_state_like(ctx, p, &opt->slots[opt->n_slots].m);
        if (status != GD_OK) {
            gd_tensor_release(p);
            opt->slots[opt->n_slots].param = NULL;
            goto fail;
        }
        status = make_state_like(ctx, p, &opt->slots[opt->n_slots].v);
        if (status != GD_OK) {
            gd_tensor_release(p);
            gd_tensor_release(opt->slots[opt->n_slots].m);
            opt->slots[opt->n_slots].param = NULL;
            opt->slots[opt->n_slots].m = NULL;
            goto fail;
        }
        opt->n_slots += 1;
    }

    status = gd_tensor_desc_contiguous(GD_DTYPE_F32, gd_context_default_device(ctx), 0, NULL,
                                       &step_desc);
    if (status != GD_OK) {
        goto fail;
    }
    status = gd_tensor_empty(ctx, &step_desc, &opt->step);
    if (status != GD_OK) {
        goto fail;
    }

    *out = opt;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;

fail:
    gd_optimizer_destroy(opt);
    return status;
}

void gd_optimizer_destroy(gd_optimizer *optimizer)
{
    int i = 0;

    if (optimizer == NULL) {
        return;
    }
    for (i = 0; i < optimizer->n_slots; ++i) {
        gd_tensor_release(optimizer->slots[i].param);
        gd_tensor_release(optimizer->slots[i].m);
        gd_tensor_release(optimizer->slots[i].v);
    }
    free(optimizer->slots);
    gd_tensor_release(optimizer->step);
    free(optimizer);
    _gd_set_last_error(GD_OK, NULL);
}

static gd_status validate_lr_tensor(gd_tensor *lr_scalar)
{
    const gd_tensor_desc *desc = NULL;
    if (lr_scalar == NULL) {
        return GD_OK;
    }
    if (gd_tensor_dtype(lr_scalar) != GD_DTYPE_F32) {
        return _gd_error(GD_ERR_DTYPE, "adamw lr tensor must be F32");
    }
    desc = _gd_tensor_desc_ptr(lr_scalar);
    if (desc->ndim != 0) {
        return _gd_error(GD_ERR_SHAPE, "adamw lr tensor must be scalar");
    }
    return GD_OK;
}

static gd_status optimizer_step_impl(gd_context *ctx,
                                     gd_optimizer *optimizer,
                                     gd_tensor *lr_scalar)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    _gd_op_attrs attrs = {0};
    int i = 0;

    if (ctx == NULL || optimizer == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_optimizer_step argument is NULL");
    }
    status = validate_lr_tensor(lr_scalar);
    if (status != GD_OK) {
        return status;
    }
    graph = _gd_context_active_graph(ctx);
    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "gd_optimizer_step requires an active graph");
    }

    status = _gd_graph_emit_inplace(graph, _GD_OP_STEP_INC, &optimizer->step, 1, NULL);
    if (status != GD_OK) {
        return status;
    }

    attrs.lr = optimizer->config.lr;
    attrs.beta1 = optimizer->config.beta1;
    attrs.beta2 = optimizer->config.beta2;
    attrs.eps = optimizer->config.eps;
    attrs.weight_decay = optimizer->config.weight_decay;

    for (i = 0; i < optimizer->n_slots; ++i) {
        gd_tensor *grad = NULL;
        gd_tensor *inputs[6];
        int n_inputs = lr_scalar != NULL ? 6 : 5;

        status = _gd_tensor_ensure_grad(ctx, optimizer->slots[i].param, &grad);
        if (status != GD_OK) {
            return status;
        }
        inputs[0] = optimizer->slots[i].param;
        inputs[1] = grad;
        inputs[2] = optimizer->slots[i].m;
        inputs[3] = optimizer->slots[i].v;
        inputs[4] = optimizer->step;
        inputs[5] = lr_scalar;
        status = _gd_graph_emit_inplace(graph, _GD_OP_ADAMW_STEP, inputs, n_inputs, &attrs);
        if (status != GD_OK) {
            return status;
        }
    }

    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_optimizer_step(gd_context *ctx, gd_optimizer *optimizer)
{
    return optimizer_step_impl(ctx, optimizer, NULL);
}

gd_status gd_optimizer_step_lr(gd_context *ctx,
                               gd_optimizer *optimizer,
                               gd_tensor *lr_scalar)
{
    if (lr_scalar == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_optimizer_step_lr lr_scalar is NULL");
    }
    return optimizer_step_impl(ctx, optimizer, lr_scalar);
}

gd_status gd_lr_scheduler_value(const gd_lr_scheduler_config *config,
                                int step,
                                float *lr_out)
{
    const double pi = 3.14159265358979323846264338327950288;
    double lr;
    double progress;
    int warmup_steps;
    int total_steps;

    if (config == NULL || lr_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "lr scheduler argument is NULL");
    }
    if (!isfinite(config->max_lr) || !isfinite(config->min_lr) ||
        config->max_lr < 0.0F || config->min_lr < 0.0F ||
        config->min_lr > config->max_lr) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid lr scheduler bounds");
    }
    if (config->warmup_steps < 0 || config->total_steps <= 0 ||
        config->warmup_steps > config->total_steps) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid lr scheduler steps");
    }
    if (step < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "lr scheduler step must be nonnegative");
    }

    warmup_steps = config->warmup_steps;
    total_steps = config->total_steps;
    if (warmup_steps > 0 && step < warmup_steps) {
        lr = (double)config->max_lr * (double)step / (double)warmup_steps;
    } else if (total_steps == warmup_steps) {
        lr = (double)config->min_lr;
    } else {
        progress = (double)(step - warmup_steps) / (double)(total_steps - warmup_steps);
        if (progress < 0.0) {
            progress = 0.0;
        }
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

gd_status gd_lr_scheduler_write(gd_context *ctx,
                                const gd_lr_scheduler_config *config,
                                int step,
                                gd_tensor *lr_scalar,
                                float *lr_out)
{
    float lr = 0.0F;
    gd_status status;

    if (ctx == NULL || lr_scalar == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "lr scheduler write argument is NULL");
    }
    status = validate_lr_tensor(lr_scalar);
    if (status != GD_OK) {
        return status;
    }
    status = gd_lr_scheduler_value(config, step, &lr);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_copy_from_cpu(ctx, lr_scalar, &lr, sizeof(lr));
    if (status != GD_OK) {
        return status;
    }
    if (lr_out != NULL) {
        *lr_out = lr;
    }
    return GD_OK;
}

gd_status gd_optimizer_zero_grad(gd_context *ctx, gd_optimizer *optimizer)
{
    gd_status status = GD_OK;
    int i = 0;

    if (ctx == NULL || optimizer == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_optimizer_zero_grad argument is NULL");
    }
    for (i = 0; i < optimizer->n_slots; ++i) {
        gd_tensor *grad = NULL;

        status = _gd_tensor_ensure_grad(ctx, optimizer->slots[i].param, &grad);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_tensor_zero(grad);
        if (status != GD_OK) {
            return status;
        }
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}
