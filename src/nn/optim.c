#include "gradients/optim.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gradients/ops.h"

#include "../core/internal.h"
#include "../core/tensor_internal.h"
#include "../graph/graph_internal.h"

typedef struct adamw_slot {
    gd_tensor *param;  /* retained */
    gd_tensor *m;      /* owned */
    gd_tensor *v;      /* owned */
    float weight_decay;
    float lr_scale;
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

static gd_status validate_group(const gd_param_group *group)
{
    if (group == NULL || group->n_params < 0 ||
        (group->params == NULL && group->n_params > 0)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "adamw parameter group is invalid");
    }
    if (!isfinite(group->weight_decay) || group->weight_decay < 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "adamw group weight_decay must be finite and nonnegative");
    }
    if (!isfinite(group->lr_scale) || group->lr_scale <= 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "adamw group lr_scale must be finite and positive");
    }
    return GD_OK;
}

static gd_status add_adamw_slot(gd_context *ctx,
                                 gd_optimizer *opt,
                                 gd_tensor *p,
                                 float weight_decay,
                                 float lr_scale)
{
    gd_status status = GD_OK;
    int j = 0;

    if (p == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "optimizer parameter is NULL");
    }
    if (gd_tensor_dtype(p) != GD_DTYPE_F32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "adamw v1 supports F32 parameters only");
    }
    if (!gd_tensor_requires_grad(p)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "optimizer parameter must require grad");
    }
    for (j = 0; j < opt->n_slots; ++j) {
        if (same_parameter(opt->slots[j].param, p)) {
            if (opt->slots[j].weight_decay != weight_decay ||
                opt->slots[j].lr_scale != lr_scale) {
                return _gd_error(GD_ERR_INVALID_ARGUMENT,
                                 "tied parameter appears in conflicting adamw groups");
            }
            return GD_OK; /* tied weights share a single optimizer state */
        }
    }

    status = gd_tensor_retain(p);
    if (status != GD_OK) {
        return status;
    }
    opt->slots[opt->n_slots].param = p;
    opt->slots[opt->n_slots].weight_decay = weight_decay;
    opt->slots[opt->n_slots].lr_scale = lr_scale;
    status = make_state_like(ctx, p, &opt->slots[opt->n_slots].m);
    if (status != GD_OK) {
        gd_tensor_release(p);
        opt->slots[opt->n_slots].param = NULL;
        return status;
    }
    status = make_state_like(ctx, p, &opt->slots[opt->n_slots].v);
    if (status != GD_OK) {
        gd_tensor_release(p);
        gd_tensor_release(opt->slots[opt->n_slots].m);
        opt->slots[opt->n_slots].param = NULL;
        opt->slots[opt->n_slots].m = NULL;
        return status;
    }
    opt->n_slots += 1;
    return GD_OK;
}

static int count_group_params(const gd_param_group *groups, int n_groups)
{
    int total = 0;
    int i = 0;
    for (i = 0; i < n_groups; ++i) {
        total += groups[i].n_params;
    }
    return total;
}

gd_status gd_adamw_create_groups(gd_context *ctx,
                                 const gd_param_group *groups,
                                 int n_groups,
                                 const gd_adamw_config *config,
                                 gd_optimizer **out)
{
    gd_status status = GD_OK;
    gd_optimizer *opt = NULL;
    gd_tensor_desc step_desc;
    int total_params = 0;
    int g = 0;

    if (ctx == NULL || groups == NULL || n_groups <= 0 || config == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_adamw_create_groups argument is invalid");
    }
    *out = NULL;
    status = validate_config(config);
    if (status != GD_OK) {
        return status;
    }
    for (g = 0; g < n_groups; ++g) {
        status = validate_group(&groups[g]);
        if (status != GD_OK) {
            return status;
        }
    }
    total_params = count_group_params(groups, n_groups);

    opt = calloc(1U, sizeof(*opt));
    if (opt == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate optimizer");
    }
    opt->ctx = ctx;
    opt->config = *config;
    if (opt->config.state_dtype == GD_DTYPE_INVALID) {
        opt->config.state_dtype = GD_DTYPE_F32;
    }
    if (total_params > 0) {
        opt->slots = calloc((size_t)total_params, sizeof(*opt->slots));
        if (opt->slots == NULL) {
            free(opt);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate optimizer slots");
        }
    }

    for (g = 0; g < n_groups; ++g) {
        int i = 0;
        for (i = 0; i < groups[g].n_params; ++i) {
            status = add_adamw_slot(ctx, opt, groups[g].params[i], groups[g].weight_decay,
                                    groups[g].lr_scale);
            if (status != GD_OK) {
                goto fail;
            }
        }
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

gd_status gd_adamw_create(gd_context *ctx,
                          gd_tensor **params,
                          int n_params,
                          const gd_adamw_config *config,
                          gd_optimizer **out)
{
    gd_param_group group;

    if (config == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_adamw_create argument is invalid");
    }
    if ((params == NULL && n_params > 0) || n_params < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_adamw_create argument is invalid");
    }
    group.params = params;
    group.n_params = n_params;
    group.weight_decay = config->weight_decay;
    group.lr_scale = 1.0F;
    return gd_adamw_create_groups(ctx, &group, 1, config, out);
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

    for (i = 0; i < optimizer->n_slots; ++i) {
        gd_tensor *grad = NULL;
        gd_tensor *inputs[6];
        int n_inputs = lr_scalar != NULL ? 6 : 5;

        status = _gd_tensor_ensure_grad(ctx, optimizer->slots[i].param, &grad);
        if (status != GD_OK) {
            return status;
        }
        attrs.weight_decay = optimizer->slots[i].weight_decay;
        attrs.scale = optimizer->slots[i].lr_scale;
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

static const unsigned char optimizer_magic[8] = {'G', 'D', 'O', 'P', 'T', '1', 0, 0};

static gd_status optim_write_all(FILE *f, const void *data, size_t nbytes)
{
    if (nbytes == 0U) {
        return GD_OK;
    }
    if (fwrite(data, 1U, nbytes, f) != nbytes) {
        return _gd_error(GD_ERR_IO, "failed to write optimizer checkpoint");
    }
    return GD_OK;
}

static gd_status optim_read_all(FILE *f, void *data, size_t nbytes)
{
    if (nbytes == 0U) {
        return GD_OK;
    }
    if (fread(data, 1U, nbytes, f) != nbytes) {
        return _gd_error(GD_ERR_IO, "failed to read optimizer checkpoint");
    }
    return GD_OK;
}

static gd_status tensor_nbytes(gd_tensor *tensor, size_t *nbytes_out)
{
    return gd_tensor_desc_nbytes(_gd_tensor_desc_ptr(tensor), nbytes_out, NULL);
}

gd_status gd_optimizer_save(gd_optimizer *optimizer, const char *path)
{
    gd_status status = GD_OK;
    FILE *f = NULL;
    uint32_t version = 1U;
    uint32_t n_slots = 0U;
    int i = 0;

    if (optimizer == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_optimizer_save argument is NULL");
    }
    if (optimizer->n_slots < 0) {
        return _gd_error(GD_ERR_INVALID_STATE, "invalid optimizer slot count");
    }
    n_slots = (uint32_t)optimizer->n_slots;
    f = fopen(path, "wb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open optimizer checkpoint for write");
    }
    status = optim_write_all(f, optimizer_magic, sizeof(optimizer_magic));
    if (status == GD_OK) { status = optim_write_all(f, &version, sizeof(version)); }
    if (status == GD_OK) { status = optim_write_all(f, &n_slots, sizeof(n_slots)); }
    if (status == GD_OK) {
        float step = 0.0F;
        status = gd_tensor_copy_to_cpu(optimizer->ctx, optimizer->step, &step, sizeof(step));
        if (status == GD_OK) {
            status = optim_write_all(f, &step, sizeof(step));
        }
    }
    for (i = 0; i < optimizer->n_slots && status == GD_OK; ++i) {
        adamw_slot *slot = &optimizer->slots[i];
        const gd_tensor_desc *desc = _gd_tensor_desc_ptr(slot->param);
        size_t nbytes = 0U;
        uint64_t nbytes64 = 0U;
        uint32_t dtype = (uint32_t)desc->dtype;
        int32_t ndim = (int32_t)desc->ndim;
        void *m_buf = NULL;
        void *v_buf = NULL;

        status = tensor_nbytes(slot->param, &nbytes);
        if (status != GD_OK) { break; }
        nbytes64 = (uint64_t)nbytes;
        m_buf = malloc(nbytes);
        v_buf = malloc(nbytes);
        if (m_buf == NULL || v_buf == NULL) {
            free(m_buf);
            free(v_buf);
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate optimizer save buffer");
            break;
        }
        status = gd_tensor_copy_to_cpu(optimizer->ctx, slot->m, m_buf, nbytes);
        if (status == GD_OK) {
            status = gd_tensor_copy_to_cpu(optimizer->ctx, slot->v, v_buf, nbytes);
        }
        if (status == GD_OK) { status = optim_write_all(f, &dtype, sizeof(dtype)); }
        if (status == GD_OK) { status = optim_write_all(f, &ndim, sizeof(ndim)); }
        if (status == GD_OK) { status = optim_write_all(f, desc->sizes, sizeof(desc->sizes)); }
        if (status == GD_OK) { status = optim_write_all(f, &slot->weight_decay, sizeof(float)); }
        if (status == GD_OK) { status = optim_write_all(f, &slot->lr_scale, sizeof(float)); }
        if (status == GD_OK) { status = optim_write_all(f, &nbytes64, sizeof(nbytes64)); }
        if (status == GD_OK) { status = optim_write_all(f, m_buf, nbytes); }
        if (status == GD_OK) { status = optim_write_all(f, v_buf, nbytes); }
        free(m_buf);
        free(v_buf);
    }
    if (fclose(f) != 0 && status == GD_OK) {
        status = _gd_error(GD_ERR_IO, "failed to close optimizer checkpoint");
    }
    return status;
}

gd_status gd_optimizer_load(gd_optimizer *optimizer, const char *path)
{
    gd_status status = GD_OK;
    FILE *f = NULL;
    unsigned char magic[8];
    uint32_t version = 0U;
    uint32_t n_slots = 0U;
    int i = 0;

    if (optimizer == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_optimizer_load argument is NULL");
    }
    f = fopen(path, "rb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open optimizer checkpoint for read");
    }
    status = optim_read_all(f, magic, sizeof(magic));
    if (status == GD_OK && memcmp(magic, optimizer_magic, sizeof(optimizer_magic)) != 0) {
        status = _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid optimizer checkpoint magic");
    }
    if (status == GD_OK) { status = optim_read_all(f, &version, sizeof(version)); }
    if (status == GD_OK && version != 1U) {
        status = _gd_error(GD_ERR_UNSUPPORTED, "unsupported optimizer checkpoint version");
    }
    if (status == GD_OK) { status = optim_read_all(f, &n_slots, sizeof(n_slots)); }
    if (status == GD_OK && n_slots != (uint32_t)optimizer->n_slots) {
        status = _gd_error(GD_ERR_INVALID_ARGUMENT, "optimizer checkpoint slot count mismatch");
    }
    if (status == GD_OK) {
        float step = 0.0F;
        status = optim_read_all(f, &step, sizeof(step));
        if (status == GD_OK) {
            status = gd_tensor_copy_from_cpu(optimizer->ctx, optimizer->step, &step, sizeof(step));
        }
    }
    for (i = 0; i < optimizer->n_slots && status == GD_OK; ++i) {
        adamw_slot *slot = &optimizer->slots[i];
        const gd_tensor_desc *desc = _gd_tensor_desc_ptr(slot->param);
        uint32_t dtype = 0U;
        int32_t ndim = 0;
        int64_t sizes[GD_MAX_DIMS];
        float weight_decay = 0.0F;
        float lr_scale = 0.0F;
        uint64_t nbytes64 = 0U;
        size_t nbytes = 0U;
        size_t expected_nbytes = 0U;
        void *m_buf = NULL;
        void *v_buf = NULL;
        int dim = 0;

        memset(sizes, 0, sizeof(sizes));
        status = optim_read_all(f, &dtype, sizeof(dtype));
        if (status == GD_OK) { status = optim_read_all(f, &ndim, sizeof(ndim)); }
        if (status == GD_OK) { status = optim_read_all(f, sizes, sizeof(sizes)); }
        if (status == GD_OK) { status = optim_read_all(f, &weight_decay, sizeof(weight_decay)); }
        if (status == GD_OK) { status = optim_read_all(f, &lr_scale, sizeof(lr_scale)); }
        if (status == GD_OK) { status = optim_read_all(f, &nbytes64, sizeof(nbytes64)); }
        if (status != GD_OK) { break; }
        if (nbytes64 > (uint64_t)SIZE_MAX) {
            status = _gd_error(GD_ERR_INVALID_ARGUMENT, "optimizer tensor is too large");
            break;
        }
        nbytes = (size_t)nbytes64;
        status = tensor_nbytes(slot->param, &expected_nbytes);
        if (status != GD_OK) { break; }
        if ((uint32_t)desc->dtype != dtype || (int32_t)desc->ndim != ndim ||
            expected_nbytes != nbytes || slot->weight_decay != weight_decay ||
            slot->lr_scale != lr_scale) {
            status = _gd_error(GD_ERR_SHAPE, "optimizer checkpoint slot metadata mismatch");
            break;
        }
        for (dim = 0; dim < desc->ndim; ++dim) {
            if (desc->sizes[dim] != sizes[dim]) {
                status = _gd_error(GD_ERR_SHAPE, "optimizer checkpoint slot shape mismatch");
                break;
            }
        }
        if (status != GD_OK) { break; }
        m_buf = malloc(nbytes);
        v_buf = malloc(nbytes);
        if (m_buf == NULL || v_buf == NULL) {
            free(m_buf);
            free(v_buf);
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate optimizer load buffer");
            break;
        }
        status = optim_read_all(f, m_buf, nbytes);
        if (status == GD_OK) { status = optim_read_all(f, v_buf, nbytes); }
        if (status == GD_OK) { status = gd_tensor_copy_from_cpu(optimizer->ctx, slot->m, m_buf, nbytes); }
        if (status == GD_OK) { status = gd_tensor_copy_from_cpu(optimizer->ctx, slot->v, v_buf, nbytes); }
        free(m_buf);
        free(v_buf);
    }
    if (fclose(f) != 0 && status == GD_OK) {
        status = _gd_error(GD_ERR_IO, "failed to close optimizer checkpoint");
    }
    return status;
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
