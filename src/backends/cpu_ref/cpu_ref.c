#include "cpu_backend.h"

#include <stdlib.h>
#include <string.h>

#include "../../core/internal.h"
#include "../../core/tensor_internal.h"
#include "../backend.h"

typedef struct _gd_exec_buffer {
    gd_storage *storage;  /* owned for produced values, borrowed for leaves */
    size_t offset;
    bool owned;
} _gd_exec_buffer;

struct _gd_executable {
    const gd_graph *graph;
    int n_values;
    _gd_exec_buffer *buffers;
};

static gd_status value_ptr(_gd_executable *exe,
                           int value_id,
                           void **data_out,
                           const gd_tensor_desc **desc_out)
{
    _gd_exec_buffer *buffer = NULL;
    void *base = NULL;
    gd_status status = GD_OK;

    if (value_id < 0 || value_id >= exe->n_values) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "value id is out of range");
    }
    buffer = &exe->buffers[value_id];
    status = gd_storage_data_cpu(buffer->storage, &base);
    if (status != GD_OK) {
        return status;
    }
    *data_out = (unsigned char *)base + buffer->offset;
    if (desc_out != NULL) {
        *desc_out = &exe->graph->values[value_id].desc;
    }
    return GD_OK;
}

static gd_status require_f32(const gd_tensor_desc *desc)
{
    if (desc->dtype != GD_DTYPE_F32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "CPU_REF kernel requires F32 in v1");
    }
    return GD_OK;
}

static gd_status cpu_run_node(_gd_executable *exe, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *in_data[_GD_OP_MAX_INPUTS] = {0};
    const gd_tensor_desc *in_desc[_GD_OP_MAX_INPUTS] = {0};
    void *out_data = NULL;
    const gd_tensor_desc *out_desc = NULL;
    int i = 0;

    if (node->n_outputs < 0 || node->n_outputs > 3) {
        return _gd_error(GD_ERR_INTERNAL, "CPU_REF expects 0..3-output nodes");
    }

    for (i = 0; i < node->n_inputs; ++i) {
        status = value_ptr(exe, node->inputs[i], &in_data[i], &in_desc[i]);
        if (status != GD_OK) {
            return status;
        }
    }
    if (node->n_outputs >= 1) {
        status = value_ptr(exe, node->outputs[0], &out_data, &out_desc);
        if (status != GD_OK) {
            return status;
        }
    }

    switch (node->op) {
    case _GD_OP_ADD:
    case _GD_OP_MUL:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_elementwise(node->op, out_desc, out_data,
                                     in_desc[0], in_data[0], in_desc[1], in_data[1]);
    case _GD_OP_SCALE:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_scale(out_desc, out_data, in_data[0], node->attrs.scale);
    case _GD_OP_RELU:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_relu(out_desc, out_data, in_data[0]);
    case _GD_OP_SILU:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_silu(out_desc, out_data, in_data[0]);
    case _GD_OP_MATMUL:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_matmul(out_desc, out_data,
                                in_desc[0], in_data[0], node->attrs.trans_a,
                                in_desc[1], in_data[1], node->attrs.trans_b);
    case _GD_OP_LINEAR:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_linear(out_desc, out_data,
                                in_desc[0], in_data[0],
                                in_desc[1], in_data[1],
                                node->attrs.trans_b,
                                node->attrs.has_bias ? in_data[2] : NULL);
    case _GD_OP_SUM:
    case _GD_OP_MEAN:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_reduce(out_desc, out_data, in_desc[0], in_data[0],
                                node->attrs.dim, node->op == _GD_OP_MEAN);
    case _GD_OP_SOFTMAX:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_softmax(out_desc, out_data, in_data[0], node->attrs.dim);
    case _GD_OP_RMS_NORM:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_rms_norm(out_desc, out_data, in_data[0], in_data[1],
                                  node->attrs.eps);
    case _GD_OP_CROSS_ENTROPY:
        status = require_f32(in_desc[0]);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_cross_entropy(out_data, in_desc[0], in_data[0],
                                       in_desc[1], in_data[1], node->attrs.dim);
    case _GD_OP_CAST:
        return _gd_cpu_k_cast(out_desc, out_data, in_desc[0], in_data[0]);
    case _GD_OP_GELU:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_gelu(out_desc, out_data, in_data[0], node->attrs.gelu_tanh);
    case _GD_OP_GELU_BWD:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_gelu_bwd(out_desc, out_data, in_data[0], in_data[1],
                                  node->attrs.gelu_tanh);
    case _GD_OP_TRANSPOSE:
        return _gd_cpu_k_transpose(out_desc, out_data, in_desc[0], in_data[0],
                                   node->attrs.perm);
    case _GD_OP_EMBEDDING:
        return _gd_cpu_k_embedding(out_desc, out_data, in_desc[0], in_data[0],
                                   in_desc[1], in_data[1]);
    case _GD_OP_EMBEDDING_BWD:
        return _gd_cpu_k_embedding_bwd(out_desc, out_data, in_desc[0], in_data[0],
                                       in_desc[1], in_data[1]);
    case _GD_OP_RMS_NORM_BWD:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_rms_norm_bwd(out_desc, out_data, in_data[0], in_data[1],
                                      in_data[2], node->attrs.eps);
    case _GD_OP_RMS_NORM_WBWD:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_rms_norm_wbwd(in_desc[0], out_data, in_data[0], in_data[1],
                                       node->attrs.eps);
    case _GD_OP_ROPE:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_rope(out_desc, out_data, in_data[0], in_desc[1], in_data[1],
                              node->attrs.rope_theta, node->attrs.rope_n_dims,
                              node->attrs.rope_interleaved, 1.0F);
    case _GD_OP_ROPE_BWD:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_rope(out_desc, out_data, in_data[0], in_desc[1], in_data[1],
                              node->attrs.rope_theta, node->attrs.rope_n_dims,
                              node->attrs.rope_interleaved, -1.0F);
    case _GD_OP_SDPA: {
        const gd_tensor_desc *bias_desc = node->attrs.has_bias ? in_desc[3] : NULL;
        const float *bias = node->attrs.has_bias ? in_data[3] : NULL;

        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_sdpa(out_desc, out_data, in_desc[0], in_data[0],
                              in_desc[1], in_data[1], in_desc[2], in_data[2],
                              bias_desc, bias,
                              node->attrs.attn_scale, node->attrs.causal,
                              node->attrs.sliding_window);
    }
    case _GD_OP_SDPA_BWD: {
        /* inputs: go, q, k, v[, bias] ; outputs: dq, dk, dv */
        void *dk_data = NULL;
        void *dv_data = NULL;
        const gd_tensor_desc *dummy = NULL;
        const gd_tensor_desc *bias_desc = node->attrs.has_bias ? in_desc[4] : NULL;
        const float *bias = node->attrs.has_bias ? in_data[4] : NULL;

        status = value_ptr(exe, node->outputs[1], &dk_data, &dummy);
        if (status != GD_OK) {
            return status;
        }
        status = value_ptr(exe, node->outputs[2], &dv_data, &dummy);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_sdpa_bwd(in_desc[1], in_data[1], in_desc[2], in_data[2],
                                  in_desc[3], in_data[3], bias_desc, bias, in_data[0],
                                  out_data, dk_data, dv_data,
                                  node->attrs.attn_scale, node->attrs.causal,
                                  node->attrs.sliding_window);
    }
    case _GD_OP_COPY:
        return _gd_cpu_k_copy(out_desc, out_data, in_desc[0], in_data[0]);
    case _GD_OP_RELU_BWD:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_relu_bwd(out_desc, out_data, in_data[0], in_data[1]);
    case _GD_OP_SILU_BWD:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_silu_bwd(out_desc, out_data, in_data[0], in_data[1]);
    case _GD_OP_SOFTMAX_BWD:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_softmax_bwd(out_desc, out_data, in_data[0], in_data[1],
                                     node->attrs.dim);
    case _GD_OP_SUM_BWD:
    case _GD_OP_MEAN_BWD:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_sum_bwd(out_desc, out_data, in_data[0], node->attrs.dim,
                                 node->op == _GD_OP_MEAN_BWD);
    case _GD_OP_CROSS_ENTROPY_BWD:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_cross_entropy_bwd(in_desc[0], out_data, in_data[0],
                                           in_desc[1], in_data[1], in_data[2],
                                           node->attrs.dim);
    case _GD_OP_STEP_INC:
        status = require_f32(in_desc[0]);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_step_inc(in_data[0]);
    case _GD_OP_ADAMW_STEP:
        status = require_f32(in_desc[0]);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_adamw(in_desc[0], in_data[0], in_data[1], in_data[2],
                               in_data[3], in_data[4],
                               node->attrs.lr, node->attrs.beta1, node->attrs.beta2,
                               node->attrs.eps, node->attrs.weight_decay);
    case _GD_OP_REDUCE_TO:
        status = require_f32(out_desc);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_reduce_to(out_desc, out_data, in_desc[0], in_data[0]);
    case _GD_OP_ASSERT_FINITE:
        status = require_f32(in_desc[0]);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_assert_finite(in_desc[0], in_data[0]);
    case _GD_OP_ASSERT_CLOSE:
        status = require_f32(in_desc[0]);
        if (status != GD_OK) {
            return status;
        }
        return _gd_cpu_k_assert_close(in_desc[0], in_data[0], in_data[1],
                                      node->attrs.atol, node->attrs.rtol);
    case _GD_OP_INVALID:
    case _GD_OP_BACKWARD:
    case _GD_OP_ZERO_GRAD:
    case _GD_OP_OPTIMIZER_STEP:
        return _gd_error(GD_ERR_UNSUPPORTED, "op is not implemented in CPU_REF yet");
    }
    return _gd_error(GD_ERR_INTERNAL, "unknown op kind");
}

/* ---- Compiled executable: naive per-value buffer plan. ---- */

static void cpu_executable_free(_gd_backend *self, _gd_executable *exe)
{
    int i = 0;

    (void)self;
    if (exe == NULL) {
        return;
    }
    for (i = 0; i < exe->n_values; ++i) {
        if (exe->buffers[i].owned) {
            gd_storage_release(exe->buffers[i].storage);
        }
    }
    free(exe->buffers);
    free(exe);
}

static gd_status cpu_compile(_gd_backend *self, gd_graph *graph, _gd_executable **out)
{
    gd_status status = GD_OK;
    _gd_executable *exe = NULL;
    int n = 0;
    int i = 0;

    (void)self;
    *out = NULL;
    n = graph->n_values;

    exe = calloc(1U, sizeof(*exe));
    if (exe == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate executable");
    }
    exe->graph = graph;
    exe->n_values = n;
    exe->buffers = calloc((size_t)(n > 0 ? n : 1), sizeof(*exe->buffers));
    if (exe->buffers == NULL) {
        free(exe);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate exec buffers");
    }

    for (i = 0; i < n; ++i) {
        const _gd_value *value = &graph->values[i];

        if (value->external != NULL) {
            gd_storage *storage = gd_tensor_storage(value->external);

            if (!_gd_tensor_is_contiguous(value->external)) {
                status = _gd_error(GD_ERR_UNSUPPORTED, "CPU_REF requires contiguous leaf inputs");
                goto fail;
            }
            if (storage == NULL) {
                status = _gd_error(GD_ERR_INVALID_STATE, "leaf input has no storage");
                goto fail;
            }
            exe->buffers[i].storage = storage;
            exe->buffers[i].offset = (size_t)value->desc.storage_offset_bytes;
            exe->buffers[i].owned = false;
        } else {
            gd_storage_desc sdesc;
            gd_storage *storage = NULL;
            size_t nbytes = 0U;
            size_t alignment = 0U;

            status = gd_tensor_desc_nbytes(&value->desc, &nbytes, &alignment);
            if (status != GD_OK) {
                goto fail;
            }
            sdesc = (gd_storage_desc){value->desc.device, self->caps.default_memory,
                                      nbytes, alignment};
            status = gd_storage_create(graph->ctx, &sdesc, &storage);
            if (status != GD_OK) {
                goto fail;
            }
            exe->buffers[i].storage = storage;
            exe->buffers[i].offset = 0U;
            exe->buffers[i].owned = true;
        }
    }

    *out = exe;
    return GD_OK;

fail:
    cpu_executable_free(self, exe);
    return status;
}

static gd_status cpu_execute(_gd_backend *self, _gd_executable *exe)
{
    gd_status status = GD_OK;
    int i = 0;

    (void)self;
    for (i = 0; i < exe->graph->n_nodes; ++i) {
        status = cpu_run_node(exe, &exe->graph->nodes[i]);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static gd_status cpu_execute_until(_gd_backend *self, _gd_executable *exe, int node_id)
{
    gd_status status = GD_OK;
    int i = 0;

    (void)self;
    for (i = 0; i <= node_id && i < exe->graph->n_nodes; ++i) {
        status = cpu_run_node(exe, &exe->graph->nodes[i]);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static gd_status cpu_value_storage(_gd_backend *self, _gd_executable *exe, int value_id,
                                   gd_storage **storage_out, size_t *offset_out)
{
    (void)self;
    if (value_id < 0 || value_id >= exe->n_values) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "value id is out of range");
    }
    *storage_out = exe->buffers[value_id].storage;
    *offset_out = exe->buffers[value_id].offset;
    return GD_OK;
}

/* ---- Backend registration. ---- */

static int is_power_of_two(size_t value)
{
    return value != 0U && (value & (value - 1U)) == 0U;
}

static gd_status normalize_alignment(size_t requested, size_t *out)
{
    size_t alignment = requested;

    if (alignment < sizeof(void *)) {
        alignment = sizeof(void *);
    }
    if (!is_power_of_two(alignment)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "alignment must be a power of two");
    }
    *out = alignment;
    return GD_OK;
}

static gd_status cpu_storage_alloc(_gd_backend *self, const gd_storage_desc *desc,
                                   void **handle_out)
{
    gd_status status = GD_OK;
    size_t alignment = 0U;
    void *data = NULL;
    int rc = 0;

    (void)self;
    *handle_out = NULL;
    if (desc->device.type != GD_DEVICE_CPU) {
        return _gd_error(GD_ERR_DEVICE, "cpu_ref storage requires a CPU device");
    }
    if (desc->memory != GD_MEM_HOST) {
        return _gd_error(GD_ERR_UNSUPPORTED, "cpu_ref storage supports GD_MEM_HOST only");
    }
    status = normalize_alignment(desc->alignment, &alignment);
    if (status != GD_OK) {
        return status;
    }
    rc = posix_memalign(&data, alignment, desc->nbytes);
    if (rc != 0) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "posix_memalign failed");
    }
    memset(data, 0, desc->nbytes);
    *handle_out = data;
    return GD_OK;
}

static void cpu_storage_free(_gd_backend *self, void *handle)
{
    (void)self;
    free(handle);
}

static gd_status cpu_storage_host_ptr(_gd_backend *self, void *handle, void **ptr_out)
{
    (void)self;
    *ptr_out = handle;
    return GD_OK;
}

static gd_status cpu_upload(_gd_backend *self, void *dst_handle, size_t dst_off,
                            const void *src, size_t nbytes)
{
    (void)self;
    memcpy((unsigned char *)dst_handle + dst_off, src, nbytes);
    return GD_OK;
}

static gd_status cpu_download(_gd_backend *self, void *src_handle, size_t src_off,
                              void *dst, size_t nbytes)
{
    (void)self;
    memcpy(dst, (const unsigned char *)src_handle + src_off, nbytes);
    return GD_OK;
}

static gd_status cpu_backend_init(_gd_backend *self, gd_context *ctx, int device_index)
{
    (void)ctx;
    (void)device_index;
    self->caps.host_visible = true;
    self->caps.supports_cpu_ref = true;
    self->caps.default_memory = GD_MEM_HOST;
    return GD_OK;
}

static void cpu_backend_shutdown(_gd_backend *self)
{
    (void)self;
}

static gd_status cpu_backend_synchronize(_gd_backend *self)
{
    (void)self;
    return GD_OK; /* CPU_REF executes synchronously */
}

static const _gd_backend_vtable cpu_backend_vtable = {
    .type = GD_DEVICE_CPU,
    .name = "cpu_ref",
    .init = cpu_backend_init,
    .shutdown = cpu_backend_shutdown,
    .storage_alloc = cpu_storage_alloc,
    .storage_free = cpu_storage_free,
    .storage_host_ptr = cpu_storage_host_ptr,
    .upload = cpu_upload,
    .download = cpu_download,
    .compile = cpu_compile,
    .execute = cpu_execute,
    .execute_until = cpu_execute_until,
    .executable_free = cpu_executable_free,
    .value_storage = cpu_value_storage,
    .supports_node = NULL,
    .synchronize = cpu_backend_synchronize,
};

gd_status _gd_cpu_backend_register(gd_context *ctx)
{
    return _gd_context_register_backend(ctx, &cpu_backend_vtable);
}
