#include "cpu_op.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../core/internal.h"
#include "../../core/tensor_internal.h"
#include "../backend.h"

#include "cpu_registry.inc"

typedef struct _gd_exec_buffer {
    gd_storage *storage;  /* owned for produced/runtime-bound values, borrowed for leaves */
    size_t offset;
    bool owned;
    bool input_binding;
} _gd_exec_buffer;

struct _gd_executable {
    const gd_graph *graph;
    int n_values;
    _gd_exec_buffer *buffers;
    uint64_t run_id;
};

gd_status _gd_cpu_exec_value(_gd_cpu_exec *exec,
                             int value_id,
                             void **data_out,
                             const gd_tensor_desc **desc_out)
{
    _gd_exec_buffer *buffer = NULL;
    void *base = NULL;
    gd_status status = GD_OK;

    if (exec == NULL || data_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "CPU_REF exec value argument is NULL");
    }
    if (value_id < 0 || value_id >= exec->n_values) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "value id is out of range");
    }
    buffer = &exec->buffers[value_id];
    if (buffer->storage == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "CPU_REF graph input is unbound");
    }
    status = gd_storage_data_cpu(buffer->storage, &base);
    if (status != GD_OK) {
        return status;
    }
    *data_out = (unsigned char *)base + buffer->offset;
    if (desc_out != NULL) {
        *desc_out = &exec->graph->values[value_id].desc;
    }
    return GD_OK;
}

gd_status _gd_cpu_exec_input(_gd_cpu_exec *exec,
                             const _gd_node *node,
                             int input_index,
                             void **data_out,
                             const gd_tensor_desc **desc_out)
{
    if (node == NULL || input_index < 0 || input_index >= node->n_inputs) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "CPU_REF input index is out of range");
    }
    return _gd_cpu_exec_value(exec, node->inputs[input_index], data_out, desc_out);
}

gd_status _gd_cpu_exec_output(_gd_cpu_exec *exec,
                              const _gd_node *node,
                              int output_index,
                              void **data_out,
                              const gd_tensor_desc **desc_out)
{
    if (node == NULL || output_index < 0 || output_index >= node->n_outputs) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "CPU_REF output index is out of range");
    }
    return _gd_cpu_exec_value(exec, node->outputs[output_index], data_out, desc_out);
}

uint64_t _gd_cpu_exec_run_id(const _gd_cpu_exec *exec)
{
    return exec == NULL ? 0U : exec->run_id;
}

gd_status _gd_cpu_require_f32(const gd_tensor_desc *desc)
{
    if (desc == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "CPU_REF descriptor is NULL");
    }
    if (desc->dtype != GD_DTYPE_F32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "CPU_REF kernel requires F32 in v1");
    }
    return GD_OK;
}

const _gd_cpu_op *_gd_cpu_op_for(_gd_op_kind kind)
{
    unsigned i = 0;

    for (i = 0u; i < g_cpu_op_count; ++i) {
        if (g_cpu_ops[i] != NULL && g_cpu_ops[i]->kind == kind) {
            return g_cpu_ops[i];
        }
    }
    return NULL;
}

gd_status _gd_cpu_support_default(const _gd_node *node)
{
    if (node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "CPU_REF node is NULL");
    }
    return _gd_op_validate_arity(node->op, node->n_inputs, node->n_outputs);
}

static gd_status cpu_run_node(_gd_executable *exe, const _gd_node *node)
{
    gd_status status = GD_OK;
    const _gd_cpu_op *op = NULL;

    if (node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "CPU_REF node is NULL");
    }
    op = _gd_cpu_op_for(node->op);
    if (op == NULL || op->run == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "op is not implemented in CPU_REF yet");
    }
    status = op->support != NULL ? op->support(node) : _gd_cpu_support_default(node);
    if (status != GD_OK) {
        return status;
    }
    return op->run(exe, node);
}

/* ---- Compiled executable: lifetime-planned value buffers. ---- */

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
    _gd_memory_plan plan = {0};
    gd_storage **slot_storage = NULL;
    int n = 0;
    int i = 0;

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

    status = _gd_memory_plan_build(graph, &plan);
    if (status != GD_OK) {
        goto fail;
    }
    if (plan.n_slots > 0) {
        slot_storage = calloc((size_t)plan.n_slots, sizeof(*slot_storage));
        if (slot_storage == NULL) {
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate CPU_REF plan slots");
            goto fail;
        }
        for (i = 0; i < plan.n_slots; ++i) {
            gd_storage_desc sdesc = {0};
            sdesc.device = (gd_device){GD_DEVICE_CPU, self->device_index};
            sdesc.memory = self->caps.default_memory;
            sdesc.nbytes = plan.slot_nbytes[i];
            sdesc.alignment = plan.slot_alignment[i];
            status = gd_storage_create(graph->ctx, &sdesc, &slot_storage[i]);
            if (status != GD_OK) {
                goto fail;
            }
        }
    }

    for (i = 0; i < n; ++i) {
        const _gd_value *value = &graph->values[i];
        int slot = plan.values != NULL ? plan.values[i].slot : -1;

        if (value->kind == _GD_VALUE_INPUT) {
            exe->buffers[i].storage = NULL;
            exe->buffers[i].offset = 0U;
            exe->buffers[i].owned = false;
            exe->buffers[i].input_binding = true;
        } else if (value->external != NULL) {
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
        } else if (slot >= 0) {
            status = gd_storage_retain(slot_storage[slot]);
            if (status != GD_OK) {
                goto fail;
            }
            exe->buffers[i].storage = slot_storage[slot];
            exe->buffers[i].offset = 0U;
            exe->buffers[i].owned = true;
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

    for (i = 0; i < plan.n_slots; ++i) {
        gd_storage_release(slot_storage[i]);
    }
    free(slot_storage);
    _gd_memory_plan_free(&plan);
    *out = exe;
    return GD_OK;

fail:
    if (slot_storage != NULL) {
        for (i = 0; i < plan.n_slots; ++i) {
            gd_storage_release(slot_storage[i]);
        }
        free(slot_storage);
    }
    _gd_memory_plan_free(&plan);
    cpu_executable_free(self, exe);
    return status;
}

static gd_status cpu_apply_runner_bindings(_gd_executable *exe,
                                            const gd_graph_runner *runner)
{
    gd_status status = _gd_graph_runner_validate_ready(runner);
    int i = 0;

    if (status != GD_OK) {
        return status;
    }
    if (runner->graph != exe->graph) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "runner graph mismatch");
    }
    for (i = 0; i < exe->graph->n_inputs; ++i) {
        gd_graph_input *input = exe->graph->inputs[i];
        gd_tensor *tensor = runner->bindings[i];
        gd_storage *storage = gd_tensor_storage(tensor);
        _gd_exec_buffer *buffer = NULL;

        if (input == NULL || input->value_id < 0 || input->value_id >= exe->n_values ||
            storage == NULL) {
            return _gd_error(GD_ERR_INVALID_STATE, "invalid CPU_REF graph input binding");
        }
        buffer = &exe->buffers[input->value_id];
        if (!buffer->input_binding) {
            return _gd_error(GD_ERR_INTERNAL, "CPU_REF binding target is not an input");
        }
        if (buffer->owned && buffer->storage != storage) {
            gd_storage_release(buffer->storage);
            buffer->storage = NULL;
            buffer->owned = false;
        }
        if (buffer->storage != storage) {
            status = gd_storage_retain(storage);
            if (status != GD_OK) {
                return status;
            }
            buffer->storage = storage;
            buffer->owned = true;
        }
        buffer->offset = (size_t)_gd_tensor_desc_ptr(tensor)->storage_offset_bytes;
    }
    return GD_OK;
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
    exe->run_id += 1U;
    return GD_OK;
}

static gd_status cpu_execute_bound(_gd_backend *self,
                                   _gd_executable *exe,
                                   const gd_graph_runner *runner)
{
    gd_status status = cpu_apply_runner_bindings(exe, runner);
    if (status != GD_OK) {
        return status;
    }
    return cpu_execute(self, exe);
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
    exe->run_id += 1U;
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

static gd_status cpu_check_node(_gd_backend *self,
                                const gd_graph *graph,
                                const _gd_node *node)
{
    const _gd_cpu_op *op = NULL;

    (void)self;
    (void)graph;
    if (node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "CPU_REF node is NULL");
    }
    op = _gd_cpu_op_for(node->op);
    if (op == NULL || op->run == NULL) {
        char msg[112];
        (void)snprintf(msg, sizeof(msg), "cpu_ref has no kernel for op '%s'",
                       _gd_op_kind_name(node->op));
        return _gd_error(GD_ERR_UNSUPPORTED, msg);
    }
    return op->support != NULL ? op->support(node) : _gd_cpu_support_default(node);
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
    .execute_bound = cpu_execute_bound,
    .execute_until = cpu_execute_until,
    .executable_free = cpu_executable_free,
    .value_storage = cpu_value_storage,
    .check_node = cpu_check_node,
    .synchronize = cpu_backend_synchronize,
};

gd_status _gd_cpu_backend_register(gd_context *ctx)
{
    return _gd_context_register_backend(ctx, &cpu_backend_vtable);
}
