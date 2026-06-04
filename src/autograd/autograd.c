#include "autograd_internal.h"

#include "../core/backend.h"
#include "../core/memory_internal.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define GD_AUTOGRAD_MAX_NODES 4096U
#define GD_AUTOGRAD_MAX_REFS 24576U
#define GD_AUTOGRAD_MAX_ATTR_BYTES 65536U
#define GD_AUTOGRAD_MAX_GRADS 4096U
#define GD_AUTOGRAD_GRAD_ALIGNMENT 256U

static void gd_autograd_reset(gd_autograd_state *state)
{
    if (state == NULL) {
        return;
    }
    state->n_nodes = 0U;
    state->n_refs = 0U;
    state->attrs_used = 0U;
    state->n_grads = 0U;
}

static void *gd_autograd_calloc(size_t count, size_t size)
{
    if (count == 0U || size == 0U || count > SIZE_MAX / size) {
        return NULL;
    }
    return calloc(count, size);
}

gd_status gd_autograd_state_create(gd_context *ctx, gd_autograd_state **out_state)
{
    gd_autograd_state *state;
    if (out_state == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_state = NULL;
    state = (gd_autograd_state *)gd_autograd_calloc(1U, sizeof(*state));
    if (state == NULL) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "autograd state allocation failed");
    }
    state->nodes = (gd_tape_node *)gd_autograd_calloc(GD_AUTOGRAD_MAX_NODES, sizeof(*state->nodes));
    state->refs = (gd_tape_ref *)gd_autograd_calloc(GD_AUTOGRAD_MAX_REFS, sizeof(*state->refs));
    state->attrs = (unsigned char *)gd_autograd_calloc(GD_AUTOGRAD_MAX_ATTR_BYTES, sizeof(*state->attrs));
    state->grads = (gd_grad_slot *)gd_autograd_calloc(GD_AUTOGRAD_MAX_GRADS, sizeof(*state->grads));
    if (state->nodes == NULL || state->refs == NULL || state->attrs == NULL || state->grads == NULL) {
        gd_autograd_state_destroy(state);
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "autograd tape allocation failed");
    }
    state->cap_nodes = GD_AUTOGRAD_MAX_NODES;
    state->cap_refs = GD_AUTOGRAD_MAX_REFS;
    state->attrs_cap = GD_AUTOGRAD_MAX_ATTR_BYTES;
    state->cap_grads = GD_AUTOGRAD_MAX_GRADS;
    state->user_enabled = true;
    state->recording = false;
    *out_state = state;
    return GD_OK;
}

void gd_autograd_state_destroy(gd_autograd_state *state)
{
    if (state == NULL) {
        return;
    }
    free(state->nodes);
    free(state->refs);
    free(state->attrs);
    free(state->grads);
    memset(state, 0, sizeof(*state));
    free(state);
}

gd_status gd_autograd_on_begin(gd_context *ctx)
{
    gd_autograd_state *state = gd_context_autograd(ctx);
    if (state == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    gd_autograd_reset(state);
    state->recording = state->user_enabled && gd_context_scope_mode(ctx) == GD_SCOPE_TRAIN;
    return GD_OK;
}

void gd_autograd_on_end(gd_context *ctx)
{
    gd_autograd_state *state = gd_context_autograd(ctx);
    if (state != NULL) {
        state->recording = false;
    }
}

static gd_status gd_autograd_check_ref_capacity(gd_autograd_state *state,
                                                uint16_t n_inputs,
                                                uint16_t n_outputs,
                                                uint16_t n_saved)
{
    uint32_t needed = (uint32_t)n_inputs + (uint32_t)n_outputs + (uint32_t)n_saved;
    if (needed > state->cap_refs || state->n_refs > state->cap_refs - needed) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    return GD_OK;
}

static void gd_autograd_push_ref(gd_autograd_state *state, const gd_tensor *tensor)
{
    gd_tape_ref *ref = &state->refs[state->n_refs];
    memset(ref, 0, sizeof(*ref));
    if (tensor != NULL) {
        ref->tensor = *tensor;
        ref->id = tensor->id;
        ref->version = tensor->version;
    }
    state->n_refs += 1U;
}

gd_status gd_autograd_record(gd_context *ctx,
                             gd_op_kind op,
                             const gd_tensor *const *inputs,
                             uint16_t n_inputs,
                             gd_tensor *const *outputs,
                             uint16_t n_outputs,
                             const void *attrs,
                             uint32_t attrs_size,
                             const gd_tensor *const *saved,
                             uint16_t n_saved)
{
    gd_autograd_state *state;
    gd_tape_node *node;
    bool needs_grad = false;
    uint16_t i;
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    state = gd_context_autograd(ctx);
    if (state == NULL || !state->recording || gd_context_scope_mode(ctx) != GD_SCOPE_TRAIN) {
        return GD_OK;
    }
    if ((n_inputs != 0U && inputs == NULL) || n_outputs == 0U || outputs == NULL ||
        (n_saved != 0U && saved == NULL) || (attrs_size != 0U && attrs == NULL)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid autograd record arguments");
    }
    for (i = 0U; i < n_inputs; ++i) {
        if (inputs[i] == NULL) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "autograd input is NULL");
        }
        if (inputs[i]->requires_grad) {
            if (inputs[i]->id == 0U) {
                return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "requires_grad tensor has no id");
            }
            needs_grad = true;
        }
    }
    if (!needs_grad) {
        return GD_OK;
    }
    for (i = 0U; i < n_outputs; ++i) {
        if (outputs[i] == NULL || outputs[i]->id == 0U) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid autograd output");
        }
    }
    if (state->n_nodes >= state->cap_nodes) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "autograd tape node capacity exceeded");
    }
    if (gd_autograd_check_ref_capacity(state, n_inputs, n_outputs, n_saved) != GD_OK) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "autograd tape ref capacity exceeded");
    }
    if (attrs_size > state->attrs_cap || state->attrs_used > state->attrs_cap - attrs_size) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "autograd attr capacity exceeded");
    }

    node = &state->nodes[state->n_nodes];
    memset(node, 0, sizeof(*node));
    node->op = op;
    node->first_input = state->n_refs;
    node->n_inputs = n_inputs;
    for (i = 0U; i < n_inputs; ++i) {
        gd_autograd_push_ref(state, inputs[i]);
    }
    node->first_output = state->n_refs;
    node->n_outputs = n_outputs;
    for (i = 0U; i < n_outputs; ++i) {
        outputs[i]->requires_grad = true;
        outputs[i]->is_leaf = false;
        gd_autograd_push_ref(state, outputs[i]);
    }
    node->first_saved = state->n_refs;
    node->n_saved = n_saved;
    for (i = 0U; i < n_saved; ++i) {
        gd_autograd_push_ref(state, saved[i]);
    }
    node->attrs_offset = state->attrs_used;
    node->attrs_size = attrs_size;
    if (attrs_size != 0U) {
        memcpy(state->attrs + state->attrs_used, attrs, attrs_size);
        state->attrs_used += attrs_size;
    }
    state->n_nodes += 1U;
    return GD_OK;
}

const gd_tensor *gd_tape_input(const gd_autograd_state *tape,
                               const gd_tape_node *node,
                               uint16_t index)
{
    if (tape == NULL || node == NULL || index >= node->n_inputs ||
        node->first_input > tape->n_refs ||
        (uint32_t)index >= tape->n_refs - node->first_input) {
        return NULL;
    }
    return &tape->refs[node->first_input + (uint32_t)index].tensor;
}

const gd_tensor *gd_tape_output(const gd_autograd_state *tape,
                                const gd_tape_node *node,
                                uint16_t index)
{
    if (tape == NULL || node == NULL || index >= node->n_outputs ||
        node->first_output > tape->n_refs ||
        (uint32_t)index >= tape->n_refs - node->first_output) {
        return NULL;
    }
    return &tape->refs[node->first_output + (uint32_t)index].tensor;
}

const gd_tensor *gd_tape_saved(const gd_autograd_state *tape,
                               const gd_tape_node *node,
                               uint16_t index)
{
    if (tape == NULL || node == NULL || index >= node->n_saved ||
        node->first_saved > tape->n_refs ||
        (uint32_t)index >= tape->n_refs - node->first_saved) {
        return NULL;
    }
    return &tape->refs[node->first_saved + (uint32_t)index].tensor;
}

const void *gd_tape_attrs(const gd_autograd_state *tape,
                          const gd_tape_node *node,
                          uint32_t expected_size)
{
    if (tape == NULL || node == NULL || node->attrs_size != expected_size ||
        node->attrs_offset > tape->attrs_used ||
        node->attrs_size > tape->attrs_used - node->attrs_offset) {
        return NULL;
    }
    return tape->attrs + node->attrs_offset;
}

static gd_grad_slot *gd_find_grad_slot(gd_autograd_state *state, uint64_t tensor_id)
{
    uint32_t i;
    if (state == NULL || tensor_id == 0U) {
        return NULL;
    }
    for (i = 0U; i < state->n_grads; ++i) {
        if (state->grads[i].occupied && state->grads[i].tensor_id == tensor_id) {
            return &state->grads[i];
        }
    }
    return NULL;
}

static const gd_grad_slot *gd_find_grad_slot_const(const gd_autograd_state *state, uint64_t tensor_id)
{
    uint32_t i;
    if (state == NULL || tensor_id == 0U) {
        return NULL;
    }
    for (i = 0U; i < state->n_grads; ++i) {
        if (state->grads[i].occupied && state->grads[i].tensor_id == tensor_id) {
            return &state->grads[i];
        }
    }
    return NULL;
}

static gd_status gd_create_grad_slot(gd_bwd_ctx *bwd,
                                     uint64_t tensor_id,
                                     const gd_tensor *like,
                                     gd_grad_slot **out_slot)
{
    gd_grad_slot *slot;
    gd_status st;
    if (bwd == NULL || bwd->tape == NULL || like == NULL || out_slot == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (bwd->tape->n_grads >= bwd->tape->cap_grads) {
        return gd_context_set_error(bwd->ctx, GD_ERR_OUT_OF_MEMORY, "autograd grad capacity exceeded");
    }
    slot = &bwd->tape->grads[bwd->tape->n_grads];
    memset(slot, 0, sizeof(*slot));
    st = gd_tensor_zeros(bwd->ctx,
                         GD_ARENA_SCRATCH,
                         like->dtype,
                         like->rank,
                         like->shape,
                         GD_AUTOGRAD_GRAD_ALIGNMENT,
                         &slot->grad);
    if (st != GD_OK) {
        return st;
    }
    slot->grad.requires_grad = false;
    slot->grad.is_leaf = false;
    slot->tensor_id = tensor_id;
    slot->occupied = true;
    bwd->tape->n_grads += 1U;
    *out_slot = slot;
    return GD_OK;
}

static bool gd_tensors_same_shape(const gd_tensor *a, const gd_tensor *b)
{
    uint32_t i;
    if (a == NULL || b == NULL || a->dtype != b->dtype || a->rank != b->rank) {
        return false;
    }
    for (i = 0U; i < a->rank; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

static gd_status gd_backend_accumulate_tensor(gd_context *ctx,
                                              gd_tensor *dst,
                                              const gd_tensor *src)
{
    gd_backend *backend;
    gd_status st;
    int64_t numel;
    size_t count;
    if (ctx == NULL || dst == NULL || src == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, dst);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, src);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_tensors_same_shape(dst, src)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "gradient accumulation shape mismatch");
    }
    if (!gd_tensor_is_contiguous(dst) || !gd_tensor_is_contiguous(src)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "gradient accumulation requires contiguous tensors");
    }
    st = gd_tensor_numel(src, &numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "gradient accumulation invalid shape");
    }
    if ((uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "gradient accumulation count overflow");
    }
    count = (size_t)numel;
    backend = gd_context_backend(ctx);
    if (backend == NULL || dst->storage.buffer == NULL || src->storage.buffer == NULL) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "gradient accumulation missing backend buffer");
    }
    st = gd_backend_accumulate(backend,
                               (gd_backend_buffer *)dst->storage.buffer,
                               gd_tensor_storage_offset(dst),
                               (gd_backend_buffer *)src->storage.buffer,
                               gd_tensor_storage_offset(src),
                               count,
                               (uint32_t)dst->dtype);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend gradient accumulation failed");
    }
    dst->version += 1U;
    if (dst->version == 0U) {
        dst->version = 1U;
    }
    return GD_OK;
}

static bool gd_node_has_output_grad(gd_autograd_state *state, const gd_tape_node *node)
{
    uint16_t i;
    if (state == NULL || node == NULL) {
        return false;
    }
    for (i = 0U; i < node->n_outputs; ++i) {
        const gd_tensor *out = gd_tape_output(state, node, i);
        if (out != NULL && gd_find_grad_slot(state, out->id) != NULL) {
            return true;
        }
    }
    return false;
}

bool gd_autograd_get_grad(gd_bwd_ctx *bwd, uint64_t tensor_id, gd_tensor *out_grad)
{
    gd_grad_slot *slot;
    if (out_grad != NULL) {
        memset(out_grad, 0, sizeof(*out_grad));
    }
    if (bwd == NULL || out_grad == NULL) {
        return false;
    }
    slot = gd_find_grad_slot(bwd->tape, tensor_id);
    if (slot == NULL) {
        return false;
    }
    *out_grad = slot->grad;
    return true;
}

gd_status gd_autograd_accumulate(gd_bwd_ctx *bwd,
                                 uint64_t tensor_id,
                                 const gd_tensor *contrib)
{
    gd_grad_slot *slot;
    gd_status st;
    if (contrib == NULL) {
        return GD_OK;
    }
    if (bwd == NULL || bwd->ctx == NULL || bwd->tape == NULL || tensor_id == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    slot = gd_find_grad_slot(bwd->tape, tensor_id);
    if (slot == NULL) {
        st = gd_create_grad_slot(bwd, tensor_id, contrib, &slot);
        if (st != GD_OK) {
            return st;
        }
    }
    return gd_backend_accumulate_tensor(bwd->ctx, &slot->grad, contrib);
}

static gd_status gd_seed_output_grad(gd_bwd_ctx *bwd,
                                     const gd_tensor *output,
                                     const gd_tensor *grad_output)
{
    gd_tensor seed;
    gd_status st;
    if (output == NULL || output->id == 0U) {
        return gd_context_set_error(bwd->ctx, GD_ERR_INVALID_ARGUMENT, "invalid backward output tensor");
    }
    if (grad_output != NULL) {
        if (!gd_tensors_same_shape(output, grad_output)) {
            return gd_context_set_error(bwd->ctx, GD_ERR_INVALID_ARGUMENT, "backward seed shape mismatch");
        }
        return gd_autograd_accumulate(bwd, output->id, grad_output);
    }
    st = gd_tensor_ones(bwd->ctx,
                        GD_ARENA_SCRATCH,
                        output->dtype,
                        output->rank,
                        output->shape,
                        GD_AUTOGRAD_GRAD_ALIGNMENT,
                        &seed);
    if (st != GD_OK) {
        return st;
    }
    seed.requires_grad = false;
    seed.is_leaf = false;
    return gd_autograd_accumulate(bwd, output->id, &seed);
}

gd_status gd_backward_many(gd_context *ctx,
                           uint32_t n_outputs,
                           const gd_tensor *const *outputs,
                           const gd_tensor *const *grad_outputs)
{
    gd_autograd_state *state;
    gd_bwd_ctx bwd;
    bool old_recording;
    gd_status st = GD_OK;
    uint32_t i;
    if (ctx == NULL || n_outputs == 0U || outputs == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_context_in_scope(ctx)) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "backward requires active scope");
    }
    state = gd_context_autograd(ctx);
    if (state == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    old_recording = state->recording;
    state->recording = false;
    bwd.ctx = ctx;
    bwd.tape = state;

    for (i = 0U; i < n_outputs; ++i) {
        const gd_tensor *grad = grad_outputs != NULL ? grad_outputs[i] : NULL;
        st = gd_seed_output_grad(&bwd, outputs[i], grad);
        if (st != GD_OK) {
            goto done;
        }
    }

    for (i = state->n_nodes; i > 0U; --i) {
        const gd_tape_node *node = &state->nodes[i - 1U];
        const gd_autograd_rule *rule;
        if (!gd_node_has_output_grad(state, node)) {
            continue;
        }
        rule = gd_autograd_rule_for(node->op);
        if (rule == NULL || rule->backward == NULL) {
            st = gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "missing autograd rule");
            goto done;
        }
        st = rule->backward(&bwd, node);
        if (st != GD_OK) {
            goto done;
        }
    }

done:
    state->recording = old_recording;
    return st;
}

gd_status gd_backward(gd_context *ctx,
                      const gd_tensor *output,
                      const gd_tensor *grad_output)
{
    const gd_tensor *outputs[1];
    const gd_tensor *grad_outputs[1];
    outputs[0] = output;
    grad_outputs[0] = grad_output;
    return gd_backward_many(ctx, 1U, outputs, grad_outputs);
}

gd_status gd_tensor_grad(gd_context *ctx,
                         const gd_tensor *tensor,
                         gd_tensor *out_grad)
{
    const gd_autograd_state *state;
    const gd_grad_slot *slot;
    if (out_grad != NULL) {
        memset(out_grad, 0, sizeof(*out_grad));
    }
    if (ctx == NULL || tensor == NULL || out_grad == NULL || tensor->id == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    state = gd_context_autograd_const(ctx);
    slot = gd_find_grad_slot_const(state, tensor->id);
    if (slot == NULL) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "tensor has no accumulated gradient");
    }
    *out_grad = slot->grad;
    return GD_OK;
}

gd_status gd_zero_grad(gd_context *ctx)
{
    gd_autograd_state *state;
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    state = gd_context_autograd(ctx);
    if (state == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    state->n_grads = 0U;
    return GD_OK;
}

gd_status gd_set_grad_enabled(gd_context *ctx, bool enabled)
{
    gd_autograd_state *state;
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    state = gd_context_autograd(ctx);
    if (state == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    state->user_enabled = enabled;
    state->recording = enabled && gd_context_scope_mode(ctx) == GD_SCOPE_TRAIN;
    return GD_OK;
}

bool gd_is_grad_enabled(const gd_context *ctx)
{
    const gd_autograd_state *state = gd_context_autograd_const(ctx);
    return state != NULL && state->recording;
}
