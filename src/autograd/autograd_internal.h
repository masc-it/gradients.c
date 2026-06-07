#ifndef GD_AUTOGRAD_INTERNAL_H
#define GD_AUTOGRAD_INTERNAL_H

#include <stdbool.h>
#include <stdint.h>

#include <gradients/autograd.h>
#include <gradients/status.h>
#include <gradients/tensor.h>

#include "../ops/op_kind.h"

typedef struct gd_autograd_state gd_autograd_state;

typedef struct gd_tape_ref {
    gd_tensor tensor;
    uint64_t id;
    uint32_t version;
} gd_tape_ref;

typedef struct gd_tape_node {
    gd_op_kind op;
    uint32_t first_input;
    uint16_t n_inputs;
    uint32_t first_output;
    uint16_t n_outputs;
    uint32_t first_saved;
    uint16_t n_saved;
    uint32_t attrs_offset;
    uint32_t attrs_size;
} gd_tape_node;

typedef struct gd_grad_slot {
    uint64_t tensor_id;
    gd_tensor grad;
    bool occupied;
} gd_grad_slot;

typedef struct gd_live_span_slot {
    gd_span span;
    uint32_t refs;
} gd_live_span_slot;

struct gd_autograd_state {
    bool recording;
    bool user_enabled;
    gd_tape_node *nodes;
    uint32_t n_nodes;
    uint32_t cap_nodes;
    gd_tape_ref *refs;
    uint32_t n_refs;
    uint32_t cap_refs;
    unsigned char *attrs;
    uint32_t attrs_used;
    uint32_t attrs_cap;
    gd_grad_slot *grads;
    uint32_t n_grads;
    uint32_t cap_grads;
    gd_live_span_slot *live_spans;
    uint32_t n_live_spans;
    uint32_t cap_live_spans;
};

typedef struct gd_bwd_ctx {
    gd_context *ctx;
    gd_autograd_state *tape;
} gd_bwd_ctx;

typedef gd_status (*gd_bwd_fn)(gd_bwd_ctx *bwd, const gd_tape_node *node);

typedef struct gd_autograd_rule {
    gd_op_kind kind;
    const char *name;
    gd_bwd_fn backward;
} gd_autograd_rule;

const gd_autograd_rule *gd_autograd_rule_for(gd_op_kind kind);

gd_status gd_autograd_record(gd_context *ctx,
                             gd_op_kind op,
                             const gd_tensor *const *inputs,
                             uint16_t n_inputs,
                             gd_tensor *const *outputs,
                             uint16_t n_outputs,
                             const void *attrs,
                             uint32_t attrs_size,
                             const gd_tensor *const *saved,
                             uint16_t n_saved);

const gd_tensor *gd_tape_input(const gd_autograd_state *tape,
                               const gd_tape_node *node,
                               uint16_t index);
const gd_tensor *gd_tape_output(const gd_autograd_state *tape,
                                const gd_tape_node *node,
                                uint16_t index);
const gd_tensor *gd_tape_saved(const gd_autograd_state *tape,
                               const gd_tape_node *node,
                               uint16_t index);
const void *gd_tape_attrs(const gd_autograd_state *tape,
                          const gd_tape_node *node,
                          uint32_t expected_size);

bool gd_autograd_get_grad(gd_bwd_ctx *bwd, uint64_t tensor_id, gd_tensor *out_grad);
gd_status gd_autograd_accumulate(gd_bwd_ctx *bwd,
                                 uint64_t tensor_id,
                                 const gd_tensor *contrib);

#endif /* GD_AUTOGRAD_INTERNAL_H */
