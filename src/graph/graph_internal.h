#ifndef GRADIENTS_GRAPH_INTERNAL_H
#define GRADIENTS_GRAPH_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "gradients/dtype.h"
#include "gradients/graph.h"
#include "gradients/tensor.h"
#include "op_kind.h"

#define _GD_OP_MAX_INPUTS 256

typedef enum _gd_graph_state {
    _GD_GRAPH_EMPTY = 0,
    _GD_GRAPH_BUILDING,
    _GD_GRAPH_FINALIZED,
    _GD_GRAPH_COMPILED
} _gd_graph_state;

typedef struct _gd_op_attrs {
    float scale;                 /* SCALE / CLIP_GRAD_NORM max_norm */
    int dim;                     /* SUM / MEAN / SOFTMAX / CROSS_ENTROPY class dim */
    bool keepdim;                /* SUM / MEAN */
    bool trans_a;                /* MATMUL */
    bool trans_b;                /* MATMUL / LINEAR (trans_w) */
    bool has_bias;               /* LINEAR */
    float eps;                   /* RMS_NORM / ADAMW */
    float powlu_m;               /* POWLU */
    gd_dtype cast_dtype;         /* CAST */
    bool gelu_tanh;              /* GELU: tanh approximation vs exact erf */
    int perm[GD_MAX_DIMS];      /* TRANSPOSE: output axis permutation */
    int perm_ndim;              /* TRANSPOSE: rank of perm */
    float rope_theta;           /* ROPE: base frequency */
    int rope_n_dims;            /* ROPE: rotary dims (resolved, even, <= head_dim) */
    int rope_interleaved;       /* ROPE: GPT-J interleaved vs NeoX half-split */
    float attn_scale;           /* SDPA: softmax scale (resolved; 1/sqrt(head_dim)) */
    int n_q_heads;              /* SDPA: query heads */
    int n_kv_heads;             /* SDPA: key/value heads (GQA; Hq % Hkv == 0) */
    int head_dim;               /* SDPA */
    int causal;                 /* SDPA: causal masking */
    int sliding_window;         /* SDPA: window size (0 = none) */
    int prefix_len;             /* SDPA: bidirectional prefix length for causal mask */
    gd_compute_policy compute;   /* MATMUL / LINEAR */
    float lr;                    /* ADAMW */
    float beta1;                 /* ADAMW */
    float beta2;                 /* ADAMW */
    float weight_decay;          /* ADAMW */
    float atol;                  /* ASSERT_CLOSE */
    float rtol;                  /* ASSERT_CLOSE */
    bool has_reduce_to_desc;      /* REDUCE_TO: target descriptor present */
    gd_tensor_desc reduce_to_desc; /* REDUCE_TO target output descriptor */
} _gd_op_attrs;

typedef struct _gd_value {
    int id;
    int producer_node_id;        /* -1 for external leaf values */
    gd_tensor_desc desc;
    gd_tensor *external;          /* retained leaf tensor, NULL for produced values */
    char *name;                   /* optional debug name */
} _gd_value;

typedef struct _gd_node {
    int id;
    _gd_op_kind op;
    int *inputs;
    int n_inputs;
    int *outputs;
    int n_outputs;
    _gd_op_attrs attrs;
    char *scope;
    char *name;
} _gd_node;

typedef struct _gd_backend _gd_backend;
typedef struct _gd_executable _gd_executable;

struct gd_graph {
    gd_context *ctx;
    _gd_graph_state state;
    gd_device target;
    bool has_target;
    bool has_run;
    _gd_node *nodes;
    int n_nodes;
    int node_cap;
    _gd_value *values;
    int n_values;
    int value_cap;
    gd_tensor **virtual_tensors;  /* weak handles to live virtual output tensors */
    int n_virtual;
    int virtual_cap;
    _gd_backend *backend;         /* selected at compile time */
    _gd_executable *exec;         /* backend-owned compiled artifact */
};

const char *_gd_graph_state_name(_gd_graph_state state);
const char *_gd_op_kind_name(_gd_op_kind op);

gd_status _gd_graph_clear(gd_graph *graph);
gd_status _gd_graph_note_virtual_tensor_create(gd_graph *graph, gd_tensor *tensor);
void _gd_graph_note_virtual_tensor_release(gd_graph *graph, gd_tensor *tensor);

/* Records a single-output op node into `graph`, importing each input tensor as a
 * graph value, creating the output value from `out_desc`, and returning a new
 * virtual output tensor handle (owned by the caller, refcount 1). */
gd_status _gd_graph_emit(gd_graph *graph,
                         _gd_op_kind op,
                         gd_tensor **inputs,
                         int n_inputs,
                         const _gd_op_attrs *attrs,
                         const gd_tensor_desc *out_desc,
                         gd_tensor **out_tensor);

/* Records a multi-output op node: creates `n_outputs` produced values and
 * returns a new virtual output tensor for each (each owned by the caller,
 * refcount 1). Used by ops whose backward yields several gradients (e.g. SDPA
 * -> dq, dk, dv). */
gd_status _gd_graph_emit_multi(gd_graph *graph,
                               _gd_op_kind op,
                               gd_tensor **inputs,
                               int n_inputs,
                               const _gd_op_attrs *attrs,
                               const gd_tensor_desc *out_descs,
                               int n_outputs,
                               gd_tensor **out_tensors);

/* Records a node whose single output is bound to an existing materialized
 * tensor `out_external` (used as a write target, e.g. a leaf gradient slot). */
gd_status _gd_graph_emit_to(gd_graph *graph,
                            _gd_op_kind op,
                            gd_tensor **inputs,
                            int n_inputs,
                            const _gd_op_attrs *attrs,
                            gd_tensor *out_external);

/* Records an in-place op node (no produced value) that mutates one or more of
 * its materialized input tensors directly, e.g. an optimizer step. */
gd_status _gd_graph_emit_inplace(gd_graph *graph,
                                 _gd_op_kind op,
                                 gd_tensor **inputs,
                                 int n_inputs,
                                 const _gd_op_attrs *attrs);

/* Backend-routed access: returns the value's storage + byte offset (no host
 * pointer assumption), for transfers/materialization. */
gd_status _gd_graph_value_storage(gd_graph *graph,
                                  int value_id,
                                  bool require_run,
                                  gd_storage **storage_out,
                                  size_t *offset_out,
                                  const gd_tensor_desc **desc_out);
gd_status _gd_graph_set_value_name(gd_graph *graph, int value_id, const char *name);
gd_status _gd_graph_materialize_live_virtuals(gd_graph *graph);

gd_status _gd_graph_dump_text(gd_graph *graph, const char *path);

#endif /* GRADIENTS_GRAPH_INTERNAL_H */
