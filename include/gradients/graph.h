#ifndef GRADIENTS_GRAPH_H
#define GRADIENTS_GRAPH_H

#include "gradients/context.h"
#include "gradients/device.h"
#include "gradients/status.h"
#include "gradients/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_graph gd_graph;
typedef gd_status (*gd_immediate_build_fn)(gd_context *ctx, void *user);

typedef enum gd_dump_format {
    GD_DUMP_TEXT = 0,
    GD_DUMP_DOT,
    GD_DUMP_JSON
} gd_dump_format;

gd_status gd_graph_create(gd_context *ctx, gd_graph **out);
gd_status gd_graph_destroy(gd_graph *graph);
gd_status gd_graph_begin(gd_context *ctx, gd_graph *graph);
gd_status gd_graph_end(gd_context *ctx);
gd_status gd_graph_validate(gd_graph *graph);
gd_status gd_graph_compile(gd_graph *graph, gd_device target);
gd_status gd_graph_run(gd_graph *graph);
gd_status gd_graph_reset(gd_graph *graph);
gd_status gd_graph_dump(gd_graph *graph, gd_dump_format format, const char *path);

gd_status gd_graph_run_immediate(gd_context *ctx,
                                 gd_device target,
                                 gd_immediate_build_fn build,
                                 void *user);

gd_status gd_graph_run_until(gd_graph *graph, int node_id);

/* Backend parity harness: run the same finalized graph on `reference` and
 * `target`, then compare every produced value within tolerance. Reports the
 * first mismatch (value id, producing op/node, coordinate, errors) through
 * gd_last_error and returns GD_ERR_BACKEND. Returns GD_OK when all compared
 * values agree.
 *
 * v1 assumes a side-effect-free (forward) graph: the graph is executed twice,
 * so graphs that mutate external tensors in place (e.g. optimizer steps) are
 * not yet supported. Leaves the graph compiled on `target` on success. */
typedef struct gd_compare_options {
    double atol;            /* absolute float tolerance (default 1e-5) */
    double rtol;            /* relative float tolerance (default 1e-5) */
    bool compare_externals; /* also compare external/leaf values (default false) */
} gd_compare_options;

gd_status gd_graph_compare(gd_graph *graph,
                           gd_device reference,
                           gd_device target,
                           const gd_compare_options *opts);

gd_status gd_scope_push(gd_context *ctx, const char *name);
gd_status gd_scope_pop(gd_context *ctx);
gd_status gd_tensor_set_name(gd_tensor *tensor, const char *name);

gd_status gd_tensor_materialize(gd_context *ctx, gd_tensor *tensor);
gd_status gd_tensor_to_cpu(gd_context *ctx,
                           gd_tensor *tensor,
                           void *dst,
                           size_t nbytes);
gd_status gd_debug_print_tensor(gd_context *ctx, gd_tensor *tensor, int max_elems);
gd_status gd_assert_finite(gd_context *ctx, gd_tensor *tensor);
gd_status gd_assert_close(gd_context *ctx,
                          gd_tensor *a,
                          gd_tensor *b,
                          float atol,
                          float rtol);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_GRAPH_H */
