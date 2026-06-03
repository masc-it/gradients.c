#include "gradients/graph.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core/internal.h"
#include "graph_internal.h"

/* Default tolerances applied when the caller passes opts == NULL. */
#define _GD_COMPARE_DEFAULT_ATOL 1e-5
#define _GD_COMPARE_DEFAULT_RTOL 1e-5

/* Host-side copy of one compiled value, downloaded after a backend run. */
typedef struct value_snapshot {
    bool present;          /* value was downloaded (skipped externals stay false) */
    void *data;            /* contiguous host bytes, length == nbytes */
    size_t nbytes;
    gd_tensor_desc desc;   /* logical shape/dtype of the value */
} value_snapshot;

static void free_snapshots(value_snapshot *snaps, int n)
{
    int i = 0;

    if (snaps == NULL) {
        return;
    }
    for (i = 0; i < n; ++i) {
        free(snaps[i].data);
    }
    free(snaps);
}

/* Returns true when this value should participate in the comparison. External
 * leaf values share the same input bytes across both runs, so they are skipped
 * unless the caller explicitly asks for them. */
static bool value_is_comparable(const gd_graph *graph, int value_id, bool compare_externals)
{
    const _gd_value *value = &graph->values[value_id];

    if (value->producer_node_id < 0) {
        return compare_externals;
    }
    return true;
}

/* Compile + run + synchronize the graph on `device`, then download every
 * comparable value into freshly allocated host buffers. */
static gd_status run_and_snapshot(gd_graph *graph,
                                  gd_device device,
                                  bool compare_externals,
                                  value_snapshot **snaps_out)
{
    gd_status status = GD_OK;
    value_snapshot *snaps = NULL;
    int n = graph->n_values;
    int i = 0;

    *snaps_out = NULL;

    /* compare() may be invoked on an already-compiled graph, and the second
     * backend pass recompiles after the first. gd_graph_compile requires a
     * finalized graph, so re-finalize before each pass. */
    graph->state = _GD_GRAPH_FINALIZED;
    status = gd_graph_compile(graph, device);
    if (status != GD_OK) {
        return status;
    }
    status = gd_graph_run(graph);
    if (status != GD_OK) {
        return status;
    }
    /* P4 contract: results are only guaranteed after synchronize (downloads are
     * blocking too, but synchronize makes the intent explicit). */
    status = gd_synchronize(graph->ctx, device);
    if (status != GD_OK) {
        return status;
    }

    snaps = calloc((size_t)(n > 0 ? n : 1), sizeof(*snaps));
    if (snaps == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate compare snapshots");
    }

    for (i = 0; i < n; ++i) {
        gd_storage *storage = NULL;
        size_t offset = 0U;
        const gd_tensor_desc *desc = NULL;
        size_t nbytes = 0U;
        size_t alignment = 0U;
        void *buf = NULL;

        if (!value_is_comparable(graph, i, compare_externals)) {
            continue;
        }
        status = _gd_graph_value_storage(graph, i, true, &storage, &offset, &desc);
        if (status != GD_OK) {
            free_snapshots(snaps, n);
            return status;
        }
        status = gd_tensor_desc_nbytes(desc, &nbytes, &alignment);
        if (status != GD_OK) {
            free_snapshots(snaps, n);
            return status;
        }
        if (nbytes == 0U) {
            continue;
        }
        buf = malloc(nbytes);
        if (buf == NULL) {
            free_snapshots(snaps, n);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate snapshot buffer");
        }
        status = gd_storage_copy_to_cpu(graph->ctx, storage, offset, buf, nbytes);
        if (status != GD_OK) {
            free(buf);
            free_snapshots(snaps, n);
            return status;
        }
        snaps[i].present = true;
        snaps[i].data = buf;
        snaps[i].nbytes = nbytes;
        snaps[i].desc = *desc;
    }

    *snaps_out = snaps;
    return GD_OK;
}

/* Row-major coordinate of a flat element index, formatted as "[i,j,...]". */
static void format_coord(const gd_tensor_desc *desc, int64_t flat, char *out, size_t out_sz)
{
    int64_t coord[GD_MAX_DIMS];
    size_t pos = 0U;
    int d = 0;
    int n = 0;

    for (d = desc->ndim - 1; d >= 0; --d) {
        int64_t dim = desc->sizes[d] > 0 ? desc->sizes[d] : 1;
        coord[d] = flat % dim;
        flat /= dim;
    }
    n = snprintf(out + pos, out_sz - pos, "[");
    if (n > 0) {
        pos += (size_t)n;
    }
    for (d = 0; d < desc->ndim && pos < out_sz; ++d) {
        n = snprintf(out + pos, out_sz - pos, "%s%lld",
                     d == 0 ? "" : ",", (long long)coord[d]);
        if (n > 0) {
            pos += (size_t)n;
        }
    }
    if (pos < out_sz) {
        (void)snprintf(out + pos, out_sz - pos, "]");
    }
}

/* Builds the human-readable mismatch report into gd_last_error. */
static gd_status report_mismatch(const gd_graph *graph,
                                 int value_id,
                                 int64_t first_bad,
                                 const gd_tensor_desc *desc,
                                 double ref_val,
                                 double tgt_val,
                                 double max_abs,
                                 double max_rel)
{
    const _gd_value *value = &graph->values[value_id];
    int node_id = value->producer_node_id;
    const char *op = "external";
    char coord[128];
    char msg[320];

    if (node_id >= 0 && node_id < graph->n_nodes) {
        op = _gd_op_kind_name(graph->nodes[node_id].op);
    }
    format_coord(desc, first_bad, coord, sizeof(coord));
    (void)snprintf(msg, sizeof(msg),
                   "parity mismatch at value %d (op '%s' node %d) coord %s: "
                   "ref=%g target=%g; value max abs=%g rel=%g",
                   value_id, op, node_id, coord,
                   ref_val, tgt_val, max_abs, max_rel);
    return _gd_error(GD_ERR_BACKEND, msg);
}

/* Compares one F32 value with absolute/relative tolerance. */
static gd_status compare_f32_value(const gd_graph *graph,
                                   int value_id,
                                   const value_snapshot *ref,
                                   const value_snapshot *tgt,
                                   double atol,
                                   double rtol)
{
    const float *a = (const float *)ref->data;
    const float *b = (const float *)tgt->data;
    int64_t numel = (int64_t)(ref->nbytes / sizeof(float));
    int64_t first_bad = -1;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double bad_ref = 0.0;
    double bad_tgt = 0.0;
    int64_t i = 0;

    for (i = 0; i < numel; ++i) {
        double av = (double)a[i];
        double bv = (double)b[i];
        double abs_err = fabs(av - bv);
        double denom = fabs(av);
        double rel_err = denom > 0.0 ? abs_err / denom : abs_err;

        if (abs_err > atol + rtol * denom) {
            if (first_bad < 0) {
                first_bad = i;
                bad_ref = av;
                bad_tgt = bv;
            }
            if (abs_err > max_abs) {
                max_abs = abs_err;
            }
            if (rel_err > max_rel) {
                max_rel = rel_err;
            }
        }
    }
    if (first_bad < 0) {
        return GD_OK;
    }
    return report_mismatch(graph, value_id, first_bad, &ref->desc,
                           bad_ref, bad_tgt, max_abs, max_rel);
}

/* Byte-exact comparison for non-F32 values (integer/cast outputs). */
static gd_status compare_bytes_value(const gd_graph *graph,
                                     int value_id,
                                     const value_snapshot *ref,
                                     const value_snapshot *tgt)
{
    const unsigned char *a = (const unsigned char *)ref->data;
    const unsigned char *b = (const unsigned char *)tgt->data;
    size_t elem = gd_dtype_sizeof(ref->desc.dtype);
    size_t i = 0;

    for (i = 0; i < ref->nbytes; ++i) {
        if (a[i] != b[i]) {
            int64_t flat = elem > 0U ? (int64_t)(i / elem) : (int64_t)i;
            return report_mismatch(graph, value_id, flat, &ref->desc,
                                   (double)a[i], (double)b[i], 1.0, 1.0);
        }
    }
    return GD_OK;
}

gd_status gd_graph_compare(gd_graph *graph,
                           gd_device reference,
                           gd_device target,
                           const gd_compare_options *opts)
{
    gd_status status = GD_OK;
    value_snapshot *ref_snaps = NULL;
    value_snapshot *tgt_snaps = NULL;
    double atol = _GD_COMPARE_DEFAULT_ATOL;
    double rtol = _GD_COMPARE_DEFAULT_RTOL;
    bool compare_externals = false;
    bool preserve_old = false;
    int n = 0;
    int i = 0;

    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_compare graph is NULL");
    }
    if (graph->state != _GD_GRAPH_FINALIZED && graph->state != _GD_GRAPH_COMPILED) {
        return _gd_error(GD_ERR_INVALID_STATE,
                         "gd_graph_compare requires a finalized graph");
    }
    if (opts != NULL) {
        atol = opts->atol;
        rtol = opts->rtol;
        compare_externals = opts->compare_externals;
    }

    n = graph->n_values;
    preserve_old = graph->preserve_all_values;
    graph->preserve_all_values = true;

    status = run_and_snapshot(graph, reference, compare_externals, &ref_snaps);
    if (status != GD_OK) {
        graph->preserve_all_values = preserve_old;
        return status;
    }
    status = run_and_snapshot(graph, target, compare_externals, &tgt_snaps);
    if (status != GD_OK) {
        graph->preserve_all_values = preserve_old;
        free_snapshots(ref_snaps, n);
        return status;
    }

    for (i = 0; i < n; ++i) {
        const value_snapshot *r = &ref_snaps[i];
        const value_snapshot *t = &tgt_snaps[i];

        if (!r->present || !t->present) {
            continue;
        }
        if (r->nbytes != t->nbytes || r->desc.dtype != t->desc.dtype) {
            char msg[160];
            (void)snprintf(msg, sizeof(msg),
                           "value %d differs structurally between backends "
                           "(ref %zu bytes dtype %s, target %zu bytes dtype %s)",
                           i, r->nbytes, gd_dtype_name(r->desc.dtype),
                           t->nbytes, gd_dtype_name(t->desc.dtype));
            status = _gd_error(GD_ERR_BACKEND, msg);
            break;
        }
        if (r->desc.dtype == GD_DTYPE_F32) {
            status = compare_f32_value(graph, i, r, t, atol, rtol);
        } else {
            status = compare_bytes_value(graph, i, r, t);
        }
        if (status != GD_OK) {
            break;
        }
    }

    free_snapshots(ref_snaps, n);
    free_snapshots(tgt_snaps, n);
    graph->preserve_all_values = preserve_old;
    if (status == GD_OK) {
        _gd_set_last_error(GD_OK, NULL);
    }
    return status;
}
