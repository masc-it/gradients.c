#include "gradients/graph.h"

#include <stdio.h>

#include "../core/internal.h"
#include "graph_internal.h"

static gd_status dump_shape(FILE *file, const gd_tensor_desc *desc)
{
    int i = 0;

    if (file == NULL || desc == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dump_shape argument is NULL");
    }
    if (fprintf(file, "[") < 0) {
        return _gd_error(GD_ERR_IO, "failed to write graph dump");
    }
    for (i = 0; i < desc->ndim; ++i) {
        if (fprintf(file, "%s%lld", i == 0 ? "" : ",", (long long)desc->sizes[i]) < 0) {
            return _gd_error(GD_ERR_IO, "failed to write graph dump");
        }
    }
    if (fprintf(file, "]") < 0) {
        return _gd_error(GD_ERR_IO, "failed to write graph dump");
    }
    return GD_OK;
}

gd_status _gd_graph_dump_text(gd_graph *graph, const char *path)
{
    gd_status status = GD_OK;
    FILE *file = NULL;
    int i = 0;

    if (graph == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "graph dump argument is NULL");
    }

    file = fopen(path, "w");
    if (file == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open graph dump path");
    }

    if (fprintf(file,
                "graph state=%s nodes=%d values=%d live_virtual_tensors=%d\n",
                _gd_graph_state_name(graph->state),
                graph->n_nodes,
                graph->n_values,
                graph->n_virtual) < 0) {
        (void)fclose(file);
        return _gd_error(GD_ERR_IO, "failed to write graph dump");
    }

    for (i = 0; i < graph->n_nodes; ++i) {
        const _gd_node *node = &graph->nodes[i];
        if (fprintf(file,
                    "node %d op=%s inputs=%d outputs=%d",
                    node->id,
                    _gd_op_kind_name(node->op),
                    node->n_inputs,
                    node->n_outputs) < 0) {
            (void)fclose(file);
            return _gd_error(GD_ERR_IO, "failed to write graph dump");
        }
        if (node->scope != NULL && fprintf(file, " scope=%s", node->scope) < 0) {
            (void)fclose(file);
            return _gd_error(GD_ERR_IO, "failed to write graph dump");
        }
        if (node->name != NULL && fprintf(file, " name=%s", node->name) < 0) {
            (void)fclose(file);
            return _gd_error(GD_ERR_IO, "failed to write graph dump");
        }
        if (fprintf(file, "\n") < 0) {
            (void)fclose(file);
            return _gd_error(GD_ERR_IO, "failed to write graph dump");
        }
    }

    for (i = 0; i < graph->n_values; ++i) {
        const _gd_value *value = &graph->values[i];
        int leaf = value->external != NULL;
        int requires_grad = leaf && gd_tensor_requires_grad(value->external);

        if (fprintf(file,
                    "value %d name=%s producer=%d kind=%s requires_grad=%d dtype=%s device=%s:%d shape=",
                    value->id,
                    value->name != NULL ? value->name : "-",
                    value->producer_node_id,
                    leaf ? "leaf" : "produced",
                    requires_grad,
                    gd_dtype_name(value->desc.dtype),
                    gd_device_type_name(value->desc.device.type),
                    value->desc.device.index) < 0) {
            (void)fclose(file);
            return _gd_error(GD_ERR_IO, "failed to write graph dump");
        }
        status = dump_shape(file, &value->desc);
        if (status != GD_OK) {
            (void)fclose(file);
            return status;
        }
        if (fprintf(file, " layout=%d\n", (int)value->desc.layout) < 0) {
            (void)fclose(file);
            return _gd_error(GD_ERR_IO, "failed to write graph dump");
        }
    }

    if (fclose(file) != 0) {
        return _gd_error(GD_ERR_IO, "failed to close graph dump");
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_graph_dump(gd_graph *graph, gd_dump_format format, const char *path)
{
    if (graph == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_dump argument is NULL");
    }
    if (format != GD_DUMP_TEXT) {
        return _gd_error(GD_ERR_UNSUPPORTED, "only text graph dump is implemented yet");
    }
    return _gd_graph_dump_text(graph, path);
}
