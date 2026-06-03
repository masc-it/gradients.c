#include "../../backends/cpu_ref/cpu_op.h"

static gd_status embedding_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *table_data = NULL;
    void *ids_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *table_desc = NULL;
    const gd_tensor_desc *ids_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &table_data, &table_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &ids_data, &ids_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_embedding(out_desc, out_data, table_desc, table_data, ids_desc, ids_data);
}

const _gd_cpu_op _gd_cpu_op_embedding = {
    .kind = _GD_OP_EMBEDDING,
    .name = "embedding",
    .support = _gd_cpu_support_default,
    .run = embedding_run,
};
