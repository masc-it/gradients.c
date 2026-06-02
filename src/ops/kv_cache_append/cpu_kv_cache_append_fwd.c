#include "../../backends/cpu_ref/cpu_op.h"

#include <stdint.h>
#include <string.h>

static int64_t kv_cache_pos_value(const gd_tensor_desc *pos_desc, const void *pos)
{
    return pos_desc->dtype == GD_DTYPE_I64 ? ((const int64_t *)pos)[0]
                                          : (int64_t)((const int32_t *)pos)[0];
}

static gd_status kv_cache_append_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *k_cache = NULL;
    void *v_cache = NULL;
    void *pos = NULL;
    void *k_new = NULL;
    void *v_new = NULL;
    const gd_tensor_desc *kc = NULL;
    const gd_tensor_desc *vc = NULL;
    const gd_tensor_desc *pd = NULL;
    const gd_tensor_desc *kn = NULL;
    const gd_tensor_desc *vn = NULL;
    int64_t B = 0;
    int64_t Tmax = 0;
    int64_t Tnew = 0;
    int64_t row_elems = 0;
    int64_t p = 0;
    int64_t b = 0;
    int64_t t = 0;
    size_t elem = 0U;
    size_t row_bytes = 0U;

    status = _gd_cpu_exec_input(exec, node, 0, &k_cache, &kc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 1, &v_cache, &vc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 2, &pos, &pd);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 3, &k_new, &kn);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 4, &v_new, &vn);
    if (status != GD_OK) { return status; }
    (void)vc;
    (void)vn;

    B = kc->sizes[0];
    Tmax = kc->sizes[1];
    Tnew = kn->sizes[1];
    row_elems = kc->sizes[2] * kc->sizes[3];
    elem = gd_dtype_sizeof(kc->dtype);
    row_bytes = (size_t)row_elems * elem;
    p = kv_cache_pos_value(pd, pos);
    if (p < 0 || Tnew < 0 || p > Tmax || Tnew > Tmax - p) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "kv_cache_append position exceeds cache length");
    }
    for (b = 0; b < B; ++b) {
        for (t = 0; t < Tnew; ++t) {
            size_t src = (size_t)((b * Tnew + t) * row_elems) * elem;
            size_t dst = (size_t)((b * Tmax + (p + t)) * row_elems) * elem;
            memcpy((unsigned char *)k_cache + dst, (const unsigned char *)k_new + src,
                   row_bytes);
            memcpy((unsigned char *)v_cache + dst, (const unsigned char *)v_new + src,
                   row_bytes);
        }
    }
    return GD_OK;
}

const _gd_cpu_op _gd_cpu_op_kv_cache_append = {
    .kind = _GD_OP_KV_CACHE_APPEND,
    .name = "kv_cache_append",
    .support = _gd_cpu_support_default,
    .run = kv_cache_append_run,
};
