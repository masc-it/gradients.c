#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <stdint.h>

static uint32_t dropout_hash(uint64_t seed, uint64_t run_id, uint64_t index)
{
    uint32_t h = (uint32_t)seed ^ ((uint32_t)(seed >> 32) * UINT32_C(0x9e3779b9));

    h ^= (uint32_t)run_id * UINT32_C(0x85ebca6b);
    h ^= (uint32_t)(run_id >> 32) * UINT32_C(0xc2b2ae35);
    h ^= (uint32_t)index * UINT32_C(0x27d4eb2d);
    h ^= (uint32_t)(index >> 32) * UINT32_C(0x165667b1);
    h ^= h >> 16;
    h *= UINT32_C(0x7feb352d);
    h ^= h >> 15;
    h *= UINT32_C(0x846ca68b);
    h ^= h >> 16;
    return h;
}

static int dropout_keep(uint64_t seed, uint64_t run_id, uint64_t index, float p)
{
    uint32_t r = dropout_hash(seed, run_id, index);
    float u = ((float)(r >> 8) + 0.5F) * (1.0F / 16777216.0F);

    return u >= p;
}

gd_status _gd_cpu_k_dropout(const gd_tensor_desc *desc,
                            void *out,
                            const void *x,
                            float p,
                            uint64_t seed,
                            uint64_t run_id)
{
    int64_t total = _gd_cpu_desc_numel(desc);
    float keep_scale = 1.0F / (1.0F - p);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float v = 0.0F;
        float y = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, x, i, &v);
        if (status != GD_OK) {
            return status;
        }
        y = dropout_keep(seed, run_id, (uint64_t)i, p) ? v * keep_scale : 0.0F;
        status = _gd_cpu_store_float(desc, out, i, y);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_dropout_bwd(const gd_tensor_desc *desc,
                                void *dx,
                                const void *go,
                                float p,
                                uint64_t seed,
                                uint64_t run_id)
{
    int64_t total = _gd_cpu_desc_numel(desc);
    float keep_scale = 1.0F / (1.0F - p);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float v = 0.0F;
        float y = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, go, i, &v);
        if (status != GD_OK) {
            return status;
        }
        y = dropout_keep(seed, run_id, (uint64_t)i, p) ? v * keep_scale : 0.0F;
        status = _gd_cpu_store_float(desc, dx, i, y);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}
