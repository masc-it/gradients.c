#include <metal_stdlib>
#include "metal_adamw_types.h"

using namespace metal;

#include "backends/metal/metal_common.h"

static inline float gd_adamw_apply(float p,
                                   float g,
                                   device float *m,
                                   device float *v,
                                   ulong i,
                                   constant gd_metal_adamw_args &args)
{
    const float m_new = fma(args.beta1, m[i], args.one_minus_beta1 * g);
    const float v_new = fma(args.beta2, v[i], args.one_minus_beta2 * g * g);
    const float denom = sqrt(v_new) + args.eps_scaled;
    const float p_new = p - (args.step_scale * m_new) / denom - args.decay_scale * p;

    m[i] = m_new;
    v[i] = v_new;
    return p_new;
}

static inline float gd_adamw_grad_scale(device const uchar *grad_scale_buf,
                                        constant gd_metal_adamw_args &args)
{
    if (args.has_grad_scale == 0u) {
        return 1.0f;
    }
    return *(reinterpret_cast<device const float *>(grad_scale_buf + args.grad_scale_offset));
}

static inline bool gd_adamw_found_inf(device const uchar *found_inf_buf,
                                      constant gd_metal_adamw_args &args)
{
    if (args.has_found_inf == 0u) {
        return false;
    }
    return *(reinterpret_cast<device const uint *>(found_inf_buf + args.found_inf_offset)) != 0u;
}

static inline void gd_grad_norm_mark_found_inf(device uchar *found_inf_buf, ulong found_inf_offset)
{
    device atomic_uint *flag = reinterpret_cast<device atomic_uint *>(found_inf_buf + found_inf_offset);
    atomic_store_explicit(flag, 1u, memory_order_relaxed);
}

static inline float gd_adamw_scale_grad(float g, float scale)
{
    return scale == 0.0f ? 0.0f : g * scale;
}

kernel void gd_adamw_kernel(device uchar *param [[buffer(0)]],
                            device const uchar *grad [[buffer(1)]],
                            device uchar *m_buf [[buffer(2)]],
                            device uchar *v_buf [[buffer(3)]],
                            device uchar *master_buf [[buffer(4)]],
                            constant gd_metal_adamw_args &args [[buffer(5)]],
                            device const uchar *grad_scale_buf [[buffer(6)]],
                            device const uchar *found_inf_buf [[buffer(7)]],
                            uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.count || gd_adamw_found_inf(found_inf_buf, args)) {
        return;
    }

    device float *m = reinterpret_cast<device float *>(m_buf + args.m_offset);
    device float *v = reinterpret_cast<device float *>(v_buf + args.v_offset);
    const float grad_scale = gd_adamw_grad_scale(grad_scale_buf, args);

    if (args.has_master != 0u && args.param_dtype == 1u) {
        device half *param_h = reinterpret_cast<device half *>(param + args.param_offset);
        device float *master = reinterpret_cast<device float *>(master_buf + args.master_offset);
        const float g = gd_adamw_scale_grad(args.grad_dtype == 1u
                            ? float(reinterpret_cast<device const half *>(grad + args.grad_offset)[i])
                            : reinterpret_cast<device const float *>(grad + args.grad_offset)[i],
                            grad_scale);
        const float p_new = gd_adamw_apply(master[i], g, m, v, i, args);
        master[i] = p_new;
        param_h[i] = half(p_new);
        return;
    }

    if (args.has_master == 0u && args.param_dtype == 3u) {
        device float *param_f = reinterpret_cast<device float *>(param + args.param_offset);
        const float g = gd_adamw_scale_grad(args.grad_dtype == 1u
                            ? float(reinterpret_cast<device const half *>(grad + args.grad_offset)[i])
                            : reinterpret_cast<device const float *>(grad + args.grad_offset)[i],
                            grad_scale);
        param_f[i] = gd_adamw_apply(param_f[i], g, m, v, i, args);
        return;
    }

    const ulong p_byte = args.param_offset + i * gd_dtype_elem_size(args.param_dtype);
    const ulong g_byte = args.grad_offset + i * gd_dtype_elem_size(args.grad_dtype);
    device float *master = reinterpret_cast<device float *>(master_buf + args.master_offset);

    const float g = gd_adamw_scale_grad(gd_load_float_dtype(grad, g_byte, args.grad_dtype), grad_scale);
    const float p = args.has_master != 0u ? master[i] : gd_load_float_dtype(param, p_byte, args.param_dtype);
    const float p_new = gd_adamw_apply(p, g, m, v, i, args);
    if (args.has_master != 0u) {
        master[i] = p_new;
    }
    gd_write_float_dtype(param, p_byte, args.param_dtype, p_new);
}

static inline float gd_grad_norm_load(device const uchar *grad,
                                      ulong i,
                                      constant gd_metal_grad_norm_stage_args &args)
{
    float g;
    if (args.grad_dtype == 1u) {
        g = float(reinterpret_cast<device const half *>(grad + args.grad_offset)[i]);
    } else if (args.grad_dtype == 3u) {
        g = reinterpret_cast<device const float *>(grad + args.grad_offset)[i];
    } else {
        const ulong byte = args.grad_offset + i * gd_dtype_elem_size(args.grad_dtype);
        g = gd_load_float_dtype(grad, byte, args.grad_dtype);
    }
    return g * args.grad_scale;
}

kernel void gd_grad_norm_stage_kernel(device const uchar *grad [[buffer(0)]],
                                      device uchar *partial_buf [[buffer(1)]],
                                      constant gd_metal_grad_norm_stage_args &args [[buffer(2)]],
                                      device uchar *found_inf_buf [[buffer(3)]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint tg [[threadgroup_position_in_grid]])
{
    threadgroup float sums[GD_METAL_GRAD_NORM_THREADS];
    const ulong block_start = ulong(tg) * ulong(GD_METAL_GRAD_NORM_BLOCK_ELEMS);
    float sum = 0.0f;

    for (uint j = tid; j < GD_METAL_GRAD_NORM_BLOCK_ELEMS; j += GD_METAL_GRAD_NORM_THREADS) {
        const ulong i = block_start + ulong(j);
        if (i < args.count) {
            const float g = gd_grad_norm_load(grad, i, args);
            if (!isfinite(g) && args.has_found_inf != 0u) {
                gd_grad_norm_mark_found_inf(found_inf_buf, args.found_inf_offset);
            }
            sum = fma(g, g, sum);
        }
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GD_METAL_GRAD_NORM_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        device float *partials = reinterpret_cast<device float *>(partial_buf + args.partial_offset);
        partials[tg] = sums[0];
    }
}

kernel void gd_grad_clip_finalize_kernel(device const uchar *partial_buf [[buffer(0)]],
                                         device uchar *scale_buf [[buffer(1)]],
                                         constant gd_metal_grad_clip_finalize_args &args [[buffer(2)]],
                                         uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float sums[GD_METAL_GRAD_NORM_THREADS];
    device const float *partials = reinterpret_cast<device const float *>(partial_buf + args.partial_offset);
    float sum = 0.0f;

    for (ulong i = ulong(tid); i < args.partial_count; i += ulong(GD_METAL_GRAD_NORM_THREADS)) {
        sum += partials[i];
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GD_METAL_GRAD_NORM_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        const float total_norm = sqrt(sums[0]);
        float scale = args.max_norm / (total_norm + args.eps);
        if (!isfinite(total_norm)) {
            scale = 0.0f;
        } else if (!(scale < 1.0f)) {
            scale = 1.0f;
        }
        device float *out = reinterpret_cast<device float *>(scale_buf + args.scale_offset);
        out[0] = scale * args.grad_scale;
        out[1] = total_norm;
    }
}
