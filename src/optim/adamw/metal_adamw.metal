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

kernel void gd_adamw_kernel(device uchar *param [[buffer(0)]],
                            device const uchar *grad [[buffer(1)]],
                            device uchar *m_buf [[buffer(2)]],
                            device uchar *v_buf [[buffer(3)]],
                            device uchar *master_buf [[buffer(4)]],
                            constant gd_metal_adamw_args &args [[buffer(5)]],
                            uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }

    device float *m = reinterpret_cast<device float *>(m_buf + args.m_offset);
    device float *v = reinterpret_cast<device float *>(v_buf + args.v_offset);

    if (args.has_master != 0u && args.param_dtype == 1u) {
        device half *param_h = reinterpret_cast<device half *>(param + args.param_offset);
        device float *master = reinterpret_cast<device float *>(master_buf + args.master_offset);
        const float g = args.grad_dtype == 1u
                            ? float(reinterpret_cast<device const half *>(grad + args.grad_offset)[i])
                            : reinterpret_cast<device const float *>(grad + args.grad_offset)[i];
        const float p_new = gd_adamw_apply(master[i], g, m, v, i, args);
        master[i] = p_new;
        param_h[i] = half(p_new);
        return;
    }

    if (args.has_master == 0u && args.param_dtype == 3u) {
        device float *param_f = reinterpret_cast<device float *>(param + args.param_offset);
        const float g = args.grad_dtype == 1u
                            ? float(reinterpret_cast<device const half *>(grad + args.grad_offset)[i])
                            : reinterpret_cast<device const float *>(grad + args.grad_offset)[i];
        param_f[i] = gd_adamw_apply(param_f[i], g, m, v, i, args);
        return;
    }

    const ulong p_byte = args.param_offset + i * gd_dtype_elem_size(args.param_dtype);
    const ulong g_byte = args.grad_offset + i * gd_dtype_elem_size(args.grad_dtype);
    device float *master = reinterpret_cast<device float *>(master_buf + args.master_offset);

    const float g = gd_load_float_dtype(grad, g_byte, args.grad_dtype);
    const float p = args.has_master != 0u ? master[i] : gd_load_float_dtype(param, p_byte, args.param_dtype);
    const float p_new = gd_adamw_apply(p, g, m, v, i, args);
    if (args.has_master != 0u) {
        master[i] = p_new;
    }
    gd_write_float_dtype(param, p_byte, args.param_dtype, p_new);
}
