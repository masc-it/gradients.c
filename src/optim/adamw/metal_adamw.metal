#include <metal_stdlib>
#include "metal_adamw_types.h"

using namespace metal;

#include "backends/metal/metal_common.h"

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

    const ulong p_byte = args.param_offset + i * gd_dtype_elem_size(args.param_dtype);
    const ulong g_byte = args.grad_offset + i * gd_dtype_elem_size(args.grad_dtype);
    device float *m = reinterpret_cast<device float *>(m_buf + args.m_offset);
    device float *v = reinterpret_cast<device float *>(v_buf + args.v_offset);
    device float *master = reinterpret_cast<device float *>(master_buf + args.master_offset);

    const float g = gd_load_float_dtype(grad, g_byte, args.grad_dtype);
    const float p = args.has_master != 0u ? master[i] : gd_load_float_dtype(param, p_byte, args.param_dtype);
    const float m_new = args.beta1 * m[i] + (1.0f - args.beta1) * g;
    const float v_new = args.beta2 * v[i] + (1.0f - args.beta2) * g * g;
    const float m_hat = m_new / args.bias_correction1;
    const float v_hat = v_new / args.bias_correction2;
    const float update = m_hat / (sqrt(v_hat) + args.eps);
    const float p_new = p - args.lr * (update + args.weight_decay * p);

    m[i] = m_new;
    v[i] = v_new;
    if (args.has_master != 0u) {
        master[i] = p_new;
    }
    gd_write_float_dtype(param, p_byte, args.param_dtype, p_new);
}
