#include "metal_common.metal"

kernel void gd_adamw_amp(device float *param                 [[buffer(0)]],
                         device const float *grad            [[buffer(1)]],
                         device float *m                     [[buffer(2)]],
                         device float *v                     [[buffer(3)]],
                         device const float *step            [[buffer(4)]],
                         device const atomic_uint *found_inf [[buffer(5)]],
                         constant gd_metal_adamw_params &p   [[buffer(6)]],
                         device const float *lr_tensor       [[buffer(7)]],
                         uint gid                            [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel || atomic_load_explicit(found_inf, memory_order_relaxed) != 0u) {
        return;
    }
    float lr = (p.use_lr_tensor != 0 ? lr_tensor[0] : p.lr) * p.lr_scale;
    float t = step[0];
    float bc1 = 1.0f - pow(p.beta1, t);
    float bc2 = 1.0f - pow(p.beta2, t);
    float g = grad[gid];
    float mi = p.beta1 * m[gid] + (1.0f - p.beta1) * g;
    float vi = p.beta2 * v[gid] + (1.0f - p.beta2) * g * g;
    float mhat = mi / bc1;
    float vhat = vi / bc2;
    float pp = param[gid];

    m[gid] = mi;
    v[gid] = vi;
    pp -= lr * p.weight_decay * pp;
    pp -= lr * mhat / (sqrt(vhat) + p.eps);
    param[gid] = pp;
}
