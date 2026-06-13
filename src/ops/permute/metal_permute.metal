#include <metal_stdlib>
#include "metal_permute_types.h"

using namespace metal;

#define GD_PERMUTE_DEFINE_KERNEL(SUFFIX, TYPE)                                        \
kernel void gd_permute_##SUFFIX##_kernel(device const uchar *src_bytes [[buffer(0)]], \
                                          device uchar *dst_bytes [[buffer(1)]],       \
                                          constant gd_metal_permute_args &args [[buffer(2)]], \
                                          uint gid [[thread_position_in_grid]])        \
{                                                                                     \
    ulong i = ulong(gid);                                                            \
    if (i >= args.count) {                                                           \
        return;                                                                      \
    }                                                                                \
    device const TYPE *src = reinterpret_cast<device const TYPE *>(src_bytes + args.src_offset); \
    device TYPE *dst = reinterpret_cast<device TYPE *>(dst_bytes + args.dst_offset); \
    ulong inner = args.inner;                                                        \
    ulong inner_i = inner > 1ul ? i % inner : 0ul;                                   \
    ulong block = inner > 1ul ? i / inner : i;                                       \
    ulong src_index = inner_i;                                                       \
    for (uint rd = args.active_rank; rd > 0u; --rd) {                                \
        uint d = rd - 1u;                                                            \
        ulong dim = args.dst_shape[d];                                               \
        ulong coord = block % dim;                                                   \
        block = block / dim;                                                         \
        src_index += coord * args.src_strides[args.axes[d]];                         \
    }                                                                                \
    dst[i] = src[src_index];                                                         \
}

#define GD_PERMUTE_DEFINE_BLOCK_KERNEL(SUFFIX, TYPE)                                        \
kernel void gd_permute_block_##SUFFIX##_kernel(device const uchar *src_bytes [[buffer(0)]], \
                                                device uchar *dst_bytes [[buffer(1)]],       \
                                                constant gd_metal_permute_args &args [[buffer(2)]], \
                                                uint gid [[thread_position_in_grid]])        \
{                                                                                           \
    ulong block_id = ulong(gid);                                                           \
    ulong inner = args.inner;                                                               \
    ulong n_blocks = args.count / inner;                                                    \
    if (block_id >= n_blocks) {                                                            \
        return;                                                                             \
    }                                                                                       \
    device const TYPE *src = reinterpret_cast<device const TYPE *>(src_bytes + args.src_offset); \
    device TYPE *dst = reinterpret_cast<device TYPE *>(dst_bytes + args.dst_offset);        \
    ulong block = block_id;                                                                 \
    ulong src_base = 0ul;                                                                   \
    for (uint rd = args.active_rank; rd > 0u; --rd) {                                      \
        uint d = rd - 1u;                                                                   \
        ulong dim = args.dst_shape[d];                                                      \
        ulong coord = block % dim;                                                          \
        block = block / dim;                                                                \
        src_base += coord * args.src_strides[args.axes[d]];                                 \
    }                                                                                       \
    ulong dst_base = block_id * inner;                                                      \
    for (ulong k = 0ul; k < inner; ++k) {                                                   \
        dst[dst_base + k] = src[src_base + k];                                              \
    }                                                                                       \
}

kernel void gd_permute_suffix16_kernel(device const uchar *src_bytes [[buffer(0)]],
                                        device uchar *dst_bytes [[buffer(1)]],
                                        constant gd_metal_permute_args &args [[buffer(2)]],
                                        uint2 gid [[thread_position_in_grid]])
{
    ulong vec = ulong(gid.x);
    ulong block_id = ulong(gid.y);
    ulong elem_size = ulong(args.reserved0);
    ulong inner_bytes = args.inner * elem_size;
    ulong vecs_per_block = inner_bytes >> 4;
    ulong n_blocks = args.count / args.inner;
    if (vec >= vecs_per_block || block_id >= n_blocks) {
        return;
    }
    ulong block = block_id;
    ulong src_base_elem = 0ul;
    for (uint rd = args.active_rank; rd > 0u; --rd) {
        uint d = rd - 1u;
        ulong dim = args.dst_shape[d];
        ulong coord = block % dim;
        block = block / dim;
        src_base_elem += coord * args.src_strides[args.axes[d]];
    }
    ulong src_byte = args.src_offset + src_base_elem * elem_size + vec * 16ul;
    ulong dst_byte = args.dst_offset + block_id * inner_bytes + vec * 16ul;
    device const uint4 *src = reinterpret_cast<device const uint4 *>(src_bytes + src_byte);
    device uint4 *dst = reinterpret_cast<device uint4 *>(dst_bytes + dst_byte);
    dst[0] = src[0];
}

#define GD_PERMUTE_TRANSPOSE_TILE 16u

#define GD_PERMUTE_DEFINE_TRANSPOSE_KERNEL(SUFFIX, TYPE)                                      \
kernel void gd_permute_transpose_##SUFFIX##_kernel(device const uchar *src_bytes [[buffer(0)]], \
                                                    device uchar *dst_bytes [[buffer(1)]],       \
                                                    constant gd_metal_permute_args &args [[buffer(2)]], \
                                                    uint3 tid [[thread_position_in_threadgroup]], \
                                                    uint3 tgid [[threadgroup_position_in_grid]]) \
{                                                                                              \
    threadgroup TYPE tile[GD_PERMUTE_TRANSPOSE_TILE][GD_PERMUTE_TRANSPOSE_TILE + 1u];          \
    device const TYPE *src = reinterpret_cast<device const TYPE *>(src_bytes + args.src_offset); \
    device TYPE *dst = reinterpret_cast<device TYPE *>(dst_bytes + args.dst_offset);           \
    ulong rows = args.rank == 3u ? args.dst_shape[2] : args.dst_shape[1];                      \
    ulong cols = args.rank == 3u ? args.dst_shape[1] : args.dst_shape[0];                      \
    ulong batch = ulong(tgid.z);                                                               \
    ulong src_col = ulong(tgid.x) * GD_PERMUTE_TRANSPOSE_TILE + ulong(tid.x);                  \
    ulong src_row = ulong(tgid.y) * GD_PERMUTE_TRANSPOSE_TILE + ulong(tid.y);                  \
    ulong matrix = rows * cols;                                                                \
    if (src_row < rows && src_col < cols) {                                                    \
        tile[tid.y][tid.x] = src[batch * matrix + src_row * cols + src_col];                   \
    }                                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                           \
    ulong dst_row = ulong(tgid.x) * GD_PERMUTE_TRANSPOSE_TILE + ulong(tid.y);                  \
    ulong dst_col = ulong(tgid.y) * GD_PERMUTE_TRANSPOSE_TILE + ulong(tid.x);                  \
    if (dst_row < cols && dst_col < rows) {                                                    \
        dst[batch * matrix + dst_row * rows + dst_col] = tile[tid.x][tid.y];                   \
    }                                                                                          \
}

#define GD_PERMUTE_DEFINE_HWC_TO_CHW_KERNEL(SUFFIX, TYPE)                                  \
kernel void gd_permute_hwc_to_chw_##SUFFIX##_kernel(device const uchar *src_bytes [[buffer(0)]], \
                                                     device uchar *dst_bytes [[buffer(1)]], \
                                                     constant gd_metal_permute_args &args [[buffer(2)]], \
                                                     uint gid [[thread_position_in_grid]])  \
{                                                                                           \
    device const TYPE *src = reinterpret_cast<device const TYPE *>(src_bytes + args.src_offset); \
    device TYPE *dst = reinterpret_cast<device TYPE *>(dst_bytes + args.dst_offset);        \
    ulong n = 0ul;                                                                          \
    ulong c;                                                                                \
    ulong h;                                                                                \
    ulong w;                                                                                \
    ulong C;                                                                                \
    ulong H;                                                                                \
    ulong W;                                                                                \
    ulong hw;                                                                               \
    ulong pixel = ulong(gid);                                                               \
    if (args.rank == 4u) {                                                                  \
        C = args.dst_shape[1];                                                              \
        H = args.dst_shape[2];                                                              \
        W = args.dst_shape[3];                                                              \
        hw = H * W;                                                                         \
        if (pixel >= args.dst_shape[0] * hw) {                                              \
            return;                                                                         \
        }                                                                                   \
        n = pixel / hw;                                                                     \
        pixel -= n * hw;                                                                    \
    } else {                                                                                \
        C = args.dst_shape[0];                                                              \
        H = args.dst_shape[1];                                                              \
        W = args.dst_shape[2];                                                              \
        hw = H * W;                                                                         \
        if (pixel >= hw) {                                                                  \
            return;                                                                         \
        }                                                                                   \
    }                                                                                       \
    h = pixel / W;                                                                          \
    w = pixel - h * W;                                                                      \
    ulong src_base = args.rank == 4u ? ((n * H + h) * W + w) * C : (h * W + w) * C;         \
    ulong dst_base = args.rank == 4u ? n * C * hw + pixel : pixel;                         \
    for (c = 0ul; c < C; ++c) {                                                            \
        dst[dst_base + c * hw] = src[src_base + c];                                        \
    }                                                                                       \
}

GD_PERMUTE_DEFINE_KERNEL(u8, uchar)
GD_PERMUTE_DEFINE_KERNEL(u16, ushort)
GD_PERMUTE_DEFINE_KERNEL(u32, uint)

GD_PERMUTE_DEFINE_BLOCK_KERNEL(u8, uchar)
GD_PERMUTE_DEFINE_BLOCK_KERNEL(u16, ushort)
GD_PERMUTE_DEFINE_BLOCK_KERNEL(u32, uint)

GD_PERMUTE_DEFINE_TRANSPOSE_KERNEL(u8, uchar)
GD_PERMUTE_DEFINE_TRANSPOSE_KERNEL(u16, ushort)
GD_PERMUTE_DEFINE_TRANSPOSE_KERNEL(u32, uint)

#define GD_PERMUTE_DEFINE_CHW_TO_HWC_KERNEL(SUFFIX, TYPE)                                  \
kernel void gd_permute_chw_to_hwc_##SUFFIX##_kernel(device const uchar *src_bytes [[buffer(0)]], \
                                                     device uchar *dst_bytes [[buffer(1)]], \
                                                     constant gd_metal_permute_args &args [[buffer(2)]], \
                                                     uint gid [[thread_position_in_grid]])  \
{                                                                                           \
    device const TYPE *src = reinterpret_cast<device const TYPE *>(src_bytes + args.src_offset); \
    device TYPE *dst = reinterpret_cast<device TYPE *>(dst_bytes + args.dst_offset);        \
    ulong n = 0ul;                                                                          \
    ulong c;                                                                                \
    ulong h;                                                                                \
    ulong w;                                                                                \
    ulong C;                                                                                \
    ulong H;                                                                                \
    ulong W;                                                                                \
    ulong hw;                                                                               \
    ulong pixel = ulong(gid);                                                               \
    if (args.rank == 4u) {                                                                  \
        H = args.dst_shape[1];                                                              \
        W = args.dst_shape[2];                                                              \
        C = args.dst_shape[3];                                                              \
        hw = H * W;                                                                         \
        if (pixel >= args.dst_shape[0] * hw) {                                              \
            return;                                                                         \
        }                                                                                   \
        n = pixel / hw;                                                                     \
        pixel -= n * hw;                                                                    \
    } else {                                                                                \
        H = args.dst_shape[0];                                                              \
        W = args.dst_shape[1];                                                              \
        C = args.dst_shape[2];                                                              \
        hw = H * W;                                                                         \
        if (pixel >= hw) {                                                                  \
            return;                                                                         \
        }                                                                                   \
    }                                                                                       \
    h = pixel / W;                                                                          \
    w = pixel - h * W;                                                                      \
    (void)h;                                                                                \
    (void)w;                                                                                \
    ulong src_base = args.rank == 4u ? n * C * hw + pixel : pixel;                         \
    ulong dst_base = args.rank == 4u ? (n * hw + pixel) * C : pixel * C;                   \
    for (c = 0ul; c < C; ++c) {                                                            \
        dst[dst_base + c] = src[src_base + c * hw];                                        \
    }                                                                                       \
}

GD_PERMUTE_DEFINE_HWC_TO_CHW_KERNEL(u8, uchar)
GD_PERMUTE_DEFINE_HWC_TO_CHW_KERNEL(u16, ushort)
GD_PERMUTE_DEFINE_HWC_TO_CHW_KERNEL(u32, uint)

GD_PERMUTE_DEFINE_CHW_TO_HWC_KERNEL(u8, uchar)
GD_PERMUTE_DEFINE_CHW_TO_HWC_KERNEL(u16, ushort)
GD_PERMUTE_DEFINE_CHW_TO_HWC_KERNEL(u32, uint)
