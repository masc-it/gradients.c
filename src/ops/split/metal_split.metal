#include <metal_stdlib>
#include "metal_split_types.h"

using namespace metal;

#define GD_SPLIT_DEFINE_FROM_FULL_KERNEL(SUFFIX, TYPE)                                      \
kernel void gd_split_from_full_##SUFFIX##_kernel(device const uchar *src_bytes [[buffer(0)]], \
                                                 device uchar *dst_bytes [[buffer(1)]],       \
                                                 constant gd_metal_split_args &args [[buffer(2)]], \
                                                 uint gid [[thread_position_in_grid]])       \
{                                                                                           \
    ulong i = ulong(gid);                                                                   \
    if (i >= args.count) {                                                                  \
        return;                                                                             \
    }                                                                                       \
    device const TYPE *src = reinterpret_cast<device const TYPE *>(src_bytes + args.src_offset); \
    device TYPE *dst = reinterpret_cast<device TYPE *>(dst_bytes + args.dst_offset);        \
    ulong slice_block = args.slice_axis * args.inner;                                       \
    ulong outer = i / slice_block;                                                         \
    ulong rem = i - outer * slice_block;                                                    \
    ulong src_index = outer * args.full_axis * args.inner + args.axis_offset * args.inner + rem; \
    dst[i] = src[src_index];                                                                \
}

#define GD_SPLIT_DEFINE_TO_FULL_KERNEL(SUFFIX, TYPE)                                      \
kernel void gd_split_to_full_##SUFFIX##_kernel(device const uchar *src_bytes [[buffer(0)]], \
                                               device uchar *dst_bytes [[buffer(1)]],       \
                                               constant gd_metal_split_args &args [[buffer(2)]], \
                                               uint gid [[thread_position_in_grid]])       \
{                                                                                         \
    ulong i = ulong(gid);                                                                 \
    if (i >= args.count) {                                                                \
        return;                                                                           \
    }                                                                                     \
    device const TYPE *src = reinterpret_cast<device const TYPE *>(src_bytes + args.src_offset); \
    device TYPE *dst = reinterpret_cast<device TYPE *>(dst_bytes + args.dst_offset);      \
    ulong slice_block = args.slice_axis * args.inner;                                     \
    ulong outer = i / slice_block;                                                       \
    ulong rem = i - outer * slice_block;                                                  \
    ulong dst_index = outer * args.full_axis * args.inner + args.axis_offset * args.inner + rem; \
    dst[dst_index] = src[i];                                                              \
}

kernel void gd_split_from_full_vec16_kernel(device const uchar *src_bytes [[buffer(0)]],
                                            device uchar *dst_bytes [[buffer(1)]],
                                            constant gd_metal_split_args &args [[buffer(2)]],
                                            uint2 gid [[thread_position_in_grid]])
{
    ulong vec = ulong(gid.x);
    ulong outer = ulong(gid.y);
    ulong slice_block_elems = args.slice_axis * args.inner;
    ulong slice_block_bytes = slice_block_elems * args.elem_size;
    ulong full_row_bytes = args.full_axis * args.inner * args.elem_size;
    ulong vecs_per_block = slice_block_bytes >> 4;
    ulong blocks = args.count / slice_block_elems;
    if (vec >= vecs_per_block || outer >= blocks) {
        return;
    }
    ulong src_byte = args.src_offset + outer * full_row_bytes +
                     args.axis_offset * args.inner * args.elem_size + vec * 16ul;
    ulong dst_byte = args.dst_offset + outer * slice_block_bytes + vec * 16ul;
    device const uint4 *src = reinterpret_cast<device const uint4 *>(src_bytes + src_byte);
    device uint4 *dst = reinterpret_cast<device uint4 *>(dst_bytes + dst_byte);
    dst[0] = src[0];
}

kernel void gd_split_to_full_vec16_kernel(device const uchar *src_bytes [[buffer(0)]],
                                          device uchar *dst_bytes [[buffer(1)]],
                                          constant gd_metal_split_args &args [[buffer(2)]],
                                          uint2 gid [[thread_position_in_grid]])
{
    ulong vec = ulong(gid.x);
    ulong outer = ulong(gid.y);
    ulong slice_block_elems = args.slice_axis * args.inner;
    ulong slice_block_bytes = slice_block_elems * args.elem_size;
    ulong full_row_bytes = args.full_axis * args.inner * args.elem_size;
    ulong vecs_per_block = slice_block_bytes >> 4;
    ulong blocks = args.count / slice_block_elems;
    if (vec >= vecs_per_block || outer >= blocks) {
        return;
    }
    ulong src_byte = args.src_offset + outer * slice_block_bytes + vec * 16ul;
    ulong dst_byte = args.dst_offset + outer * full_row_bytes +
                     args.axis_offset * args.inner * args.elem_size + vec * 16ul;
    device const uint4 *src = reinterpret_cast<device const uint4 *>(src_bytes + src_byte);
    device uint4 *dst = reinterpret_cast<device uint4 *>(dst_bytes + dst_byte);
    dst[0] = src[0];
}

GD_SPLIT_DEFINE_FROM_FULL_KERNEL(u8, uchar)
GD_SPLIT_DEFINE_FROM_FULL_KERNEL(u16, ushort)
GD_SPLIT_DEFINE_FROM_FULL_KERNEL(u32, uint)

GD_SPLIT_DEFINE_TO_FULL_KERNEL(u8, uchar)
GD_SPLIT_DEFINE_TO_FULL_KERNEL(u16, ushort)
GD_SPLIT_DEFINE_TO_FULL_KERNEL(u32, uint)
