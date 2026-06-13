/*
 * v2 Metal arena probe.
 *
 * Standalone Objective-C/Metal probe for gradients.c v2 device-memory shape.
 * No v1 headers. Exercises:
 *  - shared Metal arena buffers
 *  - arena suballocation + buffer offsets
 *  - scratch/data ring slots backed by Metal command buffers
 *  - direct CPU/GPU access through shared storage
 *  - state-object fence wait before reset/reuse
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "metal probe failed: %s (%s:%d)\n", (msg),        \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(cond, msg) CHECK((cond), (msg))

static NSUInteger align_up_ns(NSUInteger v, NSUInteger align)
{
    CHECK(align != 0U && (align & (align - 1U)) == 0U, "alignment power of two");
    return (v + align - 1U) & ~(align - 1U);
}

static MTLResourceOptions probe_shared_options(void)
{
    return MTLResourceStorageModeShared;
}

static NSMutableArray *gProbeKeepAlive;

typedef struct ProbeTensorRef {
    __unsafe_unretained id<MTLBuffer> buffer;
    NSUInteger offset;
    NSUInteger nbytes;
    bool cpu_visible;
    const char *name;
} ProbeTensorRef;

@interface ProbeMetalArena : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, copy) NSString *name;
@property(nonatomic) NSUInteger capacity;
@property(nonatomic) NSUInteger offset;
@property(nonatomic) NSUInteger watermark;
@property(nonatomic) MTLResourceOptions options;
@property(nonatomic) bool cpuVisible;
- (instancetype)initWithDevice:(id<MTLDevice>)device
                          name:(NSString *)name
                      capacity:(NSUInteger)capacity
                       options:(MTLResourceOptions)options;
- (void)reset;
- (ProbeTensorRef)allocBytes:(NSUInteger)nbytes
                   alignment:(NSUInteger)alignment
                        name:(const char *)name;
@end

@implementation ProbeMetalArena
- (instancetype)initWithDevice:(id<MTLDevice>)device
                          name:(NSString *)name
                      capacity:(NSUInteger)capacity
                       options:(MTLResourceOptions)options
{
    self = [super init];
    if (self == nil) {
        return nil;
    }
    _name = [name copy];
    _capacity = capacity;
    _offset = 0;
    _watermark = 0;
    _options = options;
    _buffer = [device newBufferWithLength:capacity options:options];
    CHECK(_buffer != nil, "newBufferWithLength failed");
    _cpuVisible = (_buffer.contents != NULL);
    return self;
}

- (void)reset
{
    _offset = 0;
}

- (ProbeTensorRef)allocBytes:(NSUInteger)nbytes
                   alignment:(NSUInteger)alignment
                        name:(const char *)name
{
    ProbeTensorRef ref;
    NSUInteger off;
    memset(&ref, 0, sizeof(ref));
    CHECK(nbytes > 0, "arena alloc nonzero");
    if (alignment == 0) {
        alignment = 256;
    }
    off = align_up_ns(_offset, alignment);
    CHECK(off <= _capacity && nbytes <= _capacity - off, "metal arena out of memory");
    ref.buffer = _buffer;
    ref.offset = off;
    ref.nbytes = nbytes;
    ref.cpu_visible = _cpuVisible;
    ref.name = name;
    _offset = off + nbytes;
    if (_offset > _watermark) {
        _watermark = _offset;
    }
    return ref;
}
@end

@interface ProbeRingArena : NSObject
@property(nonatomic, copy) NSString *name;
@property(nonatomic, strong) NSMutableArray<ProbeMetalArena *> *slots;
@property(nonatomic, strong) NSMutableArray *commands; /* id<MTLCommandBuffer> or NSNull */
@property(nonatomic, strong) NSMutableArray<NSNumber *> *sequences;
@property(nonatomic) NSInteger current;
@property(nonatomic) uint64_t waits;
- (instancetype)initWithDevice:(id<MTLDevice>)device
                          name:(NSString *)name
                        slots:(NSUInteger)nSlots
                  slotCapacity:(NSUInteger)slotCapacity
                       options:(MTLResourceOptions)options;
- (ProbeMetalArena *)selectSlot;
- (void)recordCommandBuffer:(id<MTLCommandBuffer>)cmd sequence:(uint64_t)sequence;
- (void)waitAll;
@end

@implementation ProbeRingArena
- (instancetype)initWithDevice:(id<MTLDevice>)device
                          name:(NSString *)name
                        slots:(NSUInteger)nSlots
                  slotCapacity:(NSUInteger)slotCapacity
                       options:(MTLResourceOptions)options
{
    NSUInteger i;
    self = [super init];
    if (self == nil) {
        return nil;
    }
    CHECK(nSlots > 0, "ring needs slots");
    _name = [name copy];
    _slots = [NSMutableArray arrayWithCapacity:nSlots];
    _commands = [NSMutableArray arrayWithCapacity:nSlots];
    _sequences = [NSMutableArray arrayWithCapacity:nSlots];
    _current = -1;
    _waits = 0;
    for (i = 0; i < nSlots; ++i) {
        NSString *slotName = [NSString stringWithFormat:@"%@[%lu]", name, (unsigned long)i];
        ProbeMetalArena *arena = [[ProbeMetalArena alloc] initWithDevice:device
                                                                    name:slotName
                                                                capacity:slotCapacity
                                                                 options:options];
        [_slots addObject:arena];
        [_commands addObject:[NSNull null]];
        [_sequences addObject:@0];
    }
    return self;
}

- (id<MTLCommandBuffer>)commandAt:(NSUInteger)idx
{
    id obj = _commands[idx];
    return obj == [NSNull null] ? nil : (id<MTLCommandBuffer>)obj;
}

- (BOOL)isSlotFree:(NSUInteger)idx
{
    id<MTLCommandBuffer> cmd = [self commandAt:idx];
    if (cmd == nil) {
        return YES;
    }
    if (cmd.status == MTLCommandBufferStatusCompleted) {
        _commands[idx] = [NSNull null];
        return YES;
    }
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "command buffer error on slot %lu: %s\n", (unsigned long)idx,
                cmd.error.localizedDescription.UTF8String);
        exit(1);
    }
    return NO;
}

- (ProbeMetalArena *)selectSlot
{
    NSUInteger n = _slots.count;
    NSUInteger start = _current < 0 ? 0U : ((NSUInteger)_current + 1U) % n;
    NSUInteger attempt;
    for (attempt = 0; attempt < n; ++attempt) {
        NSUInteger idx = (start + attempt) % n;
        if ([self isSlotFree:idx]) {
            _current = (NSInteger)idx;
            [_slots[idx] reset];
            return _slots[idx];
        }
    }

    /* Ring exhausted: wait oldest recorded command buffer. */
    {
        NSUInteger i;
        NSUInteger oldest = 0;
        uint64_t best = UINT64_MAX;
        id<MTLCommandBuffer> cmd;
        for (i = 0; i < n; ++i) {
            uint64_t seq = [_sequences[i] unsignedLongLongValue];
            if (seq != 0U && seq < best) {
                best = seq;
                oldest = i;
            }
        }
        cmd = [self commandAt:oldest];
        CHECK(cmd != nil, "ring exhausted but oldest command nil");
        [cmd waitUntilCompleted];
        CHECK(cmd.status == MTLCommandBufferStatusCompleted, "oldest command completed");
        _commands[oldest] = [NSNull null];
        _waits += 1U;
        _current = (NSInteger)oldest;
        [_slots[oldest] reset];
        return _slots[oldest];
    }
}

- (void)recordCommandBuffer:(id<MTLCommandBuffer>)cmd sequence:(uint64_t)sequence
{
    CHECK(_current >= 0, "record command needs current slot");
    _commands[(NSUInteger)_current] = cmd;
    _sequences[(NSUInteger)_current] = @(sequence);
}

- (void)waitAll
{
    NSUInteger i;
    for (i = 0; i < _slots.count; ++i) {
        id<MTLCommandBuffer> cmd = [self commandAt:i];
        if (cmd != nil) {
            [cmd waitUntilCompleted];
            CHECK(cmd.status == MTLCommandBufferStatusCompleted, "ring waitAll completed");
            _commands[i] = [NSNull null];
        }
    }
}
@end

typedef struct ProbeStateObject {
    ProbeTensorRef storage;
    __unsafe_unretained id<MTLCommandBuffer> lastCommand;
    uint64_t sequence;
} ProbeStateObject;

typedef struct ProbeMetalContext {
    __unsafe_unretained id<MTLDevice> device;
    __unsafe_unretained id<MTLCommandQueue> queue;
    __unsafe_unretained id<MTLLibrary> library;
    __unsafe_unretained id<MTLComputePipelineState> addOnePSO;
    __unsafe_unretained id<MTLComputePipelineState> spinPSO;
    __unsafe_unretained id<MTLComputePipelineState> biasGeluPSO;
    __unsafe_unretained id<MTLComputePipelineState> rmsNormPSO;
    __unsafe_unretained id<MTLComputePipelineState> layerNormPSO;
    __unsafe_unretained ProbeRingArena *scratchRing;
    __unsafe_unretained ProbeRingArena *dataRing;
    __unsafe_unretained ProbeMetalArena *scratch;
    __unsafe_unretained ProbeMetalArena *data;
    __unsafe_unretained id<MTLCommandBuffer> currentCommand;
    uint64_t nextSequence;
} ProbeMetalContext;

static NSString *probe_metal_source(void)
{
    return @"#include <metal_stdlib>\n"
           @"using namespace metal;\n"
           @"kernel void add_one(device const float *x [[buffer(0)]],\n"
           @"                    device float *y [[buffer(1)]],\n"
           @"                    uint gid [[thread_position_in_grid]]) {\n"
           @"    y[gid] = x[gid] + 1.0f;\n"
           @"}\n"
           @"kernel void spin(device float *out [[buffer(0)]],\n"
           @"                 constant uint &iters [[buffer(1)]],\n"
           @"                 uint gid [[thread_position_in_grid]]) {\n"
           @"    float x = ((float)(gid + 1u)) * 0.001f;\n"
           @"    for (uint i = 0u; i < iters; ++i) {\n"
           @"        x = x * 1.000000119f + 0.00000013f;\n"
           @"        if (x > 10000.0f) { x *= 0.0001f; }\n"
           @"    }\n"
           @"    out[gid] = x;\n"
           @"}\n"
           @"kernel void bias_gelu_f16(device const uchar *xBytes [[buffer(0)]],\n"
           @"                          device const half *bias [[buffer(1)]],\n"
           @"                          device uchar *yBytes [[buffer(2)]],\n"
           @"                          constant uint &rows [[buffer(3)]],\n"
           @"                          constant uint &cols [[buffer(4)]],\n"
           @"                          constant uint &xRowBytes [[buffer(5)]],\n"
           @"                          constant uint &yRowBytes [[buffer(6)]],\n"
           @"                          uint gid [[thread_position_in_grid]]) {\n"
           @"    const uint total = rows * cols;\n"
           @"    if (gid >= total) { return; }\n"
           @"    const uint r = gid / cols;\n"
           @"    const uint c = gid - r * cols;\n"
           @"    device const half *xRow = (device const half *)(xBytes + ((ulong)r) * ((ulong)xRowBytes));\n"
           @"    device half *yRow = (device half *)(yBytes + ((ulong)r) * ((ulong)yRowBytes));\n"
           @"    const float v = (float)xRow[c] + (float)bias[c];\n"
           @"    const float v3 = v * v * v;\n"
           @"    yRow[c] = (half)(0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * v3))));\n"
           @"}\n"
           @"kernel void rms_norm_f16(device const uchar *xBytes [[buffer(0)]],\n"
           @"                         device const half *scale [[buffer(1)]],\n"
           @"                         device uchar *yBytes [[buffer(2)]],\n"
           @"                         constant uint &rows [[buffer(3)]],\n"
           @"                         constant uint &cols [[buffer(4)]],\n"
           @"                         constant uint &xRowBytes [[buffer(5)]],\n"
           @"                         constant uint &yRowBytes [[buffer(6)]],\n"
           @"                         constant float &eps [[buffer(7)]],\n"
           @"                         uint3 tgid [[threadgroup_position_in_grid]],\n"
           @"                         uint tid [[thread_index_in_threadgroup]]) {\n"
           @"    threadgroup float sums[256];\n"
           @"    const uint r = tgid.x;\n"
           @"    if (r >= rows) { return; }\n"
           @"    device const half *xRow = (device const half *)(xBytes + ((ulong)r) * ((ulong)xRowBytes));\n"
           @"    float ss = 0.0f;\n"
           @"    for (uint c = tid; c < cols; c += 256u) { const float v = (float)xRow[c]; ss += v * v; }\n"
           @"    sums[tid] = ss;\n"
           @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           @"    for (uint stride = 128u; stride > 0u; stride >>= 1u) {\n"
           @"        if (tid < stride) { sums[tid] += sums[tid + stride]; }\n"
           @"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           @"    }\n"
           @"    const float inv = rsqrt(sums[0] / (float)cols + eps);\n"
           @"    device half *yRow = (device half *)(yBytes + ((ulong)r) * ((ulong)yRowBytes));\n"
           @"    for (uint c = tid; c < cols; c += 256u) { yRow[c] = (half)(((float)xRow[c]) * inv * ((float)scale[c])); }\n"
           @"}\n"
           @"kernel void layer_norm_f16(device const uchar *xBytes [[buffer(0)]],\n"
           @"                           device const half *gamma [[buffer(1)]],\n"
           @"                           device const half *beta [[buffer(2)]],\n"
           @"                           device uchar *yBytes [[buffer(3)]],\n"
           @"                           constant uint &rows [[buffer(4)]],\n"
           @"                           constant uint &cols [[buffer(5)]],\n"
           @"                           constant uint &xRowBytes [[buffer(6)]],\n"
           @"                           constant uint &yRowBytes [[buffer(7)]],\n"
           @"                           constant float &eps [[buffer(8)]],\n"
           @"                           uint3 tgid [[threadgroup_position_in_grid]],\n"
           @"                           uint tid [[thread_index_in_threadgroup]]) {\n"
           @"    threadgroup float sums[256];\n"
           @"    threadgroup float sqs[256];\n"
           @"    const uint r = tgid.x;\n"
           @"    if (r >= rows) { return; }\n"
           @"    device const half *xRow = (device const half *)(xBytes + ((ulong)r) * ((ulong)xRowBytes));\n"
           @"    float s = 0.0f;\n"
           @"    float ss = 0.0f;\n"
           @"    for (uint c = tid; c < cols; c += 256u) { const float v = (float)xRow[c]; s += v; ss += v * v; }\n"
           @"    sums[tid] = s;\n"
           @"    sqs[tid] = ss;\n"
           @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           @"    for (uint stride = 128u; stride > 0u; stride >>= 1u) {\n"
           @"        if (tid < stride) { sums[tid] += sums[tid + stride]; sqs[tid] += sqs[tid + stride]; }\n"
           @"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           @"    }\n"
           @"    const float mean = sums[0] / (float)cols;\n"
           @"    const float var = max(sqs[0] / (float)cols - mean * mean, 0.0f);\n"
           @"    const float inv = rsqrt(var + eps);\n"
           @"    device half *yRow = (device half *)(yBytes + ((ulong)r) * ((ulong)yRowBytes));\n"
           @"    for (uint c = tid; c < cols; c += 256u) {\n"
           @"        const float v = (((float)xRow[c]) - mean) * inv * ((float)gamma[c]) + ((float)beta[c]);\n"
           @"        yRow[c] = (half)v;\n"
           @"    }\n"
           @"}\n";
}

static void write_floats(ProbeTensorRef t, const float *values, NSUInteger n)
{
    CHECK(t.cpu_visible, "write_floats requires shared buffer");
    CHECK(t.nbytes >= n * sizeof(float), "write_floats size");
    memcpy((unsigned char *)t.buffer.contents + t.offset, values, n * sizeof(float));
}

static void read_floats(ProbeTensorRef t, float *values, NSUInteger n)
{
    CHECK(t.cpu_visible, "read_floats requires shared buffer");
    CHECK(t.nbytes >= n * sizeof(float), "read_floats size");
    memcpy(values, (unsigned char *)t.buffer.contents + t.offset, n * sizeof(float));
}

static ProbeTensorRef tensor_subref(ProbeTensorRef base,
                                    NSUInteger relOffset,
                                    NSUInteger nbytes,
                                    const char *name)
{
    ProbeTensorRef ref = base;
    CHECK(relOffset <= base.nbytes && nbytes <= base.nbytes - relOffset,
          "tensor_subref bounds");
    ref.offset = base.offset + relOffset;
    ref.nbytes = nbytes;
    ref.name = name;
    return ref;
}

static uint64_t splitmix64(uint64_t *state)
{
    uint64_t z;
    *state += UINT64_C(0x9e3779b97f4a7c15);
    z = *state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31U);
}

static float rand_uniform_signed(uint64_t *state, float scale)
{
    uint64_t z = splitmix64(state);
    float u = ((float)(z >> 40U) + 0.5f) * (1.0f / 16777216.0f);
    return (2.0f * u - 1.0f) * scale;
}

static float f16_bits_to_f32(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000U) << 16U;
    uint32_t exp = ((uint32_t)h >> 10U) & 0x1fU;
    uint32_t mant = (uint32_t)h & 0x03ffU;
    uint32_t bits;
    union { uint32_t u; float f; } v;
    if (exp == 0U) {
        if (mant == 0U) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400U) == 0U) {
                mant <<= 1U;
                e -= 1;
            }
            mant &= 0x03ffU;
            bits = sign | ((uint32_t)(e + 127) << 23U) | (mant << 13U);
        }
    } else if (exp == 31U) {
        bits = sign | 0x7f800000U | (mant << 13U);
    } else {
        bits = sign | ((exp - 15U + 127U) << 23U) | (mant << 13U);
    }
    v.u = bits;
    return v.f;
}

static uint16_t f32_to_f16_bits(float f)
{
    union { float f; uint32_t u; } v;
    uint32_t sign;
    uint32_t mant;
    int exp;
    v.f = f;
    sign = (v.u >> 16U) & 0x8000U;
    exp = (int)((v.u >> 23U) & 0xffU) - 127 + 15;
    mant = v.u & 0x7fffffU;
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant = (mant | 0x800000U) >> (uint32_t)(1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000U) >> 13U));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00U);
    }
    mant += 0x1000U;
    if ((mant & 0x800000U) != 0U) {
        mant = 0U;
        exp += 1;
        if (exp >= 31) {
            return (uint16_t)(sign | 0x7c00U);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10U) | (mant >> 13U));
}

static void write_matrix_f16_from_f32(ProbeTensorRef t,
                                      NSUInteger rows,
                                      NSUInteger cols,
                                      NSUInteger rowBytes,
                                      const float *values,
                                      float *rounded)
{
    NSUInteger r;
    NSUInteger c;
    CHECK(t.cpu_visible, "write_matrix_f16_from_f32 requires shared buffer");
    CHECK(t.nbytes >= rows * rowBytes, "write_matrix_f16_from_f32 size");
    CHECK(rowBytes >= cols * sizeof(uint16_t), "write_matrix_f16_from_f32 rowBytes");
    for (r = 0; r < rows; ++r) {
        uint16_t *row = (uint16_t *)((unsigned char *)t.buffer.contents + t.offset + r * rowBytes);
        for (c = 0; c < cols; ++c) {
            uint16_t bits = f32_to_f16_bits(values[r * cols + c]);
            row[c] = bits;
            if (rounded != NULL) {
                rounded[r * cols + c] = f16_bits_to_f32(bits);
            }
        }
    }
}

static void read_matrix_f16_to_f32(ProbeTensorRef t,
                                   NSUInteger rows,
                                   NSUInteger cols,
                                   NSUInteger rowBytes,
                                   float *values)
{
    NSUInteger r;
    NSUInteger c;
    CHECK(t.cpu_visible, "read_matrix_f16_to_f32 requires shared buffer");
    CHECK(t.nbytes >= rows * rowBytes, "read_matrix_f16_to_f32 size");
    CHECK(rowBytes >= cols * sizeof(uint16_t), "read_matrix_f16_to_f32 rowBytes");
    for (r = 0; r < rows; ++r) {
        uint16_t *row = (uint16_t *)((unsigned char *)t.buffer.contents + t.offset + r * rowBytes);
        for (c = 0; c < cols; ++c) {
            values[r * cols + c] = f16_bits_to_f32(row[c]);
        }
    }
}

static void fill_matrix_f16(ProbeTensorRef t,
                            NSUInteger rows,
                            NSUInteger cols,
                            NSUInteger rowBytes,
                            float value)
{
    NSUInteger r;
    NSUInteger c;
    uint16_t bits = f32_to_f16_bits(value);
    CHECK(t.cpu_visible, "fill_matrix_f16 requires shared buffer");
    CHECK(t.nbytes >= rows * rowBytes, "fill_matrix_f16 size");
    CHECK(rowBytes >= cols * sizeof(uint16_t), "fill_matrix_f16 rowBytes");
    for (r = 0; r < rows; ++r) {
        uint16_t *row = (uint16_t *)((unsigned char *)t.buffer.contents + t.offset + r * rowBytes);
        for (c = 0; c < cols; ++c) {
            row[c] = bits;
        }
    }
}

static float matrix_value_f16(ProbeTensorRef t,
                              NSUInteger row,
                              NSUInteger col,
                              NSUInteger rowBytes)
{
    CHECK(t.cpu_visible, "matrix_value_f16 requires shared buffer");
    CHECK(t.nbytes >= row * rowBytes + (col + 1U) * sizeof(uint16_t),
          "matrix_value_f16 bounds");
    return f16_bits_to_f32(*(uint16_t *)((unsigned char *)t.buffer.contents +
                                        t.offset + row * rowBytes +
                                        col * sizeof(uint16_t)));
}

static void write_vector_f16_from_f32(ProbeTensorRef t,
                                      const float *values,
                                      float *rounded,
                                      NSUInteger n)
{
    NSUInteger i;
    uint16_t *dst;
    CHECK(t.cpu_visible, "write_vector_f16_from_f32 requires shared buffer");
    CHECK(t.nbytes >= n * sizeof(uint16_t), "write_vector_f16_from_f32 size");
    dst = (uint16_t *)((unsigned char *)t.buffer.contents + t.offset);
    for (i = 0; i < n; ++i) {
        uint16_t bits = f32_to_f16_bits(values[i]);
        dst[i] = bits;
        if (rounded != NULL) {
            rounded[i] = f16_bits_to_f32(bits);
        }
    }
}

static void read_vector_f16_to_f32(ProbeTensorRef t, float *values, NSUInteger n)
{
    NSUInteger i;
    uint16_t *src;
    CHECK(t.cpu_visible, "read_vector_f16_to_f32 requires shared buffer");
    CHECK(t.nbytes >= n * sizeof(uint16_t), "read_vector_f16_to_f32 size");
    src = (uint16_t *)((unsigned char *)t.buffer.contents + t.offset);
    for (i = 0; i < n; ++i) {
        values[i] = f16_bits_to_f32(src[i]);
    }
}

static void write_matrix_f16_random(ProbeTensorRef t,
                                    NSUInteger rows,
                                    NSUInteger cols,
                                    NSUInteger rowBytes,
                                    uint64_t *rng,
                                    float scale)
{
    NSUInteger r;
    NSUInteger c;
    CHECK(t.cpu_visible, "write_matrix_f16_random requires shared buffer");
    CHECK(t.nbytes >= rows * rowBytes, "write_matrix_f16_random size");
    for (r = 0; r < rows; ++r) {
        uint16_t *row = (uint16_t *)((unsigned char *)t.buffer.contents + t.offset + r * rowBytes);
        for (c = 0; c < cols; ++c) {
            row[c] = f32_to_f16_bits(rand_uniform_signed(rng, scale));
        }
    }
}

static void write_batched_matrix_f16_random(ProbeTensorRef t,
                                            NSUInteger matrices,
                                            NSUInteger rows,
                                            NSUInteger cols,
                                            NSUInteger rowBytes,
                                            NSUInteger matrixBytes,
                                            uint64_t *rng,
                                            float scale)
{
    NSUInteger b;
    CHECK(matrices > 0, "write_batched_matrix_f16_random matrices");
    CHECK(matrixBytes >= rows * rowBytes && (matrixBytes % rowBytes) == 0U,
          "write_batched_matrix_f16_random matrixBytes");
    CHECK(t.nbytes >= (matrices - 1U) * matrixBytes + rows * rowBytes,
          "write_batched_matrix_f16_random size");
    for (b = 0; b < matrices; ++b) {
        ProbeTensorRef mb = tensor_subref(t, b * matrixBytes, rows * rowBytes,
                                          "batched.f16.random.matrix");
        write_matrix_f16_random(mb, rows, cols, rowBytes, rng, scale);
    }
}

static void encode_add_one(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso,
                           ProbeTensorRef input,
                           ProbeTensorRef output,
                           NSUInteger n)
{
    MTLSize grid = MTLSizeMake(n, 1, 1);
    NSUInteger tw = pso.maxTotalThreadsPerThreadgroup < 64 ? pso.maxTotalThreadsPerThreadgroup : 64;
    MTLSize threads = MTLSizeMake(tw, 1, 1);
    [enc setComputePipelineState:pso];
    [enc setBuffer:input.buffer offset:input.offset atIndex:0];
    [enc setBuffer:output.buffer offset:output.offset atIndex:1];
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
}

static void encode_spin(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> pso,
                        ProbeTensorRef output,
                        uint32_t iters,
                        NSUInteger n)
{
    MTLSize grid = MTLSizeMake(n, 1, 1);
    NSUInteger tw = pso.maxTotalThreadsPerThreadgroup < 128 ? pso.maxTotalThreadsPerThreadgroup : 128;
    MTLSize threads = MTLSizeMake(tw, 1, 1);
    [enc setComputePipelineState:pso];
    [enc setBuffer:output.buffer offset:output.offset atIndex:0];
    [enc setBytes:&iters length:sizeof(iters) atIndex:1];
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
}

static void encode_bias_gelu_f16(id<MTLComputeCommandEncoder> enc,
                                 id<MTLComputePipelineState> pso,
                                 ProbeTensorRef input,
                                 ProbeTensorRef bias,
                                 ProbeTensorRef output,
                                 uint32_t rows,
                                 uint32_t cols,
                                 uint32_t inputRowBytes,
                                 uint32_t outputRowBytes)
{
    NSUInteger n = (NSUInteger)rows * (NSUInteger)cols;
    NSUInteger tw = pso.maxTotalThreadsPerThreadgroup < 128 ? pso.maxTotalThreadsPerThreadgroup : 128;
    MTLSize grid = MTLSizeMake(n, 1, 1);
    MTLSize threads = MTLSizeMake(tw, 1, 1);
    [enc setComputePipelineState:pso];
    [enc setBuffer:input.buffer offset:input.offset atIndex:0];
    [enc setBuffer:bias.buffer offset:bias.offset atIndex:1];
    [enc setBuffer:output.buffer offset:output.offset atIndex:2];
    [enc setBytes:&rows length:sizeof(rows) atIndex:3];
    [enc setBytes:&cols length:sizeof(cols) atIndex:4];
    [enc setBytes:&inputRowBytes length:sizeof(inputRowBytes) atIndex:5];
    [enc setBytes:&outputRowBytes length:sizeof(outputRowBytes) atIndex:6];
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
}

static void encode_rms_norm_f16(id<MTLComputeCommandEncoder> enc,
                                id<MTLComputePipelineState> pso,
                                ProbeTensorRef input,
                                ProbeTensorRef scale,
                                ProbeTensorRef output,
                                uint32_t rows,
                                uint32_t cols,
                                uint32_t inputRowBytes,
                                uint32_t outputRowBytes,
                                float eps)
{
    MTLSize groups = MTLSizeMake(rows, 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    [enc setComputePipelineState:pso];
    [enc setBuffer:input.buffer offset:input.offset atIndex:0];
    [enc setBuffer:scale.buffer offset:scale.offset atIndex:1];
    [enc setBuffer:output.buffer offset:output.offset atIndex:2];
    [enc setBytes:&rows length:sizeof(rows) atIndex:3];
    [enc setBytes:&cols length:sizeof(cols) atIndex:4];
    [enc setBytes:&inputRowBytes length:sizeof(inputRowBytes) atIndex:5];
    [enc setBytes:&outputRowBytes length:sizeof(outputRowBytes) atIndex:6];
    [enc setBytes:&eps length:sizeof(eps) atIndex:7];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

static void encode_layer_norm_f16(id<MTLComputeCommandEncoder> enc,
                                  id<MTLComputePipelineState> pso,
                                  ProbeTensorRef input,
                                  ProbeTensorRef gamma,
                                  ProbeTensorRef beta,
                                  ProbeTensorRef output,
                                  uint32_t rows,
                                  uint32_t cols,
                                  uint32_t inputRowBytes,
                                  uint32_t outputRowBytes,
                                  float eps)
{
    MTLSize groups = MTLSizeMake(rows, 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    [enc setComputePipelineState:pso];
    [enc setBuffer:input.buffer offset:input.offset atIndex:0];
    [enc setBuffer:gamma.buffer offset:gamma.offset atIndex:1];
    [enc setBuffer:beta.buffer offset:beta.offset atIndex:2];
    [enc setBuffer:output.buffer offset:output.offset atIndex:3];
    [enc setBytes:&rows length:sizeof(rows) atIndex:4];
    [enc setBytes:&cols length:sizeof(cols) atIndex:5];
    [enc setBytes:&inputRowBytes length:sizeof(inputRowBytes) atIndex:6];
    [enc setBytes:&outputRowBytes length:sizeof(outputRowBytes) atIndex:7];
    [enc setBytes:&eps length:sizeof(eps) atIndex:8];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

static void ctx_begin(ProbeMetalContext *ctx)
{
    CHECK(ctx->currentCommand == nil, "ctx_begin without active command");
    ctx->scratch = [ctx->scratchRing selectSlot];
    ctx->data = [ctx->dataRing selectSlot];
    {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        CHECK(cmd != nil, "command buffer allocation");
        CHECK(gProbeKeepAlive != nil, "command keepalive initialized");
        [gProbeKeepAlive addObject:cmd];
        ctx->currentCommand = cmd;
    }
}

static id<MTLCommandBuffer> ctx_end(ProbeMetalContext *ctx)
{
    id<MTLCommandBuffer> cmd = ctx->currentCommand;
    uint64_t seq;
    CHECK(cmd != nil, "ctx_end active command");
    seq = ++ctx->nextSequence;
    [cmd commit];
    [ctx->scratchRing recordCommandBuffer:cmd sequence:seq];
    [ctx->dataRing recordCommandBuffer:cmd sequence:seq];
    ctx->currentCommand = nil;
    return cmd;
}

static void assert_float_close(float got, float want, const char *msg)
{
    float diff = got > want ? got - want : want - got;
    if (diff > 1.0e-5f) {
        fprintf(stderr, "metal probe failed: %s got=%g want=%g\n", msg, (double)got, (double)want);
        exit(1);
    }
}

static void assert_float_near(float got, float want, float tol, const char *msg)
{
    float diff = got > want ? got - want : want - got;
    if (diff > tol) {
        fprintf(stderr, "metal probe failed: %s got=%g want=%g tol=%g\n",
                msg, (double)got, (double)want, (double)tol);
        exit(1);
    }
}

static int env_int(const char *name, int fallback)
{
    const char *v = getenv(name);
    char *end = NULL;
    long parsed;
    if (v == NULL || v[0] == '\0') {
        return fallback;
    }
    parsed = strtol(v, &end, 10);
    if (end == v || *end != '\0' || parsed <= 0L || parsed > 100000L) {
        return fallback;
    }
    return (int)parsed;
}

static NSUInteger env_ns(const char *name, NSUInteger fallback)
{
    const char *v = getenv(name);
    char *end = NULL;
    unsigned long long parsed;
    if (v == NULL || v[0] == '\0') {
        return fallback;
    }
    parsed = strtoull(v, &end, 10);
    if (end == v || *end != '\0' || parsed > 1024ULL * 1024ULL * 1024ULL) {
        return fallback;
    }
    return (NSUInteger)parsed;
}

static bool env_is(const char *name, const char *want)
{
    const char *v = getenv(name);
    return v != NULL && strcmp(v, want) == 0;
}

static NSUInteger bench_alloc_alignment(void)
{
    NSUInteger align = env_ns("GD_PROBE_ALLOC_ALIGN", 256);
    if (align == 0U || (align & (align - 1U)) != 0U) {
        return 256;
    }
    return align;
}

static NSUInteger bench_prefix_pad(void)
{
    return env_ns("GD_PROBE_BENCH_PREFIX_PAD", 0);
}

static NSUInteger bench_row_pad_bytes(void)
{
    NSUInteger pad = env_ns("GD_PROBE_ROW_PAD_BYTES", 0);
    return (pad % sizeof(uint16_t)) == 0U ? pad : 0U;
}

static NSUInteger bench_batch_pad_rows(void)
{
    return env_ns("GD_PROBE_BATCH_PAD_ROWS", 0);
}

static NSUInteger ring_slots_env(const char *name, NSUInteger fallback)
{
    NSUInteger slots = env_ns(name, fallback);
    if (slots == 0U || slots > 64U) {
        return fallback;
    }
    return slots;
}

static NSUInteger ring_slot_bytes_env(const char *name,
                                      NSUInteger fallback,
                                      NSUInteger minimum)
{
    NSUInteger nbytes = env_ns(name, fallback);
    if (nbytes < minimum) {
        nbytes = minimum;
    }
    return align_up_ns(nbytes, 256);
}

static const char *bench_storage_name(void)
{
    return "shared";
}

static MTLResourceOptions bench_resource_options(void)
{
    MTLResourceOptions options = probe_shared_options();
    if (env_is("GD_PROBE_BENCH_HAZARD", "untracked")) {
        options |= MTLResourceHazardTrackingModeUntracked;
    }
    return options;
}

static const char *bench_hazard_name(void)
{
    return env_is("GD_PROBE_BENCH_HAZARD", "untracked") ? "untracked" : "tracked";
}

static double now_s(void)
{
    return CFAbsoluteTimeGetCurrent();
}

static float gelu_tanh_f32(float x)
{
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x3)));
}

static void probe_offset_shared_readback(ProbeMetalContext *ctx)
{
    ProbeMetalArena *dataArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                                    name:@"offset_data"
                                                                capacity:4096
                                                                 options:probe_shared_options()];
    ProbeMetalArena *scratchArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                                       name:@"offset_scratch"
                                                                   capacity:4096
                                                                    options:probe_shared_options()];
    float inVals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float outVals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ProbeTensorRef padA = [dataArena allocBytes:13 alignment:1 name:"data.pad"];
    ProbeTensorRef x = [dataArena allocBytes:4 * sizeof(float) alignment:256 name:"x.shared"];
    ProbeTensorRef padB = [scratchArena allocBytes:29 alignment:1 name:"scratch.pad"];
    ProbeTensorRef y = [scratchArena allocBytes:4 * sizeof(float) alignment:256 name:"y.shared"];
    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc;
    CHECK(padA.nbytes == 13 && padB.nbytes == 29, "padding allocations live");
    CHECK(x.cpu_visible && y.cpu_visible, "shared tensors cpu visible");
    CHECK(x.offset != 0 && y.offset != 0, "tensor refs use nonzero offsets");
    CHECK((x.offset % 256U) == 0U && (y.offset % 256U) == 0U, "tensor offsets aligned");
    write_floats(x, inVals, 4);
    enc = [cmd computeCommandEncoder];
    encode_add_one(enc, ctx->addOnePSO, x, y, 4);
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    CHECK(cmd.status == MTLCommandBufferStatusCompleted, "offset add command completed");
    read_floats(y, outVals, 4);
    assert_float_close(outVals[0], 2.0f, "offset add out[0]");
    assert_float_close(outVals[3], 5.0f, "offset add out[3]");
    printf("  offset/shared readback ok x_off=%lu y_off=%lu\n",
           (unsigned long)x.offset, (unsigned long)y.offset);
}

static MPSMatrix *make_mps_matrix(ProbeTensorRef ref,
                                 NSUInteger rows,
                                 NSUInteger cols,
                                 NSUInteger rowBytes,
                                 MPSDataType dataType)
{
    MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                      columns:cols
                                                                     rowBytes:rowBytes
                                                                     dataType:dataType];
    MPSMatrix *mat = [[MPSMatrix alloc] initWithBuffer:ref.buffer offset:ref.offset
                                            descriptor:desc];
    CHECK(desc != nil && mat != nil, "make_mps_matrix");
    return mat;
}

static MPSMatrix *make_mps_batched_matrix(ProbeTensorRef ref,
                                         NSUInteger matrices,
                                         NSUInteger rows,
                                         NSUInteger cols,
                                         NSUInteger rowBytes,
                                         NSUInteger matrixBytes,
                                         MPSDataType dataType)
{
    MPSMatrixDescriptor *desc;
    MPSMatrix *mat;
    CHECK(matrices > 0, "batched matrix count");
    CHECK(matrixBytes >= rows * rowBytes && (matrixBytes % rowBytes) == 0U,
          "batched matrixBytes valid");
    CHECK(ref.nbytes >= (matrices - 1U) * matrixBytes + rows * rowBytes,
          "batched matrix buffer bounds");
    desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                 columns:cols
                                                matrices:matrices
                                                rowBytes:rowBytes
                                             matrixBytes:matrixBytes
                                                dataType:dataType];
    mat = [[MPSMatrix alloc] initWithBuffer:ref.buffer offset:ref.offset descriptor:desc];
    CHECK(desc != nil && mat != nil, "make_mps_batched_matrix");
    CHECK(desc.matrices == matrices && desc.matrixBytes == matrixBytes,
          "batched descriptor preserves strides");
    return mat;
}

static MPSMatrixMultiplication *make_mps_gemm(id<MTLDevice> device,
                                             NSUInteger M,
                                             NSUInteger K,
                                             NSUInteger N)
{
    MPSMatrixMultiplication *gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                      transposeLeft:NO
                                                                     transposeRight:NO
                                                                         resultRows:M
                                                                      resultColumns:N
                                                                    interiorColumns:K
                                                                              alpha:1.0
                                                                               beta:0.0];
    CHECK(gemm != nil, "make_mps_gemm");
    return gemm;
}

static void probe_shared_direct_write_read(ProbeMetalContext *ctx)
{
    ProbeMetalArena *stateArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                                     name:@"state_shared"
                                                                 capacity:4096
                                                                  options:probe_shared_options()];
    ProbeMetalArena *scratchArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                                       name:@"direct_scratch"
                                                                   capacity:4096
                                                                    options:probe_shared_options()];
    float inVals[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float outVals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ProbeTensorRef state = [stateArena allocBytes:4 * sizeof(float) alignment:256 name:"state.shared"];
    ProbeTensorRef y = [scratchArena allocBytes:4 * sizeof(float) alignment:256 name:"direct.y.shared"];
    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc;
    CHECK(state.cpu_visible && y.cpu_visible, "shared direct tensors cpu visible");
    write_floats(state, inVals, 4);
    enc = [cmd computeCommandEncoder];
    encode_add_one(enc, ctx->addOnePSO, state, y, 4);
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    CHECK(cmd.status == MTLCommandBufferStatusCompleted, "shared direct command completed");
    read_floats(y, outVals, 4);
    assert_float_close(outVals[0], 11.0f, "shared direct out[0]");
    assert_float_close(outVals[3], 41.0f, "shared direct out[3]");
    printf("  shared direct write/read ok\n");
}

static void probe_mps_matmul(ProbeMetalContext *ctx)
{
    enum { M = 3, K = 5, N = 4 };
    NSUInteger rowX;
    NSUInteger rowW;
    NSUInteger rowY;
    NSUInteger i;
    NSUInteger k;
    NSUInteger j;
    float xVals[M * K];
    float wVals[K * N];
    float got[M * N];
    float want[M * N];
    ProbeMetalArena *paramsArena;
    ProbeTensorRef x;
    ProbeTensorRef w;
    ProbeTensorRef y;
    MPSMatrix *xMat;
    MPSMatrix *wMat;
    MPSMatrix *yMat;
    MPSMatrixMultiplication *gemm;
    id<MTLCommandBuffer> cmd;

    if (!MPSSupportsMTLDevice(ctx->device)) {
        printf("  mps matmul skipped (MPS unsupported)\n");
        return;
    }

    rowX = [MPSMatrixDescriptor rowBytesForColumns:K dataType:MPSDataTypeFloat16];
    rowW = [MPSMatrixDescriptor rowBytesForColumns:N dataType:MPSDataTypeFloat16];
    rowY = [MPSMatrixDescriptor rowBytesForColumns:N dataType:MPSDataTypeFloat16];
    CHECK(rowX >= K * sizeof(uint16_t) && rowW >= N * sizeof(uint16_t) &&
          rowY >= N * sizeof(uint16_t), "MPS rowBytes cover rows");

    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            xVals[i * K + k] = (float)(i + k + 1U);
        }
    }
    for (k = 0; k < K; ++k) {
        for (j = 0; j < N; ++j) {
            wVals[k * N + j] = (float)((k + 1U) * (j + 2U));
        }
    }
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (k = 0; k < K; ++k) {
                sum += xVals[i * K + k] * wVals[k * N + j];
            }
            want[i * N + j] = f16_bits_to_f32(f32_to_f16_bits(sum));
            got[i * N + j] = 0.0f;
        }
    }

    paramsArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                     name:@"mps_params"
                                                 capacity:16 * 1024
                                                  options:probe_shared_options()];
    ctx_begin(ctx);
    x = [ctx->data allocBytes:M * rowX alignment:256 name:"mps.x.shared"];
    w = [paramsArena allocBytes:K * rowW alignment:256 name:"mps.w.shared"];
    y = [ctx->scratch allocBytes:M * rowY alignment:256 name:"mps.y.shared"];
    CHECK(x.cpu_visible && w.cpu_visible && y.cpu_visible, "MPS shared storage modes");
    write_matrix_f16_from_f32(x, M, K, rowX, xVals, NULL);
    write_matrix_f16_from_f32(w, K, N, rowW, wVals, NULL);

    xMat = make_mps_matrix(x, M, K, rowX, MPSDataTypeFloat16);
    wMat = make_mps_matrix(w, K, N, rowW, MPSDataTypeFloat16);
    yMat = make_mps_matrix(y, M, N, rowY, MPSDataTypeFloat16);
    gemm = make_mps_gemm(ctx->device, M, K, N);
    [gemm encodeToCommandBuffer:ctx->currentCommand leftMatrix:xMat rightMatrix:wMat
                   resultMatrix:yMat];
    cmd = ctx_end(ctx);
    [cmd waitUntilCompleted];
    CHECK(cmd.status == MTLCommandBufferStatusCompleted, "MPS matmul command completed");
    read_matrix_f16_to_f32(y, M, N, rowY, got);
    for (i = 0; i < M * N; ++i) {
        assert_float_near(got[i], want[i], 1.0e-3f, "MPS matmul result");
    }
    printf("  MPS matmul ok dtype=f16 M=%lu K=%lu N=%lu rowBytes=(%lu,%lu,%lu)\n",
           (unsigned long)M, (unsigned long)K, (unsigned long)N,
           (unsigned long)rowX, (unsigned long)rowW, (unsigned long)rowY);
}

static void probe_batched_mps_matmul(ProbeMetalContext *ctx)
{
    enum { B = 4, M = 3, K = 5, N = 6 };
    NSUInteger rowX;
    NSUInteger rowW;
    NSUInteger rowY;
    NSUInteger matrixX;
    NSUInteger matrixW;
    NSUInteger matrixY;
    NSUInteger bidx;
    NSUInteger i;
    NSUInteger k;
    NSUInteger j;
    uint64_t rng = UINT64_C(0x0b17c4ed5eed1234);
    float xRaw[B * M * K];
    float wRaw[B * K * N];
    float xVals[B * M * K];
    float wVals[B * K * N];
    float got[B * M * N];
    float want[B * M * N];
    ProbeMetalArena *paramsArena;
    ProbeTensorRef padData;
    ProbeTensorRef padParams;
    ProbeTensorRef padScratch;
    ProbeTensorRef x;
    ProbeTensorRef w;
    ProbeTensorRef y;
    MPSMatrix *xMat;
    MPSMatrix *wMat;
    MPSMatrix *yMat;
    MPSMatrixMultiplication *gemm;
    id<MTLCommandBuffer> cmd;

    if (!MPSSupportsMTLDevice(ctx->device)) {
        printf("  batched mps matmul skipped (MPS unsupported)\n");
        return;
    }

    rowX = [MPSMatrixDescriptor rowBytesForColumns:K dataType:MPSDataTypeFloat16];
    rowW = [MPSMatrixDescriptor rowBytesForColumns:N dataType:MPSDataTypeFloat16];
    rowY = [MPSMatrixDescriptor rowBytesForColumns:N dataType:MPSDataTypeFloat16];
    matrixX = (M + 1U) * rowX;
    matrixW = (K + 2U) * rowW;
    matrixY = (M + 3U) * rowY;
    CHECK(rowX >= K * sizeof(uint16_t) && rowW >= N * sizeof(uint16_t) &&
          rowY >= N * sizeof(uint16_t), "batched MPS rowBytes cover rows");

    for (bidx = 0; bidx < B; ++bidx) {
        for (i = 0; i < M; ++i) {
            for (k = 0; k < K; ++k) {
                xRaw[(bidx * M + i) * K + k] = rand_uniform_signed(&rng, 0.75f);
            }
        }
        for (k = 0; k < K; ++k) {
            for (j = 0; j < N; ++j) {
                wRaw[(bidx * K + k) * N + j] = rand_uniform_signed(&rng, 0.25f);
            }
        }
    }

    paramsArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                     name:@"batched_mps_params"
                                                 capacity:B * matrixW + 4096
                                                  options:probe_shared_options()];
    ctx_begin(ctx);
    padData = [ctx->data allocBytes:19 alignment:1 name:"batched.data.pad"];
    x = [ctx->data allocBytes:B * matrixX alignment:256 name:"batched.x.shared"];
    padParams = [paramsArena allocBytes:23 alignment:1 name:"batched.params.pad"];
    w = [paramsArena allocBytes:B * matrixW alignment:256 name:"batched.w.shared"];
    padScratch = [ctx->scratch allocBytes:29 alignment:1 name:"batched.scratch.pad"];
    y = [ctx->scratch allocBytes:B * matrixY alignment:256 name:"batched.y.shared"];
    CHECK(padData.nbytes == 19 && padParams.nbytes == 23 && padScratch.nbytes == 29,
          "batched padding allocations live");
    CHECK(x.offset != 0 && w.offset != 0 && y.offset != 0, "batched offsets nonzero");
    CHECK((x.offset % 256U) == 0U && (w.offset % 256U) == 0U &&
          (y.offset % 256U) == 0U, "batched offsets aligned");

    CHECK(x.cpu_visible && w.cpu_visible && y.cpu_visible, "batched MPS shared storage modes");
    for (bidx = 0; bidx < B; ++bidx) {
        ProbeTensorRef xb = tensor_subref(x, bidx * matrixX, M * rowX, "batched.x[b]");
        ProbeTensorRef wb = tensor_subref(w, bidx * matrixW, K * rowW, "batched.w[b]");
        write_matrix_f16_from_f32(xb, M, K, rowX, xRaw + bidx * M * K,
                                  xVals + bidx * M * K);
        write_matrix_f16_from_f32(wb, K, N, rowW, wRaw + bidx * K * N,
                                  wVals + bidx * K * N);
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (k = 0; k < K; ++k) {
                    sum += xVals[(bidx * M + i) * K + k] *
                           wVals[(bidx * K + k) * N + j];
                }
                want[(bidx * M + i) * N + j] = f16_bits_to_f32(f32_to_f16_bits(sum));
                got[(bidx * M + i) * N + j] = 0.0f;
            }
        }
    }

    xMat = make_mps_batched_matrix(x, B, M, K, rowX, matrixX, MPSDataTypeFloat16);
    wMat = make_mps_batched_matrix(w, B, K, N, rowW, matrixW, MPSDataTypeFloat16);
    yMat = make_mps_batched_matrix(y, B, M, N, rowY, matrixY, MPSDataTypeFloat16);
    gemm = make_mps_gemm(ctx->device, M, K, N);
    gemm.batchStart = 0;
    gemm.batchSize = B;
    [gemm encodeToCommandBuffer:ctx->currentCommand leftMatrix:xMat rightMatrix:wMat
                   resultMatrix:yMat];
    cmd = ctx_end(ctx);
    [cmd waitUntilCompleted];
    CHECK(cmd.status == MTLCommandBufferStatusCompleted, "batched MPS matmul command completed");

    for (bidx = 0; bidx < B; ++bidx) {
        ProbeTensorRef yb = tensor_subref(y, bidx * matrixY, M * rowY, "batched.y[b]");
        read_matrix_f16_to_f32(yb, M, N, rowY, got + bidx * M * N);
    }
    for (i = 0; i < B * M * N; ++i) {
        assert_float_near(got[i], want[i], 2.0e-3f, "batched MPS matmul result");
    }
    printf("  batched MPS matmul ok dtype=f16 B=%lu M=%lu K=%lu N=%lu rowBytes=(%lu,%lu,%lu) matrixBytes=(%lu,%lu,%lu)\n",
           (unsigned long)B, (unsigned long)M, (unsigned long)K, (unsigned long)N,
           (unsigned long)rowX, (unsigned long)rowW, (unsigned long)rowY,
           (unsigned long)matrixX, (unsigned long)matrixW, (unsigned long)matrixY);
}

static void probe_strided_sliced_mps_matmul(ProbeMetalContext *ctx)
{
    enum { M = 3, K = 4, N = 5 };
    const NSUInteger leftRowOrigin = 2;
    const NSUInteger leftColOrigin = 1;
    const NSUInteger rightRowOrigin = 1;
    const NSUInteger rightColOrigin = 2;
    const NSUInteger resultRowOrigin = 3;
    const NSUInteger resultColOrigin = 1;
    const NSUInteger leftRows = leftRowOrigin + M + 2;
    const NSUInteger leftCols = leftColOrigin + K + 3;
    const NSUInteger rightRows = rightRowOrigin + K + 2;
    const NSUInteger rightCols = rightColOrigin + N + 2;
    const NSUInteger resultRows = resultRowOrigin + M + 2;
    const NSUInteger resultCols = resultColOrigin + N + 3;
    NSUInteger rowX;
    NSUInteger rowW;
    NSUInteger rowY;
    NSUInteger i;
    NSUInteger k;
    NSUInteger j;
    uint64_t rng = UINT64_C(0x5171ded51ced1234);
    float xRaw[M * K];
    float wRaw[K * N];
    float xVals[M * K];
    float wVals[K * N];
    float got[M * N];
    float want[M * N];
    ProbeMetalArena *paramsArena;
    ProbeTensorRef padData;
    ProbeTensorRef padParams;
    ProbeTensorRef padScratch;
    ProbeTensorRef xBase;
    ProbeTensorRef wBase;
    ProbeTensorRef yBase;
    ProbeTensorRef xSlice;
    ProbeTensorRef wSlice;
    ProbeTensorRef ySlice;
    MPSMatrix *xMat;
    MPSMatrix *wMat;
    MPSMatrix *yMat;
    MPSMatrixMultiplication *gemm;
    id<MTLCommandBuffer> cmd;

    if (!MPSSupportsMTLDevice(ctx->device)) {
        printf("  strided/sliced mps matmul skipped (MPS unsupported)\n");
        return;
    }

    rowX = [MPSMatrixDescriptor rowBytesForColumns:leftCols dataType:MPSDataTypeFloat16] + 64U;
    rowW = [MPSMatrixDescriptor rowBytesForColumns:rightCols dataType:MPSDataTypeFloat16] + 64U;
    rowY = [MPSMatrixDescriptor rowBytesForColumns:resultCols dataType:MPSDataTypeFloat16] + 64U;
    CHECK(rowX > leftCols * sizeof(uint16_t) && rowW > rightCols * sizeof(uint16_t) &&
          rowY > resultCols * sizeof(uint16_t), "strided rows have padding");
    CHECK((rowX % sizeof(uint16_t)) == 0U && (rowW % sizeof(uint16_t)) == 0U &&
          (rowY % sizeof(uint16_t)) == 0U, "strided rowBytes aligned");

    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            xRaw[i * K + k] = rand_uniform_signed(&rng, 0.75f);
        }
    }
    for (k = 0; k < K; ++k) {
        for (j = 0; j < N; ++j) {
            wRaw[k * N + j] = rand_uniform_signed(&rng, 0.35f);
        }
    }

    paramsArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                     name:@"strided_mps_params"
                                                 capacity:rightRows * rowW + 4096
                                                  options:probe_shared_options()];
    ctx_begin(ctx);
    padData = [ctx->data allocBytes:17 alignment:1 name:"strided.data.pad"];
    xBase = [ctx->data allocBytes:leftRows * rowX alignment:256 name:"strided.x.base"];
    padParams = [paramsArena allocBytes:23 alignment:1 name:"strided.params.pad"];
    wBase = [paramsArena allocBytes:rightRows * rowW alignment:256 name:"strided.w.base"];
    padScratch = [ctx->scratch allocBytes:31 alignment:1 name:"strided.scratch.pad"];
    yBase = [ctx->scratch allocBytes:resultRows * rowY alignment:256 name:"strided.y.base"];
    CHECK(padData.nbytes == 17 && padParams.nbytes == 23 && padScratch.nbytes == 31,
          "strided padding allocations live");
    CHECK(xBase.cpu_visible && wBase.cpu_visible && yBase.cpu_visible,
          "strided shared storage modes");
    CHECK((xBase.offset % 256U) == 0U && (wBase.offset % 256U) == 0U &&
          (yBase.offset % 256U) == 0U, "strided offsets aligned");

    fill_matrix_f16(xBase, leftRows, leftCols, rowX, -11.0f);
    fill_matrix_f16(wBase, rightRows, rightCols, rowW, 13.0f);
    fill_matrix_f16(yBase, resultRows, resultCols, rowY, -1234.0f);

    xSlice = tensor_subref(xBase, leftRowOrigin * rowX + leftColOrigin * sizeof(uint16_t),
                           xBase.nbytes - leftRowOrigin * rowX - leftColOrigin * sizeof(uint16_t),
                           "strided.x.slice");
    wSlice = tensor_subref(wBase, rightRowOrigin * rowW + rightColOrigin * sizeof(uint16_t),
                           wBase.nbytes - rightRowOrigin * rowW - rightColOrigin * sizeof(uint16_t),
                           "strided.w.slice");
    ySlice = tensor_subref(yBase, resultRowOrigin * rowY + resultColOrigin * sizeof(uint16_t),
                           yBase.nbytes - resultRowOrigin * rowY - resultColOrigin * sizeof(uint16_t),
                           "strided.y.slice");
    write_matrix_f16_from_f32(xSlice, M, K, rowX, xRaw, xVals);
    write_matrix_f16_from_f32(wSlice, K, N, rowW, wRaw, wVals);
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (k = 0; k < K; ++k) {
                sum += xVals[i * K + k] * wVals[k * N + j];
            }
            want[i * N + j] = f16_bits_to_f32(f32_to_f16_bits(sum));
            got[i * N + j] = 0.0f;
        }
    }

    xMat = make_mps_matrix(xBase, leftRows, leftCols, rowX, MPSDataTypeFloat16);
    wMat = make_mps_matrix(wBase, rightRows, rightCols, rowW, MPSDataTypeFloat16);
    yMat = make_mps_matrix(yBase, resultRows, resultCols, rowY, MPSDataTypeFloat16);
    gemm = make_mps_gemm(ctx->device, M, K, N);
    gemm.leftMatrixOrigin = MTLOriginMake(leftRowOrigin, leftColOrigin, 0);
    gemm.rightMatrixOrigin = MTLOriginMake(rightRowOrigin, rightColOrigin, 0);
    gemm.resultMatrixOrigin = MTLOriginMake(resultRowOrigin, resultColOrigin, 0);
    [gemm encodeToCommandBuffer:ctx->currentCommand leftMatrix:xMat rightMatrix:wMat
                   resultMatrix:yMat];
    cmd = ctx_end(ctx);
    [cmd waitUntilCompleted];
    CHECK(cmd.status == MTLCommandBufferStatusCompleted,
          "strided/sliced MPS matmul command completed");

    read_matrix_f16_to_f32(ySlice, M, N, rowY, got);
    for (i = 0; i < M * N; ++i) {
        assert_float_near(got[i], want[i], 2.0e-3f,
                          "strided/sliced MPS matmul result");
    }
    assert_float_near(matrix_value_f16(yBase, 0, 0, rowY),
                      f16_bits_to_f32(f32_to_f16_bits(-1234.0f)), 1.0e-3f,
                      "strided result before origin preserved");
    assert_float_near(matrix_value_f16(yBase, resultRowOrigin, 0, rowY),
                      f16_bits_to_f32(f32_to_f16_bits(-1234.0f)), 1.0e-3f,
                      "strided result left of origin preserved");
    printf("  strided/sliced MPS matmul ok dtype=f16 M=%lu K=%lu N=%lu origins=(%lu,%lu)/(%lu,%lu)/(%lu,%lu) rowBytes=(%lu,%lu,%lu)\n",
           (unsigned long)M, (unsigned long)K, (unsigned long)N,
           (unsigned long)leftRowOrigin, (unsigned long)leftColOrigin,
           (unsigned long)rightRowOrigin, (unsigned long)rightColOrigin,
           (unsigned long)resultRowOrigin, (unsigned long)resultColOrigin,
           (unsigned long)rowX, (unsigned long)rowW, (unsigned long)rowY);
}

static void probe_linear_bias_gelu_same_command(ProbeMetalContext *ctx)
{
    enum { M = 4, K = 7, N = 6 };
    NSUInteger rowX;
    NSUInteger rowW;
    NSUInteger rowY;
    NSUInteger rowZ;
    NSUInteger i;
    NSUInteger k;
    NSUInteger j;
    uint64_t rng = UINT64_C(0x911eafb1a5ed1234);
    float xRaw[M * K];
    float wRaw[K * N];
    float bRaw[N];
    float xVals[M * K];
    float wVals[K * N];
    float bVals[N];
    float got[M * N];
    float want[M * N];
    ProbeMetalArena *paramsArena;
    ProbeTensorRef padData;
    ProbeTensorRef padParams;
    ProbeTensorRef padScratch;
    ProbeTensorRef x;
    ProbeTensorRef w;
    ProbeTensorRef bias;
    ProbeTensorRef y;
    ProbeTensorRef z;
    MPSMatrix *xMat;
    MPSMatrix *wMat;
    MPSMatrix *yMat;
    MPSMatrixMultiplication *gemm;
    id<MTLComputeCommandEncoder> enc;
    id<MTLCommandBuffer> cmd;

    if (!MPSSupportsMTLDevice(ctx->device)) {
        printf("  linear+bias+gelu skipped (MPS unsupported)\n");
        return;
    }

    rowX = [MPSMatrixDescriptor rowBytesForColumns:K dataType:MPSDataTypeFloat16];
    rowW = [MPSMatrixDescriptor rowBytesForColumns:N dataType:MPSDataTypeFloat16];
    rowY = [MPSMatrixDescriptor rowBytesForColumns:N dataType:MPSDataTypeFloat16];
    rowZ = [MPSMatrixDescriptor rowBytesForColumns:N dataType:MPSDataTypeFloat16] + 64U;
    CHECK(rowX >= K * sizeof(uint16_t) && rowW >= N * sizeof(uint16_t) &&
          rowY >= N * sizeof(uint16_t) && rowZ > N * sizeof(uint16_t),
          "linear+bias+gelu rowBytes");

    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            xRaw[i * K + k] = rand_uniform_signed(&rng, 0.5f);
        }
    }
    for (k = 0; k < K; ++k) {
        for (j = 0; j < N; ++j) {
            wRaw[k * N + j] = rand_uniform_signed(&rng, 0.25f);
        }
    }
    for (j = 0; j < N; ++j) {
        bRaw[j] = rand_uniform_signed(&rng, 0.1f);
    }

    paramsArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                     name:@"linear_bias_gelu_params"
                                                 capacity:K * rowW + N * sizeof(uint16_t) + 4096
                                                  options:probe_shared_options()];
    ctx_begin(ctx);
    padData = [ctx->data allocBytes:11 alignment:1 name:"linear.data.pad"];
    x = [ctx->data allocBytes:M * rowX alignment:256 name:"linear.x.shared"];
    padParams = [paramsArena allocBytes:13 alignment:1 name:"linear.params.pad"];
    w = [paramsArena allocBytes:K * rowW alignment:256 name:"linear.w.shared"];
    bias = [paramsArena allocBytes:N * sizeof(uint16_t) alignment:256 name:"linear.bias.shared"];
    padScratch = [ctx->scratch allocBytes:17 alignment:1 name:"linear.scratch.pad"];
    y = [ctx->scratch allocBytes:M * rowY alignment:256 name:"linear.y.shared"];
    z = [ctx->scratch allocBytes:M * rowZ alignment:256 name:"linear.z.shared"];
    CHECK(padData.nbytes == 11 && padParams.nbytes == 13 && padScratch.nbytes == 17,
          "linear padding allocations live");
    CHECK(x.cpu_visible && w.cpu_visible && bias.cpu_visible && y.cpu_visible && z.cpu_visible,
          "linear shared storage modes");
    CHECK((x.offset % 256U) == 0U && (w.offset % 256U) == 0U &&
          (bias.offset % 256U) == 0U && (y.offset % 256U) == 0U &&
          (z.offset % 256U) == 0U, "linear offsets aligned");
    write_matrix_f16_from_f32(x, M, K, rowX, xRaw, xVals);
    write_matrix_f16_from_f32(w, K, N, rowW, wRaw, wVals);
    write_vector_f16_from_f32(bias, bRaw, bVals, N);
    fill_matrix_f16(z, M, N, rowZ, -999.0f);
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (k = 0; k < K; ++k) {
                sum += xVals[i * K + k] * wVals[k * N + j];
            }
            sum = f16_bits_to_f32(f32_to_f16_bits(sum));
            want[i * N + j] = f16_bits_to_f32(f32_to_f16_bits(gelu_tanh_f32(sum + bVals[j])));
            got[i * N + j] = 0.0f;
        }
    }

    xMat = make_mps_matrix(x, M, K, rowX, MPSDataTypeFloat16);
    wMat = make_mps_matrix(w, K, N, rowW, MPSDataTypeFloat16);
    yMat = make_mps_matrix(y, M, N, rowY, MPSDataTypeFloat16);
    gemm = make_mps_gemm(ctx->device, M, K, N);
    [gemm encodeToCommandBuffer:ctx->currentCommand leftMatrix:xMat rightMatrix:wMat
                   resultMatrix:yMat];
    enc = [ctx->currentCommand computeCommandEncoder];
    encode_bias_gelu_f16(enc, ctx->biasGeluPSO, y, bias, z, M, N,
                         (uint32_t)rowY, (uint32_t)rowZ);
    [enc endEncoding];
    cmd = ctx_end(ctx);
    [cmd waitUntilCompleted];
    CHECK(cmd.status == MTLCommandBufferStatusCompleted,
          "linear+bias+gelu command completed");

    read_matrix_f16_to_f32(z, M, N, rowZ, got);
    for (i = 0; i < M * N; ++i) {
        assert_float_near(got[i], want[i], 3.0e-3f,
                          "linear+bias+gelu result");
    }
    printf("  linear+bias+gelu same command ok dtype=f16 M=%lu K=%lu N=%lu rowBytes=(%lu,%lu,%lu,%lu)\n",
           (unsigned long)M, (unsigned long)K, (unsigned long)N,
           (unsigned long)rowX, (unsigned long)rowW, (unsigned long)rowY,
           (unsigned long)rowZ);
}

static void probe_norm_reduction_case(ProbeMetalContext *ctx, NSUInteger cols)
{
    enum { ROWS = 5, MAX_D = 768 };
    const float eps = 1.0e-5f;
    NSUInteger rowX;
    NSUInteger rowRms;
    NSUInteger rowLayer;
    NSUInteger r;
    NSUInteger c;
    uint64_t rng = UINT64_C(0x90f1a5edacc01234) ^ (uint64_t)cols;
    float xRaw[ROWS * MAX_D];
    float xVals[ROWS * MAX_D];
    float scaleRaw[MAX_D];
    float scaleVals[MAX_D];
    float gammaRaw[MAX_D];
    float gammaVals[MAX_D];
    float betaRaw[MAX_D];
    float betaVals[MAX_D];
    float gotRms[ROWS * MAX_D];
    float gotLayer[ROWS * MAX_D];
    float wantRms[ROWS * MAX_D];
    float wantLayer[ROWS * MAX_D];
    float rmsInvs[ROWS];
    ProbeMetalArena *paramsArena;
    ProbeTensorRef padData;
    ProbeTensorRef padParams;
    ProbeTensorRef padScratch;
    ProbeTensorRef x;
    ProbeTensorRef scale;
    ProbeTensorRef gamma;
    ProbeTensorRef beta;
    ProbeTensorRef yRms;
    ProbeTensorRef yLayer;
    id<MTLComputeCommandEncoder> enc;
    id<MTLCommandBuffer> cmd;

    CHECK(cols <= MAX_D, "norm cols max");
    rowX = [MPSMatrixDescriptor rowBytesForColumns:cols dataType:MPSDataTypeFloat16] + 64U;
    rowRms = [MPSMatrixDescriptor rowBytesForColumns:cols dataType:MPSDataTypeFloat16] + 96U;
    rowLayer = [MPSMatrixDescriptor rowBytesForColumns:cols dataType:MPSDataTypeFloat16] + 128U;
    CHECK(rowX > cols * sizeof(uint16_t) && rowRms > cols * sizeof(uint16_t) &&
          rowLayer > cols * sizeof(uint16_t), "norm padded rowBytes");

    for (r = 0; r < ROWS; ++r) {
        for (c = 0; c < cols; ++c) {
            xRaw[r * cols + c] = rand_uniform_signed(&rng, 0.75f);
        }
    }
    for (c = 0; c < cols; ++c) {
        scaleRaw[c] = 0.75f + rand_uniform_signed(&rng, 0.25f);
        gammaRaw[c] = 0.80f + rand_uniform_signed(&rng, 0.20f);
        betaRaw[c] = rand_uniform_signed(&rng, 0.05f);
    }

    paramsArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                     name:@"norm_params"
                                                 capacity:3 * cols * sizeof(uint16_t) + 4096
                                                  options:probe_shared_options()];
    ctx_begin(ctx);
    padData = [ctx->data allocBytes:23 alignment:1 name:"norm.data.pad"];
    x = [ctx->data allocBytes:ROWS * rowX alignment:256 name:"norm.x.shared"];
    padParams = [paramsArena allocBytes:29 alignment:1 name:"norm.params.pad"];
    scale = [paramsArena allocBytes:cols * sizeof(uint16_t) alignment:256 name:"norm.scale.shared"];
    gamma = [paramsArena allocBytes:cols * sizeof(uint16_t) alignment:256 name:"norm.gamma.shared"];
    beta = [paramsArena allocBytes:cols * sizeof(uint16_t) alignment:256 name:"norm.beta.shared"];
    padScratch = [ctx->scratch allocBytes:31 alignment:1 name:"norm.scratch.pad"];
    yRms = [ctx->scratch allocBytes:ROWS * rowRms alignment:256 name:"norm.rms.y.shared"];
    yLayer = [ctx->scratch allocBytes:ROWS * rowLayer alignment:256 name:"norm.layer.y.shared"];
    CHECK(padData.nbytes == 23 && padParams.nbytes == 29 && padScratch.nbytes == 31,
          "norm padding allocations live");
    CHECK(x.cpu_visible && scale.cpu_visible && gamma.cpu_visible && beta.cpu_visible &&
          yRms.cpu_visible && yLayer.cpu_visible, "norm shared storage modes");
    CHECK((x.offset % 256U) == 0U && (scale.offset % 256U) == 0U &&
          (gamma.offset % 256U) == 0U && (beta.offset % 256U) == 0U &&
          (yRms.offset % 256U) == 0U && (yLayer.offset % 256U) == 0U,
          "norm offsets aligned");

    fill_matrix_f16(x, ROWS, rowX / sizeof(uint16_t), rowX, -222.0f);
    fill_matrix_f16(yRms, ROWS, rowRms / sizeof(uint16_t), rowRms, -333.0f);
    fill_matrix_f16(yLayer, ROWS, rowLayer / sizeof(uint16_t), rowLayer, -444.0f);
    write_matrix_f16_from_f32(x, ROWS, cols, rowX, xRaw, NULL);
    write_vector_f16_from_f32(scale, scaleRaw, NULL, cols);
    write_vector_f16_from_f32(gamma, gammaRaw, NULL, cols);
    write_vector_f16_from_f32(beta, betaRaw, NULL, cols);
    read_matrix_f16_to_f32(x, ROWS, cols, rowX, xVals);
    read_vector_f16_to_f32(scale, scaleVals, cols);
    read_vector_f16_to_f32(gamma, gammaVals, cols);
    read_vector_f16_to_f32(beta, betaVals, cols);

    for (r = 0; r < ROWS; ++r) {
        float sum = 0.0f;
        float ss = 0.0f;
        float rmsInv;
        float mean;
        float layerInv;
        for (c = 0; c < cols; ++c) {
            float v = xVals[r * cols + c];
            sum += v;
            ss += v * v;
        }
        rmsInv = 1.0f / sqrtf(ss / (float)cols + eps);
        rmsInvs[r] = rmsInv;
        mean = sum / (float)cols;
        layerInv = 1.0f / sqrtf(fmaxf(ss / (float)cols - mean * mean, 0.0f) + eps);
        for (c = 0; c < cols; ++c) {
            float v = xVals[r * cols + c];
            wantRms[r * cols + c] = f16_bits_to_f32(
                f32_to_f16_bits(v * rmsInv * scaleVals[c]));
            wantLayer[r * cols + c] = f16_bits_to_f32(
                f32_to_f16_bits((v - mean) * layerInv * gammaVals[c] + betaVals[c]));
            gotRms[r * cols + c] = 0.0f;
            gotLayer[r * cols + c] = 0.0f;
        }
    }

    enc = [ctx->currentCommand computeCommandEncoder];
    encode_rms_norm_f16(enc, ctx->rmsNormPSO, x, scale, yRms, ROWS, (uint32_t)cols,
                        (uint32_t)rowX, (uint32_t)rowRms, eps);
    [enc endEncoding];
    enc = [ctx->currentCommand computeCommandEncoder];
    encode_layer_norm_f16(enc, ctx->layerNormPSO, x, gamma, beta, yLayer, ROWS,
                          (uint32_t)cols, (uint32_t)rowX, (uint32_t)rowLayer, eps);
    [enc endEncoding];
    cmd = ctx_end(ctx);
    [cmd waitUntilCompleted];
    CHECK(cmd.status == MTLCommandBufferStatusCompleted,
          "norm reduction command completed");

    read_matrix_f16_to_f32(yRms, ROWS, cols, rowRms, gotRms);
    read_matrix_f16_to_f32(yLayer, ROWS, cols, rowLayer, gotLayer);
    for (r = 0; r < ROWS; ++r) {
        for (c = 0; c < cols; ++c) {
            if (fabsf(gotRms[r * cols + c] - wantRms[r * cols + c]) > 5.0e-3f) {
                fprintf(stderr, "metal probe failed: rms_norm_f16 result cols=%lu row=%lu col=%lu got=%g want=%g x=%g scale=%g inv=%g raw=%g\n",
                        (unsigned long)cols, (unsigned long)r, (unsigned long)c,
                        (double)gotRms[r * cols + c], (double)wantRms[r * cols + c],
                        (double)xVals[r * cols + c], (double)scaleVals[c],
                        (double)rmsInvs[r],
                        (double)(xVals[r * cols + c] * rmsInvs[r] * scaleVals[c]));
                exit(1);
            }
            assert_float_near(gotLayer[r * cols + c], wantLayer[r * cols + c], 5.0e-3f,
                              "layer_norm_f16 result");
        }
    }
    assert_float_near(matrix_value_f16(yRms, 0, cols, rowRms),
                      f16_bits_to_f32(f32_to_f16_bits(-333.0f)), 1.0e-3f,
                      "rms_norm padding preserved");
    assert_float_near(matrix_value_f16(yLayer, 0, cols, rowLayer),
                      f16_bits_to_f32(f32_to_f16_bits(-444.0f)), 1.0e-3f,
                      "layer_norm padding preserved");
    printf("  norm reductions ok dtype=f16 rows=%u cols=%lu rowBytes=(%lu,%lu,%lu)\n",
           (unsigned)ROWS, (unsigned long)cols, (unsigned long)rowX,
           (unsigned long)rowRms, (unsigned long)rowLayer);
}

static void probe_norm_reductions(ProbeMetalContext *ctx)
{
    probe_norm_reduction_case(ctx, 256);
    probe_norm_reduction_case(ctx, 512);
    probe_norm_reduction_case(ctx, 768);
}

static void run_mps_batched_gemm_bench(ProbeMetalContext *ctx,
                                       const char *label,
                                       NSUInteger batch,
                                       NSUInteger M,
                                       NSUInteger K,
                                       NSUInteger N,
                                       int warmup,
                                       int iters)
{
    NSUInteger rowPad = bench_row_pad_bytes();
    NSUInteger batchPadRows = bench_batch_pad_rows();
    NSUInteger align = bench_alloc_alignment();
    NSUInteger prefix = bench_prefix_pad();
    NSUInteger rowA = [MPSMatrixDescriptor rowBytesForColumns:K dataType:MPSDataTypeFloat16] + rowPad;
    NSUInteger rowB = [MPSMatrixDescriptor rowBytesForColumns:N dataType:MPSDataTypeFloat16] + rowPad;
    NSUInteger rowC = [MPSMatrixDescriptor rowBytesForColumns:N dataType:MPSDataTypeFloat16] + rowPad;
    NSUInteger matrixA = (M + batchPadRows) * rowA;
    NSUInteger matrixB = (K + batchPadRows) * rowB;
    NSUInteger matrixC = (M + batchPadRows) * rowC;
    NSUInteger aBytes = batch * matrixA;
    NSUInteger bBytes = batch * matrixB;
    NSUInteger cBytes = batch * matrixC;
    ProbeMetalArena *actArena;
    ProbeMetalArena *rhsArena;
    ProbeTensorRef a;
    ProbeTensorRef b;
    ProbeTensorRef c;
    MPSMatrix *aMat;
    MPSMatrix *bMat;
    MPSMatrix *cMat;
    MPSMatrixMultiplication *gemm;
    uint64_t rng = UINT64_C(0x5ca1ab1ebadc0de5) ^ ((uint64_t)batch << 48U) ^
                   ((uint64_t)M << 32U) ^ ((uint64_t)K << 16U) ^ (uint64_t)N;
    double gpu_total = 0.0;
    double wall_total = 0.0;
    int gpu_count = 0;
    int iter;
    double flops = 2.0 * (double)batch * (double)M * (double)N * (double)K;

    CHECK(batch > 0 && M > 0 && K > 0 && N > 0, "batched bench dims");
    actArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                  name:@"bench_batched_activations"
                                              capacity:prefix + aBytes + cBytes + align + 8192
                                               options:bench_resource_options()];
    rhsArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                  name:@"bench_batched_rhs"
                                              capacity:prefix + bBytes + align + 4096
                                               options:bench_resource_options()];
    if (prefix > 0U) {
        (void)[actArena allocBytes:prefix alignment:1 name:"bench.batch.act.prefix"];
        (void)[rhsArena allocBytes:prefix alignment:1 name:"bench.batch.rhs.prefix"];
    }
    a = [actArena allocBytes:aBytes alignment:align name:"bench.batch.A"];
    b = [rhsArena allocBytes:bBytes alignment:align name:"bench.batch.B"];
    c = [actArena allocBytes:cBytes alignment:align name:"bench.batch.C"];
    CHECK(a.cpu_visible && b.cpu_visible && c.cpu_visible, "batched bench shared tensors");
    aMat = make_mps_batched_matrix(a, batch, M, K, rowA, matrixA, MPSDataTypeFloat16);
    bMat = make_mps_batched_matrix(b, batch, K, N, rowB, matrixB, MPSDataTypeFloat16);
    cMat = make_mps_batched_matrix(c, batch, M, N, rowC, matrixC, MPSDataTypeFloat16);
    gemm = make_mps_gemm(ctx->device, M, K, N);
    gemm.batchStart = 0;
    gemm.batchSize = batch;

    write_batched_matrix_f16_random(a, batch, M, K, rowA, matrixA, &rng, 0.5f);
    write_batched_matrix_f16_random(b, batch, K, N, rowB, matrixB, &rng, 0.05f);

    for (iter = -warmup; iter < iters; ++iter) {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        double t0 = now_s();
        CHECK(cmd != nil, "batched bench command");
        [gemm encodeToCommandBuffer:cmd leftMatrix:aMat rightMatrix:bMat resultMatrix:cMat];
        [cmd commit];
        [cmd waitUntilCompleted];
        CHECK(cmd.status == MTLCommandBufferStatusCompleted, "batched bench command completed");
        if (iter >= 0) {
            double wall = now_s() - t0;
            double gpu = cmd.GPUEndTime - cmd.GPUStartTime;
            wall_total += wall;
            if (gpu > 0.0) {
                gpu_total += gpu;
                gpu_count += 1;
            }
        }
    }
    {
        double wall_ms = (wall_total / (double)iters) * 1000.0;
        double gpu_ms = gpu_count > 0 ? (gpu_total / (double)gpu_count) * 1000.0 : 0.0;
        double denom_s = gpu_ms > 0.0 ? gpu_ms / 1000.0 : wall_ms / 1000.0;
        double tflops = denom_s > 0.0 ? flops / denom_s / 1.0e12 : 0.0;
        double out_mb = (double)cBytes / (1024.0 * 1024.0);
        printf("  bench_mps_batch %-18s B=%3lu M=%4lu K=%4lu N=%5lu gpu_ms=%7.3f wall_ms=%7.3f tflops=%6.2f out_mb=%6.1f\n",
               label, (unsigned long)batch, (unsigned long)M, (unsigned long)K,
               (unsigned long)N, gpu_ms, wall_ms, tflops, out_mb);
    }
}

static void run_mps_gemm_bench(ProbeMetalContext *ctx,
                               const char *label,
                               NSUInteger M,
                               NSUInteger K,
                               NSUInteger N,
                               int warmup,
                               int iters)
{
    NSUInteger rowPad = bench_row_pad_bytes();
    NSUInteger align = bench_alloc_alignment();
    NSUInteger prefix = bench_prefix_pad();
    NSUInteger rowA = [MPSMatrixDescriptor rowBytesForColumns:K dataType:MPSDataTypeFloat16] + rowPad;
    NSUInteger rowB = [MPSMatrixDescriptor rowBytesForColumns:N dataType:MPSDataTypeFloat16] + rowPad;
    NSUInteger rowC = [MPSMatrixDescriptor rowBytesForColumns:N dataType:MPSDataTypeFloat16] + rowPad;
    NSUInteger aBytes = M * rowA;
    NSUInteger bBytes = K * rowB;
    NSUInteger cBytes = M * rowC;
    ProbeMetalArena *actArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                                    name:@"bench_activations"
                                                                capacity:prefix + aBytes + cBytes + align + 8192
                                                                 options:bench_resource_options()];
    ProbeMetalArena *paramArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                                      name:@"bench_params"
                                                                  capacity:prefix + bBytes + align + 4096
                                                                   options:bench_resource_options()];
    ProbeTensorRef a;
    ProbeTensorRef b;
    ProbeTensorRef c;
    if (prefix > 0U) {
        (void)[actArena allocBytes:prefix alignment:1 name:"bench.act.prefix"];
        (void)[paramArena allocBytes:prefix alignment:1 name:"bench.params.prefix"];
    }
    a = [actArena allocBytes:aBytes alignment:align name:"bench.A"];
    b = [paramArena allocBytes:bBytes alignment:align name:"bench.B"];
    c = [actArena allocBytes:cBytes alignment:align name:"bench.C"];
    CHECK(a.cpu_visible && b.cpu_visible && c.cpu_visible, "bench shared tensors");
    MPSMatrix *aMat = make_mps_matrix(a, M, K, rowA, MPSDataTypeFloat16);
    MPSMatrix *bMat = make_mps_matrix(b, K, N, rowB, MPSDataTypeFloat16);
    MPSMatrix *cMat = make_mps_matrix(c, M, N, rowC, MPSDataTypeFloat16);
    MPSMatrixMultiplication *gemm = make_mps_gemm(ctx->device, M, K, N);
    uint64_t rng = UINT64_C(0x123456789abcdef0) ^ ((uint64_t)M << 32U) ^
                   ((uint64_t)K << 16U) ^ (uint64_t)N;
    double gpu_total = 0.0;
    double wall_total = 0.0;
    int gpu_count = 0;
    int iter;
    double flops = 2.0 * (double)M * (double)N * (double)K;

    CHECK(aMat != nil && bMat != nil && cMat != nil && gemm != nil, "bench MPS objects");
    write_matrix_f16_random(a, M, K, rowA, &rng, 0.5f);
    write_matrix_f16_random(b, K, N, rowB, &rng, 0.02f);

    for (iter = -warmup; iter < iters; ++iter) {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        double t0 = now_s();
        CHECK(cmd != nil, "bench command");
        [gemm encodeToCommandBuffer:cmd leftMatrix:aMat rightMatrix:bMat resultMatrix:cMat];
        [cmd commit];
        [cmd waitUntilCompleted];
        CHECK(cmd.status == MTLCommandBufferStatusCompleted, "bench command completed");
        if (iter >= 0) {
            double wall = now_s() - t0;
            double gpu = cmd.GPUEndTime - cmd.GPUStartTime;
            wall_total += wall;
            if (gpu > 0.0) {
                gpu_total += gpu;
                gpu_count += 1;
            }
        }
    }
    {
        double wall_ms = (wall_total / (double)iters) * 1000.0;
        double gpu_ms = gpu_count > 0 ? (gpu_total / (double)gpu_count) * 1000.0 : 0.0;
        double denom_s = gpu_ms > 0.0 ? gpu_ms / 1000.0 : wall_ms / 1000.0;
        double tflops = denom_s > 0.0 ? flops / denom_s / 1.0e12 : 0.0;
        printf("  bench_mps_gemm %-18s M=%4lu K=%4lu N=%5lu gpu_ms=%7.3f wall_ms=%7.3f tflops=%6.2f\n",
               label, (unsigned long)M, (unsigned long)K, (unsigned long)N,
               gpu_ms, wall_ms, tflops);
    }
}

static void probe_transformer_mps_bench(ProbeMetalContext *ctx)
{
    int warmup = env_int("GD_PROBE_MPS_WARMUP", 2);
    int iters = env_int("GD_PROBE_MPS_ITERS", 5);
    bool only256h4 = env_is("GD_PROBE_BENCH_PROFILE", "256h4");
    if (!MPSSupportsMTLDevice(ctx->device)) {
        printf("  transformer MPS bench skipped (MPS unsupported)\n");
        return;
    }
    printf("transformer_mps_matmul_bench profile=%s dtype=f16 warmup=%d iters=%d storage=%s hazard=%s align=%lu prefix=%lu row_pad=%lu batch_pad_rows=%lu\n",
           only256h4 ? "256h4" : "all", warmup, iters, bench_storage_name(),
           bench_hazard_name(), (unsigned long)bench_alloc_alignment(),
           (unsigned long)bench_prefix_pad(), (unsigned long)bench_row_pad_bytes(),
           (unsigned long)bench_batch_pad_rows());

    run_mps_gemm_bench(ctx, "256h4.qkv", 512, 256, 3 * 256, warmup, iters);
    run_mps_gemm_bench(ctx, "256h4.proj", 512, 256, 256, warmup, iters);
    run_mps_gemm_bench(ctx, "256h4.gate_up", 512, 256, 2 * 1024, warmup, iters);
    run_mps_gemm_bench(ctx, "256h4.down", 512, 1024, 256, warmup, iters);

    if (only256h4) {
        printf("batched_attention_mps_matmul_bench profile=256h4 dtype=f16 warmup=%d iters=%d storage=%s hazard=%s align=%lu prefix=%lu row_pad=%lu batch_pad_rows=%lu\n",
               warmup, iters, bench_storage_name(), bench_hazard_name(),
               (unsigned long)bench_alloc_alignment(), (unsigned long)bench_prefix_pad(),
               (unsigned long)bench_row_pad_bytes(), (unsigned long)bench_batch_pad_rows());
        run_mps_batched_gemm_bench(ctx, "256h4.attn_qk", 4, 512, 64, 512, warmup, iters);
        run_mps_batched_gemm_bench(ctx, "256h4.attn_v", 4, 512, 512, 64, warmup, iters);
        return;
    }

    run_mps_gemm_bench(ctx, "50m.qkv", 1024, 512, 3 * 512, warmup, iters);
    run_mps_gemm_bench(ctx, "50m.proj", 1024, 512, 512, warmup, iters);
    run_mps_gemm_bench(ctx, "50m.gate_up", 1024, 512, 2 * 2048, warmup, iters);
    run_mps_gemm_bench(ctx, "50m.down", 1024, 2048, 512, warmup, iters);

    run_mps_gemm_bench(ctx, "100m.qkv", 2048, 768, 3 * 768, warmup, iters);
    run_mps_gemm_bench(ctx, "100m.proj", 2048, 768, 768, warmup, iters);
    run_mps_gemm_bench(ctx, "100m.gate_up", 2048, 768, 2 * 3072, warmup, iters);
    run_mps_gemm_bench(ctx, "100m.down", 2048, 3072, 768, warmup, iters);

    printf("batched_attention_mps_matmul_bench profile=all dtype=f16 warmup=%d iters=%d storage=%s hazard=%s align=%lu prefix=%lu row_pad=%lu batch_pad_rows=%lu\n",
           warmup, iters, bench_storage_name(), bench_hazard_name(),
           (unsigned long)bench_alloc_alignment(), (unsigned long)bench_prefix_pad(),
           (unsigned long)bench_row_pad_bytes(), (unsigned long)bench_batch_pad_rows());
    run_mps_batched_gemm_bench(ctx, "256h4.attn_qk", 4, 512, 64, 512, warmup, iters);
    run_mps_batched_gemm_bench(ctx, "256h4.attn_v", 4, 512, 512, 64, warmup, iters);
    run_mps_batched_gemm_bench(ctx, "50m.attn_qk", 8, 1024, 64, 1024, warmup, iters);
    run_mps_batched_gemm_bench(ctx, "50m.attn_v", 8, 1024, 1024, 64, warmup, iters);
    run_mps_batched_gemm_bench(ctx, "100m.attn_qk", 12, 2048, 64, 2048, warmup, iters);
    run_mps_batched_gemm_bench(ctx, "100m.attn_v", 12, 2048, 2048, 64, warmup, iters);
}

static void probe_ring_waits(ProbeMetalContext *ctx)
{
    uint64_t waits_before = ctx->scratchRing.waits + ctx->dataRing.waits;
    NSUInteger scratch_slots_n = ctx->scratchRing.slots.count;
    NSUInteger data_slots_n = ctx->dataRing.slots.count;
    NSUInteger iterations = (scratch_slots_n > data_slots_n ? scratch_slots_n : data_slots_n) + 1U;
    int *scratch_slots = (int *)calloc(iterations, sizeof(int));
    int *data_slots = (int *)calloc(iterations, sizeof(int));
    NSUInteger i;
    CHECK(scratch_slots != NULL && data_slots != NULL, "ring wait slot arrays");
    for (i = 0; i < iterations; ++i) {
        ProbeTensorRef dummy;
        ProbeTensorRef out;
        id<MTLComputeCommandEncoder> enc;
        ctx_begin(ctx);
        scratch_slots[i] = (int)ctx->scratchRing.current;
        data_slots[i] = (int)ctx->dataRing.current;
        dummy = [ctx->data allocBytes:16 alignment:256 name:"data.dummy"];
        out = [ctx->scratch allocBytes:1024 * sizeof(float) alignment:256 name:"scratch.spin"];
        CHECK(dummy.cpu_visible && out.cpu_visible, "ring slots are shared/cpu visible");
        enc = [ctx->currentCommand computeCommandEncoder];
        encode_spin(enc, ctx->spinPSO, out, 200000U, 1024);
        [enc endEncoding];
        (void)ctx_end(ctx);
    }
    [ctx->scratchRing waitAll];
    [ctx->dataRing waitAll];
    if (scratch_slots_n > 1U) {
        CHECK(scratch_slots[0] != scratch_slots[1],
              "scratch ring uses distinct initial slots");
    }
    if (data_slots_n > 1U) {
        CHECK(data_slots[0] != data_slots[1],
              "data ring uses distinct initial slots");
    }
    CHECK(ctx->scratchRing.waits + ctx->dataRing.waits >= waits_before + 1U,
          "ring exhausted caused wait");
    free(scratch_slots);
    free(data_slots);
    printf("  ring waits ok scratch_slots=%lu data_slots=%lu scratch_waits=%" PRIu64 " data_waits=%" PRIu64 "\n",
           (unsigned long)scratch_slots_n, (unsigned long)data_slots_n,
           ctx->scratchRing.waits, ctx->dataRing.waits);
}

static void state_reset_wait(ProbeMetalContext *ctx, ProbeStateObject *obj)
{
    (void)ctx;
    if (obj->lastCommand != nil && obj->lastCommand.status != MTLCommandBufferStatusCompleted) {
        [obj->lastCommand waitUntilCompleted];
        CHECK(obj->lastCommand.status == MTLCommandBufferStatusCompleted,
              "state reset command completed");
        obj->lastCommand = nil;
    }
}

static void probe_state_reset_waits(ProbeMetalContext *ctx)
{
    ProbeMetalArena *stateArena = [[ProbeMetalArena alloc] initWithDevice:ctx->device
                                                                     name:@"kv_state"
                                                                 capacity:4096
                                                                  options:probe_shared_options()];
    ProbeStateObject kv;
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    memset(&kv, 0, sizeof(kv));
    kv.storage = [stateArena allocBytes:1024 * sizeof(float) alignment:256 name:"kv.cache"];
    cmd = [ctx->queue commandBuffer];
    enc = [cmd computeCommandEncoder];
    encode_spin(enc, ctx->spinPSO, kv.storage, 200000U, 1024);
    [enc endEncoding];
    [cmd commit];
    kv.lastCommand = cmd;
    kv.sequence = ++ctx->nextSequence;
    state_reset_wait(ctx, &kv);
    CHECK(kv.lastCommand == nil, "state reset clears last command");
    printf("  state reset fence wait ok sequence=%" PRIu64 "\n", kv.sequence);
}

int main(void)
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        id<MTLLibrary> library;
        id<MTLFunction> addFn;
        id<MTLFunction> spinFn;
        id<MTLFunction> biasGeluFn;
        id<MTLFunction> rmsNormFn;
        id<MTLFunction> layerNormFn;
        id<MTLComputePipelineState> addPSO;
        id<MTLComputePipelineState> spinPSO;
        id<MTLComputePipelineState> biasGeluPSO;
        id<MTLComputePipelineState> rmsNormPSO;
        id<MTLComputePipelineState> layerNormPSO;
        id<MTLCommandQueue> queue;
        ProbeRingArena *scratchRing;
        ProbeRingArena *dataRing;
        ProbeMetalContext ctx;
        NSUInteger scratchSlots = ring_slots_env("GD_PROBE_SCRATCH_RING_SLOTS", 3);
        NSUInteger dataSlots = ring_slots_env("GD_PROBE_DATA_RING_SLOTS", 3);
        NSUInteger scratchSlotBytes = ring_slot_bytes_env("GD_PROBE_SCRATCH_SLOT_BYTES",
                                                         64 * 1024 * 1024, 4096);
        NSUInteger dataSlotBytes = ring_slot_bytes_env("GD_PROBE_DATA_SLOT_BYTES",
                                                      8 * 1024 * 1024, 1024);

        printf("v2_metal_arena_probe: start\n");
        gProbeKeepAlive = [NSMutableArray array];
        if (device == nil) {
            printf("v2_metal_arena_probe: skipped (no Metal device)\n");
            return 0;
        }
        printf("  device=%s unified=%d\n", device.name.UTF8String,
               device.hasUnifiedMemory ? 1 : 0);

        library = [device newLibraryWithSource:probe_metal_source() options:nil error:&error];
        if (library == nil) {
            fprintf(stderr, "library compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        addFn = [library newFunctionWithName:@"add_one"];
        spinFn = [library newFunctionWithName:@"spin"];
        biasGeluFn = [library newFunctionWithName:@"bias_gelu_f16"];
        rmsNormFn = [library newFunctionWithName:@"rms_norm_f16"];
        layerNormFn = [library newFunctionWithName:@"layer_norm_f16"];
        CHECK(addFn != nil && spinFn != nil && biasGeluFn != nil &&
              rmsNormFn != nil && layerNormFn != nil, "kernel functions found");
        addPSO = [device newComputePipelineStateWithFunction:addFn error:&error];
        CHECK(addPSO != nil, "add_one pipeline created");
        spinPSO = [device newComputePipelineStateWithFunction:spinFn error:&error];
        CHECK(spinPSO != nil, "spin pipeline created");
        biasGeluPSO = [device newComputePipelineStateWithFunction:biasGeluFn error:&error];
        CHECK(biasGeluPSO != nil, "bias_gelu_f16 pipeline created");
        rmsNormPSO = [device newComputePipelineStateWithFunction:rmsNormFn error:&error];
        CHECK(rmsNormPSO != nil, "rms_norm_f16 pipeline created");
        layerNormPSO = [device newComputePipelineStateWithFunction:layerNormFn error:&error];
        CHECK(layerNormPSO != nil, "layer_norm_f16 pipeline created");
        queue = [device newCommandQueue];
        CHECK(queue != nil, "command queue created");
        [gProbeKeepAlive addObject:device];
        [gProbeKeepAlive addObject:library];
        [gProbeKeepAlive addObject:addPSO];
        [gProbeKeepAlive addObject:spinPSO];
        [gProbeKeepAlive addObject:biasGeluPSO];
        [gProbeKeepAlive addObject:rmsNormPSO];
        [gProbeKeepAlive addObject:layerNormPSO];
        [gProbeKeepAlive addObject:queue];

        printf("  ring_config scratch_slots=%lu scratch_slot_bytes=%lu data_slots=%lu data_slot_bytes=%lu\n",
               (unsigned long)scratchSlots, (unsigned long)scratchSlotBytes,
               (unsigned long)dataSlots, (unsigned long)dataSlotBytes);
        scratchRing = [[ProbeRingArena alloc] initWithDevice:device
                                                        name:@"scratch"
                                                       slots:scratchSlots
                                                slotCapacity:scratchSlotBytes
                                                     options:probe_shared_options()];
        dataRing = [[ProbeRingArena alloc] initWithDevice:device
                                                     name:@"data"
                                                    slots:dataSlots
                                             slotCapacity:dataSlotBytes
                                                  options:probe_shared_options()];
        memset(&ctx, 0, sizeof(ctx));
        ctx.device = device;
        ctx.queue = queue;
        ctx.library = library;
        ctx.addOnePSO = addPSO;
        ctx.spinPSO = spinPSO;
        ctx.biasGeluPSO = biasGeluPSO;
        ctx.rmsNormPSO = rmsNormPSO;
        ctx.layerNormPSO = layerNormPSO;
        ctx.scratchRing = scratchRing;
        ctx.dataRing = dataRing;

        probe_offset_shared_readback(&ctx);
        probe_shared_direct_write_read(&ctx);
        probe_mps_matmul(&ctx);
        probe_batched_mps_matmul(&ctx);
        probe_strided_sliced_mps_matmul(&ctx);
        probe_linear_bias_gelu_same_command(&ctx);
        probe_norm_reductions(&ctx);
        probe_transformer_mps_bench(&ctx);
        probe_ring_waits(&ctx);
        probe_state_reset_waits(&ctx);

        {
            NSUInteger i;
            printf("arena_watermarks");
            for (i = 0; i < scratchRing.slots.count; ++i) {
                printf(" scratch%lu=%lu", (unsigned long)i,
                       (unsigned long)scratchRing.slots[i].watermark);
            }
            for (i = 0; i < dataRing.slots.count; ++i) {
                printf(" data%lu=%lu", (unsigned long)i,
                       (unsigned long)dataRing.slots[i].watermark);
            }
            printf("\n");
        }
        printf("v2_metal_arena_probe: ok\n");
    }
    return 0;
}
