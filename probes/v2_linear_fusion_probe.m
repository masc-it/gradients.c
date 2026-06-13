#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "v2_linear_fusion_probe failed: %s (%s:%d)\n",    \
                    (msg), __FILE__, __LINE__);                                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static NSUInteger min_ns(NSUInteger a, NSUInteger b)
{
    return a < b ? a : b;
}

static double now_s(void)
{
    return CFAbsoluteTimeGetCurrent();
}

static uint16_t f32_to_f16_bits(float value)
{
    union {
        float f;
        uint32_t u;
    } v;
    uint32_t sign;
    int32_t exp;
    uint32_t mant;
    uint32_t out_exp;
    uint32_t out_mant;
    v.f = value;
    sign = (v.u >> 16) & 0x8000U;
    exp = (int32_t)((v.u >> 23) & 0xffU) - 127;
    mant = v.u & 0x7fffffU;
    if (((v.u >> 23) & 0xffU) == 0xffU) {
        if (mant == 0U) {
            return (uint16_t)(sign | 0x7c00U);
        }
        return (uint16_t)(sign | 0x7e00U);
    }
    if (exp > 15) {
        return (uint16_t)(sign | 0x7c00U);
    }
    if (exp < -14) {
        uint32_t shifted;
        uint32_t remainder;
        uint32_t halfway;
        int32_t shift = -14 - exp;
        if (shift > 24) {
            return (uint16_t)sign;
        }
        mant |= 0x800000U;
        shifted = mant >> (uint32_t)(shift + 13);
        remainder = mant & ((1U << (uint32_t)(shift + 13)) - 1U);
        halfway = 1U << (uint32_t)(shift + 12);
        if (remainder > halfway || (remainder == halfway && (shifted & 1U) != 0U)) {
            shifted += 1U;
        }
        return (uint16_t)(sign | shifted);
    }
    out_exp = (uint32_t)(exp + 15);
    out_mant = mant >> 13;
    {
        uint32_t remainder = mant & 0x1fffU;
        if (remainder > 0x1000U || (remainder == 0x1000U && (out_mant & 1U) != 0U)) {
            out_mant += 1U;
            if (out_mant == 0x400U) {
                out_mant = 0U;
                out_exp += 1U;
                if (out_exp >= 31U) {
                    return (uint16_t)(sign | 0x7c00U);
                }
            }
        }
    }
    return (uint16_t)(sign | (out_exp << 10) | out_mant);
}

static float f16_bits_to_f32(uint16_t bits)
{
    uint32_t sign = ((uint32_t)bits & 0x8000U) << 16;
    uint32_t exp = ((uint32_t)bits >> 10) & 0x1fU;
    uint32_t mant = (uint32_t)bits & 0x3ffU;
    uint32_t out;
    union {
        uint32_t u;
        float f;
    } v;
    if (exp == 0U) {
        if (mant == 0U) {
            v.u = sign;
            return v.f;
        }
        while ((mant & 0x400U) == 0U) {
            mant <<= 1U;
            exp -= 1U;
        }
        mant &= 0x3ffU;
        exp += 1U;
    } else if (exp == 31U) {
        v.u = sign | 0x7f800000U | (mant << 13);
        return v.f;
    }
    out = sign | ((exp + (127U - 15U)) << 23) | (mant << 13);
    v.u = out;
    return v.f;
}

static const char *kernel_source(void)
{
    return "#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct BiasArgs { ulong y_off; ulong b_off; uint rows; uint cols; uint row_y; };\n"
           "struct LinearArgs { ulong x_off; ulong w_off; ulong b_off; ulong y_off; uint rows; uint inner; uint cols; uint row_x; uint row_w; uint row_y; };\n"
           "kernel void add_bias_f16(device uchar *arena [[buffer(0)]], constant BiasArgs &args [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = args.rows * args.cols;\n"
           "  if (gid >= total) { return; }\n"
           "  uint r = gid / args.cols;\n"
           "  uint c = gid - r * args.cols;\n"
           "  ulong y_byte = args.y_off + ulong(r) * ulong(args.row_y) + ulong(c) * 2ul;\n"
           "  ulong b_byte = args.b_off + ulong(c) * 2ul;\n"
           "  device half *yp = reinterpret_cast<device half *>(arena + y_byte);\n"
           "  device half *bp = reinterpret_cast<device half *>(arena + b_byte);\n"
           "  *yp = half(float(*yp) + float(*bp));\n"
           "}\n"
           "kernel void linear_fused_naive_f16(device uchar *arena [[buffer(0)]], constant LinearArgs &args [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {\n"
           "  uint c = gid.x;\n"
           "  uint r = gid.y;\n"
           "  if (r >= args.rows || c >= args.cols) { return; }\n"
           "  float acc = 0.0f;\n"
           "  for (uint k = 0; k < args.inner; ++k) {\n"
           "    ulong xb = args.x_off + ulong(r) * ulong(args.row_x) + ulong(k) * 2ul;\n"
           "    ulong wb = args.w_off + ulong(k) * ulong(args.row_w) + ulong(c) * 2ul;\n"
           "    device half *xp = reinterpret_cast<device half *>(arena + xb);\n"
           "    device half *wp = reinterpret_cast<device half *>(arena + wb);\n"
           "    acc += float(*xp) * float(*wp);\n"
           "  }\n"
           "  ulong bb = args.b_off + ulong(c) * 2ul;\n"
           "  ulong yb = args.y_off + ulong(r) * ulong(args.row_y) + ulong(c) * 2ul;\n"
           "  device half *bp = reinterpret_cast<device half *>(arena + bb);\n"
           "  device half *yp = reinterpret_cast<device half *>(arena + yb);\n"
           "  *yp = half(acc + float(*bp));\n"
           "}\n";
}

typedef struct BiasArgsHost {
    uint64_t y_off;
    uint64_t b_off;
    uint32_t rows;
    uint32_t cols;
    uint32_t row_y;
} BiasArgsHost;

typedef struct LinearArgsHost {
    uint64_t x_off;
    uint64_t w_off;
    uint64_t b_off;
    uint64_t y_off;
    uint32_t rows;
    uint32_t inner;
    uint32_t cols;
    uint32_t row_x;
    uint32_t row_w;
    uint32_t row_y;
} LinearArgsHost;

typedef struct BenchResult {
    double gpu_ms;
    double wall_ms;
    double tflops;
} BenchResult;

typedef struct LinearCase {
    const char *name;
    NSUInteger M;
    NSUInteger K;
    NSUInteger N;
} LinearCase;

typedef struct LinearBench {
    const LinearCase *shape;
    id<MTLBuffer> arena;
    uint8_t *bytes;
    size_t arena_bytes;
    size_t x_off;
    size_t w_off;
    size_t b_off;
    size_t y_mps_off;
    size_t y_single_off;
    NSUInteger row_x;
    NSUInteger row_w;
    NSUInteger row_y;
    MPSMatrix *x_mat;
    MPSMatrix *w_mat;
    MPSMatrix *y_mps_mat;
    MPSMatrixMultiplication *gemm;
} LinearBench;

static float input_x_value(NSUInteger r, NSUInteger k)
{
    int32_t v = (int32_t)((r * 13U + k * 7U) % 31U) - 15;
    return (float)v * 0.0078125f;
}

static float input_w_value(NSUInteger k, NSUInteger c)
{
    int32_t v = (int32_t)((k * 5U + c * 11U) % 29U) - 14;
    return (float)v * 0.0078125f;
}

static float input_b_value(NSUInteger c)
{
    int32_t v = (int32_t)((c * 3U) % 17U) - 8;
    return (float)v * 0.015625f;
}

static void write_matrix_formula(uint8_t *base,
                                 size_t offset,
                                 NSUInteger rows,
                                 NSUInteger cols,
                                 NSUInteger row_bytes,
                                 bool is_x)
{
    NSUInteger r;
    NSUInteger c;
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            float value = is_x ? input_x_value(r, c) : input_w_value(r, c);
            uint16_t bits = f32_to_f16_bits(value);
            memcpy(base + offset + r * row_bytes + c * sizeof(uint16_t), &bits, sizeof(bits));
        }
    }
}

static void write_bias_formula(uint8_t *base, size_t offset, NSUInteger n)
{
    NSUInteger c;
    for (c = 0; c < n; ++c) {
        uint16_t bits = f32_to_f16_bits(input_b_value(c));
        memcpy(base + offset + c * sizeof(uint16_t), &bits, sizeof(bits));
    }
}

static float read_f16_at(uint8_t *base, size_t offset, NSUInteger r, NSUInteger c, NSUInteger row_bytes)
{
    uint16_t bits;
    memcpy(&bits, base + offset + r * row_bytes + c * sizeof(uint16_t), sizeof(bits));
    return f16_bits_to_f32(bits);
}

static float expected_value(NSUInteger r, NSUInteger c, NSUInteger K)
{
    NSUInteger k;
    float sum = 0.0f;
    for (k = 0; k < K; ++k) {
        float x = f16_bits_to_f32(f32_to_f16_bits(input_x_value(r, k)));
        float w = f16_bits_to_f32(f32_to_f16_bits(input_w_value(k, c)));
        sum += x * w;
    }
    sum += f16_bits_to_f32(f32_to_f16_bits(input_b_value(c)));
    return f16_bits_to_f32(f32_to_f16_bits(sum));
}

static double linear_flops(NSUInteger M, NSUInteger K, NSUInteger N)
{
    return 2.0 * (double)M * (double)K * (double)N + (double)M * (double)N;
}

static void bench_prepare(id<MTLDevice> device, const LinearCase *shape, LinearBench *bench)
{
    MPSMatrixDescriptor *x_desc;
    MPSMatrixDescriptor *w_desc;
    MPSMatrixDescriptor *y_desc;
    size_t off;
    memset((void *)bench, 0, sizeof(*bench));
    bench->shape = shape;
    bench->row_x = [MPSMatrixDescriptor rowBytesForColumns:shape->K dataType:MPSDataTypeFloat16];
    bench->row_w = [MPSMatrixDescriptor rowBytesForColumns:shape->N dataType:MPSDataTypeFloat16];
    bench->row_y = [MPSMatrixDescriptor rowBytesForColumns:shape->N dataType:MPSDataTypeFloat16];
    CHECK(bench->row_x >= shape->K * sizeof(uint16_t) &&
          bench->row_w >= shape->N * sizeof(uint16_t) &&
          bench->row_y >= shape->N * sizeof(uint16_t), "row bytes valid");

    off = 17U;
    bench->x_off = align_up(off, 256U);
    off = bench->x_off + shape->M * bench->row_x;
    bench->w_off = align_up(off + 19U, 256U);
    off = bench->w_off + shape->K * bench->row_w;
    bench->b_off = align_up(off + 23U, 256U);
    off = bench->b_off + shape->N * sizeof(uint16_t);
    bench->y_mps_off = align_up(off + 29U, 256U);
    off = bench->y_mps_off + shape->M * bench->row_y;
    bench->y_single_off = align_up(off + 31U, 256U);
    off = bench->y_single_off + shape->M * bench->row_y;
    bench->arena_bytes = align_up(off, 4096U);
    bench->arena = [device newBufferWithLength:bench->arena_bytes options:MTLResourceStorageModeShared];
    CHECK(bench->arena != nil, "shared arena buffer");
    bench->bytes = (uint8_t *)[bench->arena contents];
    CHECK(bench->bytes != NULL, "shared arena contents");
    memset(bench->bytes, 0, bench->arena_bytes);
    write_matrix_formula(bench->bytes, bench->x_off, shape->M, shape->K, bench->row_x, true);
    write_matrix_formula(bench->bytes, bench->w_off, shape->K, shape->N, bench->row_w, false);
    write_bias_formula(bench->bytes, bench->b_off, shape->N);

    x_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:shape->M
                                                   columns:shape->K
                                                  rowBytes:bench->row_x
                                                  dataType:MPSDataTypeFloat16];
    w_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:shape->K
                                                   columns:shape->N
                                                  rowBytes:bench->row_w
                                                  dataType:MPSDataTypeFloat16];
    y_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:shape->M
                                                   columns:shape->N
                                                  rowBytes:bench->row_y
                                                  dataType:MPSDataTypeFloat16];
    bench->x_mat = [[MPSMatrix alloc] initWithBuffer:bench->arena offset:bench->x_off descriptor:x_desc];
    bench->w_mat = [[MPSMatrix alloc] initWithBuffer:bench->arena offset:bench->w_off descriptor:w_desc];
    bench->y_mps_mat = [[MPSMatrix alloc] initWithBuffer:bench->arena offset:bench->y_mps_off descriptor:y_desc];
    CHECK(bench->x_mat != nil && bench->w_mat != nil && bench->y_mps_mat != nil,
          "MPS matrix descriptors");
    bench->gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                    transposeLeft:NO
                                                   transposeRight:NO
                                                       resultRows:shape->M
                                                    resultColumns:shape->N
                                                  interiorColumns:shape->K
                                                            alpha:1.0
                                                             beta:0.0];
    CHECK(bench->gemm != nil, "MPS GEMM");
}

static void encode_mps_plus_bias(id<MTLCommandBuffer> cmd,
                                 id<MTLComputePipelineState> bias_pso,
                                 const LinearBench *bench)
{
    BiasArgsHost args;
    id<MTLComputeCommandEncoder> enc;
    NSUInteger total = bench->shape->M * bench->shape->N;
    [bench->gemm encodeToCommandBuffer:cmd
                            leftMatrix:bench->x_mat
                           rightMatrix:bench->w_mat
                          resultMatrix:bench->y_mps_mat];
    args.y_off = bench->y_mps_off;
    args.b_off = bench->b_off;
    args.rows = (uint32_t)bench->shape->M;
    args.cols = (uint32_t)bench->shape->N;
    args.row_y = (uint32_t)bench->row_y;
    enc = [cmd computeCommandEncoder];
    CHECK(enc != nil, "bias encoder");
    [enc setComputePipelineState:bias_pso];
    [enc setBuffer:bench->arena offset:0U atIndex:0U];
    [enc setBytes:&args length:sizeof(args) atIndex:1U];
    [enc dispatchThreads:MTLSizeMake(total, 1U, 1U)
   threadsPerThreadgroup:MTLSizeMake(min_ns(total, 256U), 1U, 1U)];
    [enc endEncoding];
}

static void encode_single_kernel(id<MTLCommandBuffer> cmd,
                                 id<MTLComputePipelineState> linear_pso,
                                 const LinearBench *bench)
{
    LinearArgsHost args;
    id<MTLComputeCommandEncoder> enc;
    args.x_off = bench->x_off;
    args.w_off = bench->w_off;
    args.b_off = bench->b_off;
    args.y_off = bench->y_single_off;
    args.rows = (uint32_t)bench->shape->M;
    args.inner = (uint32_t)bench->shape->K;
    args.cols = (uint32_t)bench->shape->N;
    args.row_x = (uint32_t)bench->row_x;
    args.row_w = (uint32_t)bench->row_w;
    args.row_y = (uint32_t)bench->row_y;
    enc = [cmd computeCommandEncoder];
    CHECK(enc != nil, "single kernel encoder");
    [enc setComputePipelineState:linear_pso];
    [enc setBuffer:bench->arena offset:0U atIndex:0U];
    [enc setBytes:&args length:sizeof(args) atIndex:1U];
    [enc dispatchThreads:MTLSizeMake(bench->shape->N, bench->shape->M, 1U)
   threadsPerThreadgroup:MTLSizeMake(16U, 16U, 1U)];
    [enc endEncoding];
}

static BenchResult run_bench(id<MTLCommandQueue> queue,
                             const LinearBench *bench,
                             void (^encode)(id<MTLCommandBuffer>),
                             int warmup,
                             int iters)
{
    int i;
    double wall_total = 0.0;
    double gpu_total = 0.0;
    int gpu_count = 0;
    BenchResult result;
    for (i = 0; i < warmup + iters; ++i) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        double start = now_s();
        CHECK(cmd != nil, "bench command buffer");
        encode(cmd);
        [cmd commit];
        [cmd waitUntilCompleted];
        wall_total += i >= warmup ? now_s() - start : 0.0;
        CHECK(cmd.status == MTLCommandBufferStatusCompleted, "bench command complete");
        if (i >= warmup && cmd.GPUEndTime > cmd.GPUStartTime) {
            gpu_total += cmd.GPUEndTime - cmd.GPUStartTime;
            gpu_count += 1;
        }
    }
    result.wall_ms = (wall_total / (double)iters) * 1000.0;
    result.gpu_ms = gpu_count > 0 ? (gpu_total / (double)gpu_count) * 1000.0 : 0.0;
    {
        double denom = result.gpu_ms > 0.0 ? result.gpu_ms / 1000.0 : result.wall_ms / 1000.0;
        result.tflops = linear_flops(bench->shape->M, bench->shape->K, bench->shape->N) / denom / 1.0e12;
    }
    return result;
}

static void validate_outputs(const LinearBench *bench)
{
    const NSUInteger samples[][2] = {
        {0U, 0U},
        {0U, 1U},
        {1U, 0U},
        {bench->shape->M / 2U, bench->shape->N / 2U},
        {bench->shape->M - 1U, bench->shape->N - 1U},
    };
    NSUInteger i;
    float max_err_mps = 0.0f;
    float max_err_single = 0.0f;
    for (i = 0; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        NSUInteger r = samples[i][0];
        NSUInteger c = samples[i][1];
        float want = expected_value(r, c, bench->shape->K);
        float got_mps = read_f16_at(bench->bytes, bench->y_mps_off, r, c, bench->row_y);
        float got_single = read_f16_at(bench->bytes, bench->y_single_off, r, c, bench->row_y);
        float err_mps = fabsf(got_mps - want);
        float err_single = fabsf(got_single - want);
        if (err_mps > max_err_mps) {
            max_err_mps = err_mps;
        }
        if (err_single > max_err_single) {
            max_err_single = err_single;
        }
        CHECK(err_mps <= 0.015f, "MPS+bias sample close");
        CHECK(err_single <= 0.015f, "single-kernel sample close");
    }
    printf("  validate %-14s samples_ok max_err_mps=%.6g max_err_single=%.6g\n",
           bench->shape->name, max_err_mps, max_err_single);
}

int main(void)
{
    @autoreleasepool {
        LinearCase cases[] = {
            {"tiny", 4U, 7U, 6U},
            {"small", 64U, 128U, 128U},
            {"qkv_256h4", 512U, 256U, 768U},
        };
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue;
        NSError *error = nil;
        id<MTLLibrary> library;
        id<MTLFunction> bias_fn;
        id<MTLFunction> linear_fn;
        id<MTLComputePipelineState> bias_pso;
        id<MTLComputePipelineState> linear_pso;
        int warmup = 1;
        int iters = 5;
        NSUInteger ci;

        CHECK(device != nil, "Metal device");
        CHECK(MPSSupportsMTLDevice(device), "MPS support");
        queue = [device newCommandQueue];
        CHECK(queue != nil, "command queue");
        library = [device newLibraryWithSource:[NSString stringWithUTF8String:kernel_source()]
                                       options:nil
                                         error:&error];
        CHECK(library != nil, "compile kernels");
        bias_fn = [library newFunctionWithName:@"add_bias_f16"];
        linear_fn = [library newFunctionWithName:@"linear_fused_naive_f16"];
        CHECK(bias_fn != nil && linear_fn != nil, "kernel functions");
        bias_pso = [device newComputePipelineStateWithFunction:bias_fn error:&error];
        linear_pso = [device newComputePipelineStateWithFunction:linear_fn error:&error];
        CHECK(bias_pso != nil && linear_pso != nil, "pipelines");

        printf("v2_linear_fusion_probe: compare xw+b paths warmup=%d iters=%d\n", warmup, iters);
        for (ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
            LinearBench bench;
            BenchResult mps_result;
            BenchResult single_result;
            double speedup;
            bench_prepare(device, &cases[ci], &bench);

            mps_result = run_bench(queue, &bench, ^(id<MTLCommandBuffer> cmd) {
                encode_mps_plus_bias(cmd, bias_pso, &bench);
            }, warmup, iters);
            single_result = run_bench(queue, &bench, ^(id<MTLCommandBuffer> cmd) {
                encode_single_kernel(cmd, linear_pso, &bench);
            }, warmup, iters);
            validate_outputs(&bench);
            speedup = single_result.gpu_ms > 0.0 ? single_result.gpu_ms / mps_result.gpu_ms : 0.0;
            printf("  bench %-14s M=%4lu K=%4lu N=%5lu flops=%9.3g\n",
                   bench.shape->name,
                   (unsigned long)bench.shape->M,
                   (unsigned long)bench.shape->K,
                   (unsigned long)bench.shape->N,
                   linear_flops(bench.shape->M, bench.shape->K, bench.shape->N));
            printf("    mps+epilogue    gpu_ms=%8.4f wall_ms=%8.4f tflops=%7.3f\n",
                   mps_result.gpu_ms, mps_result.wall_ms, mps_result.tflops);
            printf("    single_kernel   gpu_ms=%8.4f wall_ms=%8.4f tflops=%7.3f speedup_mps_vs_single=%6.2fx\n",
                   single_result.gpu_ms, single_result.wall_ms, single_result.tflops, speedup);
        }
        printf("v2_linear_fusion_probe: ok\n");
    }
    return 0;
}
