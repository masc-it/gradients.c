#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static uint16_t f32_to_bf16(float f){union{float f;uint32_t u;}x={f};uint32_t u=x.u;uint32_t r=(u>>16)&1;u+=0x7fff+r;return (uint16_t)(u>>16);}

typedef struct { int M, N, K; } dims3;

static double run_kernel(id<MTLDevice> dev, id<MTLCommandQueue> q,
                         id<MTLComputePipelineState> pso,
                         id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                         dims3 d, MTLSize tpg, int iters)
{
    MTLSize grid = MTLSizeMake((NSUInteger)(d.N / 64), (NSUInteger)(d.M / 64), 1);
    /* warmup */
    for (int w = 0; w < 3; ++w) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:A offset:0 atIndex:0]; [enc setBuffer:B offset:0 atIndex:1];
        [enc setBuffer:C offset:0 atIndex:2]; [enc setBytes:&d length:sizeof(d) atIndex:3];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }
    /* timed: all iters in one command buffer */
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:A offset:0 atIndex:0]; [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:C offset:0 atIndex:2]; [enc setBytes:&d length:sizeof(d) atIndex:3];
    for (int i = 0; i < iters; ++i)
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];
    NSDate *t0 = [NSDate date];
    [cb commit]; [cb waitUntilCompleted];
    double secs = -[t0 timeIntervalSinceNow];
    double flop = 2.0 * d.M * d.N * d.K * iters;
    return flop / secs / 1e9; /* GFLOP/s */
}

int main(int argc, char **argv) {
    int S = (argc > 1) ? atoi(argv[1]) : 2048;
    int iters = (argc > 2) ? atoi(argv[2]) : 30;
    dims3 d = { S, S, S };
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"bench/metal/gemm_bf16.metal" encoding:NSUTF8StringEncoding error:&err];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }

        size_t n = (size_t)S * S;
        id<MTLBuffer> Af = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Bf = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Cf = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Ab = [dev newBufferWithLength:n * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Bb = [dev newBufferWithLength:n * 2 options:MTLResourceStorageModeShared];
        float *af = Af.contents, *bf = Bf.contents;
        uint16_t *ab = Ab.contents, *bb = Bb.contents;
        for (size_t i = 0; i < n; ++i) {
            float va = (float)((i % 13) - 6) * 0.1f, vb = (float)((i % 7) - 3) * 0.1f;
            af[i] = va; bf[i] = vb; ab[i] = f32_to_bf16(va); bb[i] = f32_to_bf16(vb);
        }

        struct { const char *name; const char *fn; id<MTLBuffer> A, B; MTLSize tpg; } cases[] = {
            { "fp32 reg-blocked (shipping)", "gemm_reg_f32",  Af, Bf, MTLSizeMake(16, 16, 1) },
            { "fp32 simdgroup_matrix",       "gemm_simd_f32", Af, Bf, MTLSizeMake(128, 1, 1) },
            { "bf16 simdgroup_matrix",       "gemm_simd_bf16",Ab, Bb, MTLSizeMake(128, 1, 1) },
        };
        printf("GEMM %dx%dx%d, %d iters/batch\n", S, S, S, iters);
        for (int c = 0; c < 3; ++c) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:cases[c].fn]];
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) { fprintf(stderr, "%s pso: %s\n", cases[c].fn, err.localizedDescription.UTF8String); return 1; }
            double g = run_kernel(dev, q, pso, cases[c].A, cases[c].B, Cf, d, cases[c].tpg, iters);
            printf("  %-30s %8.1f GFLOP/s\n", cases[c].name, g);
        }
    }
    return 0;
}
