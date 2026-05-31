# MPS Foundations Tutorial

*A practical walkthrough of writing GPU compute kernels with Metal — drawn from the gradients.c Metal backend.*

> **Who this is for:** C/Python devs who know basic GPU concepts but haven't written Metal. You want to understand how to write kernels, compile them, dispatch them from Objective‑C, and think about correctness on Apple Silicon.

> **What this is not:** A guide to MPS (`MPSMatrixMultiplication`, `MPSNNImageNode`). This is the lower-level Metal compute path — shaders you write yourself in `.metal` files.

---

## Table of Contents

1. [The Metal Compute Stack](#1-the-metal-compute-stack)
2. [Your First Kernel](#2-your-first-kernel)
3. [Address Spaces & Buffer Attributes](#3-address-spaces--buffer-attributes)
4. [Thread Dispatch: Grids, Groups, and Threads](#4-thread-dispatch-grids-groups-and-threads)
5. [Host-Side Setup: Device → Queue → Library → Pipeline](#5-host-side-setup-device--queue--library--pipeline)
6. [Encoding & Dispatching](#6-encoding--dispatching)
7. [Unified Memory on Apple Silicon](#7-unified-memory-on-apple-silicon)
8. [Threadgroup Memory & Cooperative Reductions](#8-threadgroup-memory--cooperative-reductions)
9. [Broadcasting & Striding Patterns](#9-broadcasting--striding-patterns)
10. [Tiled GEMM (Shared Memory for Matmul)](#10-tiled-gemm-shared-memory-for-matmul)
11. [Softmax & Stable Numerics](#11-softmax--stable-numerics)
12. [Build Pipeline: .metal → .metallib](#12-build-pipeline-metal--metallib)
13. [Debugging & Validation](#13-debugging--validation)
14. [Reference: Kernel Catalog](#14-reference-kernel-catalog)

---

## 1. The Metal Compute Stack

Metal on Apple Silicon has three layers:

```
  .metal          – Shader source (MSL = Metal Shading Language, a C++14 dialect)
    ↓ xcrun metal
  .air            – LLVM bitcode
    ↓ xcrun metallib
  .metallib       – Device-ready binary
    ↓ newLibraryWithURL:error:
  MTLLibrary      – Runtime library object (loaded from .metallib)
    ↓ newFunctionWithName:
  MTLFunction     – A single kernel entry point
    ↓ newComputePipelineStateWithFunction:error:
  MTLComputePipelineState – Compiled + optimized dispatch object
    ↓ setComputePipelineState:
  Encode → dispatch → GPU
```

On the host side (Objective‑C, not Swift in this project):

```
  MTLDevice        → The GPU (one per system)
  MTLCommandQueue  → Serial submission queue
  MTLCommandBuffer → One batch of work
  MTLComputeCommandEncoder → Writes kernel state + dispatches
```

**Key constraint:** A `MTLComputeCommandEncoder` encodes commands in order. One encoder, multiple dispatches — each runs to completion before the next starts, no manual barriers needed between them.

---

## 2. Your First Kernel

Start with the simplest possible compute kernel — elementwise addition:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void gd_add(device const float *a          [[buffer(0)]],
                   device const float *b          [[buffer(1)]],
                   device float *out              [[buffer(2)]],
                   uint gid                        [[thread_position_in_grid]])
{
    out[gid] = a[gid] + b[gid];
}
```

### What's going on

| Element | Meaning |
|---|---|
| `kernel` | Marks this as a compute kernel (entry point from host) |
| `device const float *a` | A buffer in device-visible memory, read-only |
| `[[buffer(0)]]` | Binds this pointer to buffer index 0 (matches host-side `setBuffer:offset:atIndex:`) |
| `uint gid` | The global thread index |
| `[[thread_position_in_grid]]` | Attribute that fills `gid` with the linear position of this thread in the 1D grid |

This is the **one-thread-per-element** pattern. If your grid is `{numel, 1, 1}`, thread `gid` processes element `gid`.

### Bounds checking

Grid sizes don't have to divide evenly by threadgroup size. Still, always guard:

```metal
if ((int)gid >= p.numel) return;
```

Metal's `dispatchThreads:` automatically handles partial threadgroups — but your grid might overshoot by a few threads. This guard is cheap and prevents out-of-bounds writes.

---

## 3. Address Spaces & Buffer Attributes

MSL has explicit address spaces. You'll use three:

| Qualifier | Meaning | Lifetime |
|---|---|---|
| `device` | Global GPU memory | Allocated from host, persists |
| `constant` | Small read-only data, cached per threadgroup | Set from host via `setBytes:` |
| `threadgroup` | Shared memory within one threadgroup | Per-threadgroup, wiped between groups |

Example from the codebase — a mixed-signature kernel:

```metal
kernel void gd_add(device const float *a          [[buffer(0)]],
                   device const float *b          [[buffer(1)]],
                   device float *out              [[buffer(2)]],
                   constant gd_metal_ew_params &p  [[buffer(3)]],
                   uint gid                        [[thread_position_in_grid]])
```

- `a`, `b`, `out` are `device` — big arrays.
- `p` is `constant` — a small struct copied by the host with `setBytes:`. The `constant` space is fast and uniform across threads in the threadgroup.

### The `params` pattern

Every kernel gets a `params` struct carrying metadata (shapes, flags) so the same kernel handles many shapes. The struct is defined in a shared C/MSL header:

```c
// metal_kernel_types.h – included by both .m and .metal
typedef struct {
    int numel;
    float scale;
} gd_metal_unary_params;
```

On the host:

```objc
gd_metal_unary_params params = { .numel = (int)numel, .scale = 0.5f };
[encoder setBytes:&params length:sizeof(params) atIndex:2];
```

In the shader:

```metal
kernel void gd_scale(device const float *x   [[buffer(0)]],
                     device float *out       [[buffer(1)]],
                     constant gd_metal_unary_params &p [[buffer(2)]],
                     uint gid                [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) return;
    out[gid] = x[gid] * p.scale;
}
```

### Thread attributes summary

| Attribute | Thread Index | Shape |
|---|---|---|
| `thread_position_in_grid` | Global linear position | Up to 3D |
| `threadgroup_position_in_grid` | Which threadgroup this is | `uint3` |
| `thread_position_in_threadgroup` | Position within the threadgroup | `uint3` (local) |
| `thread_index_in_threadgroup` | Linear position within threadgroup | `uint` (1D local) |
| `threads_per_threadgroup` | Total threads in this threadgroup | `uint` (set at dispatch) |

---

## 4. Thread Dispatch: Grids, Groups, and Threads

### 1D dispatch (elementwise ops)

```objc
// Host side
MTLSize grid = MTLSizeMake(numel, 1, 1);
MTLSize tg  = MTLSizeMake(256, 1, 1); // threadgroup size
[encoder dispatchThreads:grid threadsPerThreadgroup:tg];
```

Metal rounds the grid up to the nearest multiple of the threadgroup size. Your kernel guards against overshoot with the `gid >= numel` check.

### 2D dispatch (matmul, linear)

```objc
// out[N, M] — grid is (N, M)
MTLSize grid = MTLSizeMake(n, m, 1);
MTLSize tg  = MTLSizeMake(16, 16, 1);
[encoder dispatchThreads:grid threadsPerThreadgroup:tg];
```

In the shader, `gid` is `uint2`:

```metal
kernel void gd_matmul(device const float *a [[buffer(0)]],
                      device const float *b [[buffer(1)]],
                      device float *out     [[buffer(2)]],
                      ..., uint2 gid        [[thread_position_in_grid]])
{
    int col = gid.x;
    int row = gid.y;
    // ...
}
```

### 3D dispatch (batched matmul via tiled kernel)

```metal
// grid.z = batch index
MTLSize grid = MTLSizeMake(tiles_n, tiles_m, n_batches);
MTLSize tg  = MTLSizeMake(16, 16, 1);
```

The shader unpacks:

```metal
kernel void gd_matmul_tiled(..., uint3 tgpos [[threadgroup_position_in_grid]],
                            uint3 lid   [[thread_position_in_threadgroup]])
```

### Choosing a threadgroup size

For Apple GPUs: **256 threads per threadgroup** is a solid default for 1D kernels. For 2D/3D: multiples of 16 fit the SIMD group size (32). Use `pipelineState.maxTotalThreadsPerThreadgroup` to clamp.

```
Suggested 1D:  256
Suggested 2D:  (16, 16) = 256
Suggested 2D:  (32, 8)  = 256  (preferred for matmul — more threads on the reduction dim)
```

---

## 5. Host-Side Setup: Device → Queue → Library → Pipeline

This is all done once at backend initialization.

### Step 1: Get the GPU

```objc
#import <Metal/Metal.h>

id<MTLDevice> device = MTLCreateSystemDefaultDevice();
// Returns nil if no Metal-capable GPU (rare on Apple Silicon)
```

### Step 2: Create a command queue

```objc
id<MTLCommandQueue> queue = [device newCommandQueue];
// Long-lived, reuse for the lifetime of your backend
```

### Step 3: Load the metallib

```objc
// From a .metallib file bundled with your app
id<MTLLibrary> library = [device newLibraryWithURL:metallibURL error:&error];
// Or from source (for prototyping):
// id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
```

The gradients.c project loads from a prebuilt `gradients.metallib`:

```objc
NSString *path = [NSString stringWithUTF8String:metallib_path];
NSURL *url = [NSURL fileURLWithPath:path];
library = [device newLibraryWithURL:url error:&error];
```

### Step 4: Build pipeline states

```objc
id<MTLFunction> fn = [library newFunctionWithName:@"gd_add"];
id<MTLComputePipelineState> pipeline;
pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
```

Cache these in a dictionary keyed by op kind or function name. Building a pipeline state compiles the function for the specific GPU — it's expensive, so do it once and reuse.

### Step 5: Store as ARC-backed state

```objc
@interface GDMetalState : NSObject
@property (strong) id<MTLDevice> device;
@property (strong) id<MTLCommandQueue> queue;
@property (strong) id<MTLLibrary> library;
@property (strong) NSMutableDictionary<NSNumber *, id<MTLComputePipelineState>> *pipelines;
@property (strong) id<MTLCommandBuffer> inFlight;
@end
```

---

## 6. Encoding & Dispatching

Every `execute` call creates a command buffer and encoder, dispatches all kernels, and commits.

### Single-run pattern

```objc
// 1. One command buffer
id<MTLCommandBuffer> cmd = [queue commandBuffer];

// 2. One encoder for the whole graph
id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

// 3. Encode each node in order
for (int i = 0; i < n_nodes; i++) {
    [enc setComputePipelineState:pipeline];

    // Bind buffers
    [enc setBuffer:aBuffer offset:0 atIndex:0];
    [enc setBuffer:bBuffer offset:0 atIndex:1];
    [enc setBuffer:outBuffer offset:0 atIndex:2];

    // Bind constant params
    [enc setBytes:&params length:sizeof(params) atIndex:3];

    // Dispatch
    MTLSize grid = MTLSizeMake(numel, 1, 1);
    MTLSize tg   = MTLSizeMake(256, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

// 4. Finalize
[enc endEncoding];

// 5. Commit (non-blocking)
[cmd commit];

// 6. Wait (blocking — done later by synchronize)
[cmd waitUntilCompleted];
```

### Coherency contract

On Apple Silicon with `MTLResourceStorageModeShared`:

- **CPU writes → GPU sees them** only if writes complete *before* `[cmd commit]`.
- **GPU writes → CPU sees them** only after `[cmd waitUntilCompleted]`.

This is strict. It doesn't matter that memory is physically unified — the GPU caches in its own tile memory. Always stage input data before `commit`, and always wait before reading results.

```
❌ Wrong:
    memcpy(buffer.contents, data, nbytes);
    [cmd commit];
    // GPU may not see the data
    [cmd waitUntilCompleted];
    memcpy(result, buffer.contents, nbytes);

✅ Correct:
    memcpy(buffer.contents, data, nbytes);   // CPU->GPU before commit
    [cmd commit];
    [cmd waitUntilCompleted];                  // barrier
    memcpy(result, buffer.contents, nbytes);  // GPU->CPU after completion
```

### Staging leaves (graph execution pattern)

In gradients.c, the compile step allocates separate Metal buffers for every graph value. `execute` must copy CPU tensor data into those buffers *before* commit — this is called **leaf staging**:

```objc
for each external leaf value:
    memcpy(metalBuffer.contents, cpuTensorData, nbytes);

[cmd commit];
```

After `waitUntilCompleted`, mutated values (gradients, optimizer state) must be copied back to the CPU tensor storage — **writeback**.

---

## 7. Unified Memory on Apple Silicon

Apple Silicon uses a unified memory architecture (UMA): the CPU and GPU share the same physical memory. This eliminates PCIe transfer latency and makes `storageModeShared` buffer allocations truly zero-copy in hardware.

### What unified memory means in practice

```
┌─────────────────────────────────────┐
│              Physical RAM            │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ CPU core  │  │  GPU cores       │ │
│  │ (L1/L2)   │  │  (tile mem)      │ │
│  └──────────┘  └──────────────────┘ │
│         ↕              ↕            │
│    ┌──────────────────────────┐     │
│    │   Shared MTLBuffer       │     │
│    │   (one physical copy)    │     │
│    └──────────────────────────┘     │
└─────────────────────────────────────┘
```

**You still need command-buffer synchronization.** The GPU caches in its tile memory (per-core SRAM), not in the L1/L2 cache hierarchy the CPU uses. Without explicit synchronization:

- GPU might read stale tile-cache data even if CPU wrote to the shared buffer.
- CPU might read stale L1/L2 cache even if GPU wrote to the shared buffer.

### Allocation

```objc
id<MTLBuffer> buffer = [device newBufferWithLength:nbytes
                                           options:MTLResourceStorageModeShared];
```

For v1 kernels, use `storageModeShared` exclusively. It's the simplest mental model:

1. CPU writes to `buffer.contents`.
2. Commit command buffer.
3. GPU computes, writes results back to same buffer.
4. Wait for completion.
5. CPU reads from `buffer.contents`.

No staging buffers. No blit encoders. Just `memcpy` to/from `buffer.contents`.

### Zero-fill

```objc
memset(buffer.contents, 0, nbytes);
```

You must zero-fill manually after allocation if you need zeros (Metal doesn't guarantee zeroed memory for `newBufferWithLength:`).

---

## 8. Threadgroup Memory & Cooperative Reductions

This is where GPU programming gets interesting. When threads in a threadgroup need to share partial results, they use `threadgroup` memory — a fast SRAM scratchpad (typically 16–64 KB per threadgroup on Apple GPUs).

### Pattern: cooperative sum over a row

RMSNorm requires computing `sum(x^2)` over the last dimension for each row. Instead of having one thread loop over all elements (slow for large dimensions), the entire threadgroup cooperates:

```metal
kernel void gd_rms_norm(device const float *x               [[buffer(0)]],
                        device const float *weight          [[buffer(1)]],
                        device float *out                   [[buffer(2)]],
                        constant gd_metal_rmsnorm_params &p  [[buffer(3)]],
                        uint tgid  [[threadgroup_position_in_grid]],
                        uint tid    [[thread_index_in_threadgroup]],
                        uint tgsz   [[threads_per_threadgroup]])
{
    int r = (int)tgid;                    // one threadgroup per row
    threadgroup float part[GD_RMS_TG];    // shared partial sums

    // Phase 1: each thread sums a strided subset of elements
    float local = 0.0f;
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        float v = x[base + c];
        local += v * v;
    }
    part[tid] = local;

    // Phase 2: tree reduction in threadgroup memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            part[tid] += part[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // part[0] now holds the sum
    float rms = sqrt(part[0] / (float)p.last + p.eps);
    float inv = 1.0f / rms;

    // Phase 3: normalize (one element per thread)
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        out[base + c] = x[base + c] * inv * weight[c];
    }
}
```

### Threadgroup memory rules

1. **Declare as a local array** in the kernel: `threadgroup float part[256];`
2. **Barrier before reading** another thread's write: `threadgroup_barrier(mem_flags::mem_threadgroup);`
3. **Barrier between reduction passes** — every iteration needs one.
4. **Size is per-threadgroup** and limited. Apple GPUs typically have 16–64 KB. `GD_RMS_TG = 256` floats = 1024 bytes, well within range.

### Tree reduction visual

```
Threads:  0  1  2  3  4  5  6  7
         [a0][a1][a2][a3][a4][a5][a6][a7]   ← each thread writes its partial result

stride=4: [a0+a4][a1+a5][a2+a6][a3+a7]
           ↓
stride=2: [a0+a4+a2+a6][a1+a5+a3+a7]
           ↓
stride=1: [a0+a4+a2+a6+a1+a5+a3+a7]
           ↓
part[0] = total sum
```

Each step halves the active threads. After `log2(tgsz)` steps, thread 0 has the total. All threads sync after each step via `threadgroup_barrier`.

### Two-value reduction (RMSNorm backward)

The RMSNorm backward needs two sums — `sum(x^2)` and `sum(go * weight * x)` — over the same row. Rather than doing two separate reductions (doubling the bandwidth), compute both in one pass:

```metal
float lss = 0.0f;   // sum(x^2)
float la = 0.0f;    // sum(go * weight * x)
for (int c = tid; c < p.last; c += tgsz) {
    float v = x[base + c];
    lss += v * v;
    la += go[base + c] * weight[c] * v;
}
pss[tid] = lss;
pa[tid] = la;
// Reduce both arrays simultaneously
for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        pss[tid] += pss[tid + stride];
        pa[tid] += pa[tid + stride];
    }
    barrier;
}
// pss[0] = total x², pa[0] = total A
```

This halves global memory reads compared to doing two separate kernel dispatches.

---

## 9. Broadcasting & Striding Patterns

NumPy-style broadcasting is a hallmark of the gradients.c ops. The shader must reproduce the same `broadcast_offset` logic the CPU kernel uses.

### The problem

```
A shape: [3, 1]    (size-1 dims are broadcast)
B shape:    [4]
Out shape: [3, 4]
```

Thread with output index `[1, 2]` needs to read `A[1, 0]` (because the size-1 dim maps to coord 0) and `B[2]` (right-aligned).

### The solution

Walk the output index right-aligned against the input's own dims. For each input dimension, if the input size is 1, force coord to 0 (broadcast).

```metal
static int gd_broadcast_offset(thread const int *out_index,
                               int out_ndim,
                               constant int *in_sizes,
                               int in_ndim)
{
    int stride = 1;
    int offset = 0;
    for (int i = in_ndim - 1; i >= 0; --i) {
        int out_pos = out_ndim - (in_ndim - i);
        int coord = (in_sizes[i] == 1) ? 0 : out_index[out_pos];
        offset += coord * stride;
        stride *= in_sizes[i];
    }
    return offset;
}
```

This is used in the binary op body:

```metal
static inline void gd_binary(device const float *a,
                             device const float *b,
                             device float *out,
                             constant gd_metal_ew_params &p,
                             uint gid, int op)
{
    // Decompose linear gid into N-dimensional output index
    int index[GD_METAL_MAX_DIMS];
    int lin = (int)gid;
    for (int i = p.ndim - 1; i >= 0; --i) {
        index[i] = lin % p.out_sizes[i];
        lin /= p.out_sizes[i];
    }
    // Broadcast each input to the output index
    int ao = gd_broadcast_offset(index, p.ndim, p.a_sizes, p.a_ndim);
    int bo = gd_broadcast_offset(index, p.ndim, p.b_sizes, p.b_ndim);
    out[gid] = (op == 0) ? (a[ao] + b[bo]) : (a[ao] * b[bo]);
}
```

The same pattern generalizes to batch broadcasting in matmul — the batch dimensions (leading dims) use the same broadcast logic, while the matrix dimensions use standard row/col addressing.

---

## 10. Tiled GEMM (Shared Memory for Matmul)

The naive matmul kernel reads each element of A and B from device memory once *per output element* — O(M×N×K) reads. The tiled version reads each element from device memory once per tile instead — O(M×N×K / TILE) reads.

### The tiled kernel structure

```
One threadgroup → one TILE×TILE output block
Each thread in the threadgroup:
  1. Loads one element of A and one element of B from device memory into threadgroup memory
  2. Sync (barrier)
  3. Accumulates partial dot products from the threadgroup tile
  4. Sync (barrier)
  5. Repeat for next K-tile
```

```metal
#define GD_METAL_GEMM_TILE 16

kernel void gd_matmul_tiled(device const float *a [[buffer(0)]],
                            device const float *b [[buffer(1)]],
                            device float *out     [[buffer(2)]],
                            ..., uint3 tgpos [[threadgroup_position_in_grid]],
                            uint3 lid   [[thread_position_in_threadgroup]])
{
    threadgroup float As[16][16];
    threadgroup float Bs[16][16];

    int col = tgpos.x * 16 + lid.x;   // output column (global)
    int row = tgpos.y * 16 + lid.y;   // output row (global)
    int batch = tgpos.z;              // batch index (3D grid)

    float acc = 0.0f;
    int n_tiles = (K + 15) / 16;

    for (int t = 0; t < n_tiles; ++t) {
        // Each thread loads one element into threadgroup memory
        int a_k = t * 16 + lid.x;          // K index within this tile
        int b_k = t * 16 + lid.y;

        if (row < M && a_k < K)
            As[lid.y][lid.x] = A[row * K + a_k];
        else
            As[lid.y][lid.x] = 0.0f;

        if (col < N && b_k < K)
            Bs[lid.y][lid.x] = B[b_k * N + col];
        else
            Bs[lid.y][lid.x] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // One inner loop — each thread accumulates over the tile's K dimension
        for (int i = 0; i < 16; ++i)
            acc += As[lid.y][i] * Bs[i][lid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        out[row * N + col] = acc;
}
```

### Why tiling works

Device memory bandwidth is precious. Threadgroup memory is ~100× faster (SRAM vs DRAM). Each A/B element is loaded into threadgroup memory once per tile, then read 16× (the tile width) by different threads in the inner loop.

**Tile size = 16** is a safe choice for Apple GPUs. Larger tiles (32) reduce device reads further but consume more threadgroup memory (32² × 2 × 4 bytes = 8 KB — still fine) and reduce the number of threadgroups that can run concurrently.

For the `linear` layer (matmul with bias), the tiled kernel adds a bias term after accumulation:

```metal
out[r * out_features + o] = (p.has_bias ? bias[o] : 0.0f) + acc;
```

---

## 11. Softmax & Stable Numerics

The numerically stable softmax is a two-pass algorithm:

1. **Find max** over the reduction axis (prevent overflow in `exp`).
2. **Compute exp(x - max)**, sum, then divide.

```metal
kernel void gd_softmax(device const float *x    [[buffer(0)]],
                       device float *out        [[buffer(1)]],
                       constant gd_metal_softmax_params &p [[buffer(2)]],
                       uint gid                 [[thread_position_in_grid]])
{
    int total = p.outer * p.inner;
    if ((int)gid >= total) return;

    int o = (int)gid / p.inner;     // outer index
    int in = (int)gid % p.inner;    // inner index

    // Pass 1: find max
    float maxv = -INFINITY;
    for (int c = 0; c < p.d; ++c) {
        float v = x[(o * p.d + c) * p.inner + in];
        if (v > maxv) maxv = v;
    }

    // Pass 2: exp(x - max) and sum
    float sum = 0.0f;
    for (int c = 0; c < p.d; ++c) {
        float e = exp(x[(o * p.d + c) * p.inner + in] - maxv);
        out[idx] = e;        // store exp for reuse
        sum += e;
    }

    // Pass 3: normalize
    for (int c = 0; c < p.d; ++c) {
        out[idx] /= sum;
    }
}
```

This is **one thread per output position** — each thread processes its full row. For GPU efficiency on large rows, you'd use a threadgroup reduction instead (one threadgroup per row, cooperative exp+sum). The codebase uses the simpler per-thread loop for correctness-first v1.

### Cross-entropy backward

The backward combines softmax gradient with cross-entropy in one kernel:

```metal
// dlogits = scale * (softmax - onehot)
float scale = go_scalar[0] / (float)p.positions;
// recompute softmax to avoid storing it:
float maxv, sum;
// ... compute softmax stats ...
float pc = exp(logit - maxv) / sum;
float onehot = (c == target) ? 1.0f : 0.0f;
dlogits[...] = scale * (pc - onehot);
```

This recomputes the softmax during backward rather than storing the forward probabilities — a **memory-efficient tradeoff** typical of training frameworks.

---

## 12. Build Pipeline: .metal → .metallib

The Makefile rule compiles Metal shaders:

```makefile
# Compile .metal to .air (LLVM bitcode)
$(BUILD_DIR)/%.air: %.metal
    @mkdir -p $(@D)
    xcrun -sdk macosx metal -c $< -o $@

# Link .air files into .metallib (device-ready binary)
$(METALLIB): $(patsubst %.metal,$(BUILD_DIR)/%.air,$(METAL_SHADERS))
    @mkdir -p $(@D)
    xcrun -sdk macosx metallib $^ -o $@
```

### Runtime metallib discovery

```objc
NSString *metallibPath = nil;
char *env = getenv("GRADIENTS_METALLIB");
if (env) {
    metallibPath = [NSString stringWithUTF8String:env];
} else {
    // Default: next to the executable
    metallibPath = [[NSBundle mainBundle] pathForResource:@"gradients" ofType:@"metallib"];
}
NSURL *url = [NSURL fileURLWithPath:metallibPath];
id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
```

If the `.metallib` can't be found, the backend fails to initialize. The gradients.c logic falls back to CPU-only gracefully.

### Conditional compilation in MSL

```metal
#include <metal_stdlib>
#include "metal_kernel_types.h"   // shared with ObjC host
```

The shared header file must be valid in both C and MSL — use only plain `int`/`float` structs, no ObjC types.

---

## 13. Debugging & Validation

### 1. Bounds checking

The #1 Metal bug: out-of-bounds writes from oversized grids. Always guard:

```metal
if ((int)gid >= p.numel) return;
```

### 2. Parity testing

The gradients.c project tests every kernel via `gd_graph_compare(CPU, METAL, ...)`. This runs the same computation graph on both backends and compares element-by-element. Your workflow should be:

1. Implement the kernel.
2. Verify output on CPU with a reference implementation.
3. Run on Metal.
4. Compare every element at 1e-4 tolerance.

### 3. Command buffer errors

```objc
[cmd waitUntilCompleted];
if (cmd.status == MTLCommandBufferStatusError) {
    NSLog(@"Metal error: %@", cmd.error.localizedDescription);
    // Common: out-of-bounds access, invalid pipeline state, function not found
}
```

### 4. ASan / leak checking

Build with Address Sanitizer (`-fsanitize=address`). The metal backend uses `__bridge_retained`/`__bridge_transfer` for buffer handles — mismatched retains cause leaks or use-after-free.

### 5. Apple GPUs and float precision

Apple GPUs use IEEE 754 float32 for `float`. Accumulation order differs from CPU (parallel vs sequential), so expect small numerical differences (1e-4–1e-6 depending on op). Use relative + absolute tolerance comparison, not exact equality.

### 6. Common pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Missing barrier between threadgroup write and read | Wrong reduction result | `threadgroup_barrier(...)` |
| Grid overshoots output size | GPU crash (out of bounds) | Bounds guard + ensure `dispatchThreads:` grid matches |
| CPU writes after commit but before GPU reads | Stale data | Stage all inputs before `commit` |
| CPU reads before waitUntilCompleted | Stale results | Always `waitUntilCompleted` before reading shared buffer |
| Pipeline function name mismatch | `newFunctionWithName:` returns nil | Match kernel name exactly (case-sensitive) |
| Threadgroup memory too large | Build error or low occupancy | Keep threadgroup arrays ≤ 16 KB for 256 threads |

---

## 14. Reference: Kernel Catalog

All kernels from the gradients.c Metal backend, grouped by pattern.

### Elementwise (1 thread per element, 1D grid)

| Kernel | Op | Pattern |
|---|---|---|
| `gd_add` | `out = a + b` | Binary with broadcast |
| `gd_mul` | `out = a * b` | Binary with broadcast (same body, op flag) |
| `gd_scale` | `out = x * scale` | Unary, `constant` param |
| `gd_relu` | `out = max(x, 0)` | Unary |
| `gd_silu` | `out = x / (1 + exp(-x))` | Unary |
| `gd_copy` | `out = x` (bit-exact u32) | Elementwise word copy |
| `gd_cast` | F32↔I32 conversion | Elementwise with dtype dispatch |
| `gd_gelu` | GELU (tanh or erf approx) | Unary with math |
| `gd_scale` ← note | `step[0] += 1` | Single-element (gid==0 only) |

### Matmul/Linear (2D or 3D grid)

| Kernel | Grid | Feature |
|---|---|---|
| `gd_matmul` | 2D (N, M×batch) | Naive, transpose A/B, batch broadcast |
| `gd_matmul_tiled` | 3D (tile_N, tile_M, batch) | Threadgroup tiled, 16×16 tile |
| `gd_linear` | 2D (out_features, rows) | Naive, bias, transpose W |
| `gd_linear_tiled` | 3D (tile_out, tile_rows, 1) | Threadgroup tiled |

### Reductions (one thread per output position)

| Kernel | Reduction | Notes |
|---|---|---|
| `gd_reduce` | Sum/mean over one dim | `outer × d × inner` |
| `gd_softmax` | Stable softmax | Two-pass max-subtract |
| `gd_cross_entropy` | Scalar cross-entropy loss | Threadgroup reduction, I32 targets |
| `gd_reduce_to` | Broadcast backward sum | Scatter input across broadcast dims |
| `gd_rms_norm` | RMS normalization | Threadgroup cooperative tree reduction |
| `gd_rms_norm_bwd` | RMSNorm backward dx | Two-value tree reduction (x² + A) |
| `gd_rms_norm_wbwd` | RMSNorm backward dweight | Tiled rows, cached per-row rms_inv |

### Attention (SDPA)

| Kernel | Grid | Notes |
|---|---|---|
| `gd_sdpa` | (B × Hq × Tq) | Two-pass softmax, no score matrix |
| `gd_sdpa_bwd_dq` | (B × Hq × Tq) | Recomputes softmax, accumulates dq per query |
| `gd_sdpa_bwd_dkv` | (B × Hkv × Tk) | Loops query-heads, accumulates dk/dv per kv |

### Embedding & Position

| Kernel | Grid | Notes |
|---|---|---|
| `gd_embedding` | (N × dim) | Row gather: `out[p,c] = table[ids[p], c]` |
| `gd_embedding_bwd` | (vocab × dim) | Scatter-add: each (v,c) sums matching rows |
| `gd_transpose` | numel | Physical permute via `in_strides[perm[k]]` |
| `gd_rope` | B × T × heads | Rotary embedding, sin_sign for forward/backward |

### Optimizer

| Kernel | Notes |
|---|---|
| `gd_step_inc` | `step[0] += 1.0f`, single-thread |
| `gd_adamw` | AdamW with decoupled weight decay |

### Backward kernels (mirror forward patterns)

| Kernel | Gradient of | Pattern |
|---|---|---|
| `gd_relu_bwd` | ReLU | `dx = (x > 0) ? go : 0` |
| `gd_silu_bwd` | SiLU | `dx = go * s * (1 + x * (1 - s))` |
| `gd_softmax_bwd` | Softmax | `dx = y * (go - dot(go, y))` |
| `gd_sum_bwd`/`gd_mean_bwd` | Sum/Mean | Broadcast go back over d |
| `gd_cross_entropy_bwd` | Cross-entropy | `dlogits = scale * (softmax - onehot)` |
| `gd_gelu_bwd` | GELU | Derivative of tanh or erf approx |

---

## Appendix: Checklist for Adding a New Kernel

1. **Write the MSL kernel** in `kernels.metal`.
2. **Add params struct** in `metal_kernel_types.h` if needed.
3. **Register the mapping** in `g_metal_kernels[]` in `metal_backend.m`.
4. **Build pipeline** — the init loop automatically creates one for the new function name.
5. **Add encode case** in the `execute` encode loop (if the kernel needs special buffer binding).
6. **Extend `supports_node`** to recognize the new op.
7. **Run parity test** — existing `gd_graph_compare` against CPU.
8. **Add a standalone test** in `tests/test_metal.c` for the new kernel.
9. **Rebuild**: `make` picks up the `.metal` change, recompiles to `.air`, re-links `.metallib`.

---

*Derived from the gradients.c Metal backend (`src/backends/metal/kernels.metal`, `metal_backend.m`, `metal_kernel_types.h`). Full source at `/Users/mascit/projects/gradients.c`.*
