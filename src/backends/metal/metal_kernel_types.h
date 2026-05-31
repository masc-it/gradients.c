#ifndef GRADIENTS_METAL_KERNEL_TYPES_H
#define GRADIENTS_METAL_KERNEL_TYPES_H

/* Shared layout contract between the Objective-C host (metal_backend.m) and the
 * Metal shading language kernels (kernels.metal). Keep this POD and restricted
 * to types valid in both C and MSL (plain int/float). Sizes are element counts;
 * strides are not needed because v1 produced values are contiguous and inputs
 * broadcast by shape, matching the CPU reference kernels. */

#define GD_METAL_MAX_DIMS 8

/* Threadgroup tile size for the tiled GEMM kernels (matmul/linear). Shared so
 * the host dispatches matching threadgroup dimensions. */
#define GD_METAL_GEMM_TILE 16

/* Register-blocked GEMM: one threadgroup computes a GEMM_BM x GEMM_BN output
 * block, streaming K in GEMM_BK-deep tiles through threadgroup memory; each
 * thread owns a GEMM_TM x GEMM_TN micro-tile in registers (float4 rows when
 * GEMM_TN==4), so A/B values loaded once are reused across TM*TN FMAs. This
 * raises arithmetic intensity far above the 1-thread-per-output tiled kernel.
 * Threads per group = (GEMM_BN/GEMM_TN) x (GEMM_BM/GEMM_TM). Keep TN==4 so the
 * inner accumulator is a float4. */
#define GD_METAL_GEMM_BM 64
#define GD_METAL_GEMM_BN 64
#define GD_METAL_GEMM_BK 8
#define GD_METAL_GEMM_TM 4
#define GD_METAL_GEMM_TN 4

/* FlashAttention-style tiled SDPA forward: query-block (threads per group), key
 * tile (staged in threadgroup memory), and the max head_dim the tiled kernel
 * supports. The host falls back to the reference SDPA kernel when
 * head_dim > GD_METAL_SDPA_DHT. Shared so host dispatch matches the kernel. */
#define GD_METAL_SDPA_BQ 64
#define GD_METAL_SDPA_BK 16
#define GD_METAL_SDPA_DHT 64
/* dK/dV split kernel: one threadgroup handles this many keys, with the 64
 * threads divided into key lanes over head_dim channels. */
#define GD_METAL_SDPA_DKV_KEYS 8

/* Fused LM-head cross entropy processes vocab in chunks so logits/dlogits
 * scratch is [N,chunk] instead of [N,V]. */
#define GD_METAL_LMCE_CHUNK 1024

/* Split-K / flash-decoding for long-context forward SDPA. A causal kernel is
 * critical-path bound: the heaviest query block must scan all Tk keys, and that
 * one threadgroup saturates the GPU regardless of how many lighter blocks the
 * causal block-skip removed. Split-K shortens that path by partitioning the key
 * range across up to GD_METAL_SDPA_SPLIT_MAX threadgroups (each scans ~SPLIT_MIN
 * keys), writing partial online-softmax state to scratch, then a cheap combine
 * pass merges the splits. Splits are auto-selected from Tk: ceil(Tk/SPLIT_MIN)
 * capped at SPLIT_MAX, so short sequences resolve to 1 split and stay on the
 * existing single-pass kernel (no scratch, no regression). */
#define GD_METAL_SDPA_SPLIT_MIN 128
#define GD_METAL_SDPA_SPLIT_MAX 8

/* Fused add+rms_norm kernels cache one normalized row in threadgroup memory to
 * avoid materializing/reloading the residual-add output inside the norm. Keep
 * conservative; shapes above this limit fall back to unfused RMS kernels. */
#define GD_METAL_FUSED_RMS_MAX 2048

/* Elementwise binary ops (add/mul) with NumPy-style right-aligned broadcasting.
 * `out_sizes` describes the contiguous output; `a_sizes`/`b_sizes` describe each
 * input's own shape so the shader can reproduce broadcast_offset() from the CPU
 * kernel exactly. */
typedef struct gd_metal_ew_params {
    int ndim;                          /* output rank */
    int numel;                         /* output element count */
    int a_ndim;
    int b_ndim;
    int same_shape;                    /* a,b,out all same contiguous shape */
    int out_sizes[GD_METAL_MAX_DIMS];
    int a_sizes[GD_METAL_MAX_DIMS];
    int b_sizes[GD_METAL_MAX_DIMS];
} gd_metal_ew_params;

/* Unary elementwise ops over a contiguous tensor (scale/relu/silu/copy). `scale`
 * is only read by SCALE; the rest ignore it. */
typedef struct gd_metal_unary_params {
    int numel;
    float scale;
} gd_metal_unary_params;

typedef struct gd_metal_powlu_params {
    int numel;
    float m;
} gd_metal_powlu_params;

/* dtype codes shared with the cast kernel (a small closed set, not the full
 * gd_dtype enum, so the host maps explicitly). */
#define GD_METAL_DT_F32 0
#define GD_METAL_DT_I32 1

typedef struct gd_metal_cast_params {
    int numel;
    int src_dtype;  /* GD_METAL_DT_* */
    int dst_dtype;  /* GD_METAL_DT_* */
} gd_metal_cast_params;

/* Batched matmul with NumPy-style batch broadcasting and optional transposes.
 * The shader reproduces the CPU reference's per-batch base-offset computation,
 * so batch_sizes carry only the leading (rank-2) dims. */
typedef struct gd_metal_matmul_params {
    int m;
    int n;
    int k;
    int a_cols;        /* a's last-dim size, for addressing */
    int b_cols;        /* b's last-dim size */
    int a_mat;         /* elements per a matrix (a_rows*a_cols) */
    int b_mat;         /* elements per b matrix */
    int out_mat;       /* m*n */
    int trans_a;
    int trans_b;
    int batch_ndim;    /* out rank - 2 */
    int a_batch_ndim;  /* a rank - 2 */
    int b_batch_ndim;  /* b rank - 2 */
    int out_batch_sizes[GD_METAL_MAX_DIMS];
    int a_batch_sizes[GD_METAL_MAX_DIMS];
    int b_batch_sizes[GD_METAL_MAX_DIMS];
} gd_metal_matmul_params;

typedef struct gd_metal_linear_params {
    int rows;
    int in_features;
    int out_features;
    int trans_w;
    int has_bias;
} gd_metal_linear_params;

/* sum/mean over one dim: x viewed as [outer, d, inner], reduced over d. */
typedef struct gd_metal_reduce_params {
    int outer;
    int inner;
    int d;
    int mean;
} gd_metal_reduce_params;

/* softmax over one dim: x viewed as [outer, d, inner]. */
typedef struct gd_metal_softmax_params {
    int outer;
    int inner;
    int d;
} gd_metal_softmax_params;

typedef struct gd_metal_rmsnorm_params {
    int rows;
    int last;
    float eps;
} gd_metal_rmsnorm_params;

/* cross_entropy: logits viewed as [outer, classes, inner], scalar mean loss.
 * Also used by cross_entropy_bwd. */
typedef struct gd_metal_ce_params {
    int outer;
    int inner;
    int classes;
    int positions;     /* outer*inner */
} gd_metal_ce_params;

typedef struct gd_metal_lmce_params {
    int rows;
    int dim;
    int vocab;
    int chunk_start;
    int chunk_size;
    int first_chunk;
} gd_metal_lmce_params;

/* reduce_to: sum `go` (broadcast shape) down into the target shape. */
typedef struct gd_metal_reduce_to_params {
    int target_ndim;
    int target_numel;
    int go_ndim;
    int go_numel;
    int target_sizes[GD_METAL_MAX_DIMS];
    int go_sizes[GD_METAL_MAX_DIMS];
} gd_metal_reduce_to_params;

typedef struct gd_metal_adamw_params {
    int numel;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
} gd_metal_adamw_params;

typedef struct gd_metal_gelu_params {
    int numel;
    int tanh_approx;
} gd_metal_gelu_params;

/* Physical permute of contiguous 4-byte elements. in_strides are element
 * strides of the contiguous input; out is contiguous. */
typedef struct gd_metal_transpose_params {
    int ndim;
    int numel;
    int out_sizes[GD_METAL_MAX_DIMS];
    int in_strides[GD_METAL_MAX_DIMS];
    int perm[GD_METAL_MAX_DIMS];
} gd_metal_transpose_params;

typedef struct gd_metal_embedding_params {
    int n;     /* number of ids */
    int dim;   /* embedding dimension */
    int vocab; /* table rows */
} gd_metal_embedding_params;

/* Scaled dot-product attention. Head-major q[B,Tq,Hq,Dh], k/v[B,Tk,Hkv,Dh]. */
typedef struct gd_metal_sdpa_params {
    int B;
    int Tq;
    int Tk;
    int Hq;
    int Hkv;
    int Dh;
    float scale;
    int causal;
    int window;
    int prefix_len;
    int has_bias;
    int Bb;   /* bias broadcast dims over [B, Hq, Tq, Tk] (each 1 or full) */
    int Hb;
    int Tqb;
    int Tkb;
    int n_splits;   /* split-K: number of key partitions (>=1) */
    int split_len;  /* split-K: keys per partition (BK-aligned multiple) */
} gd_metal_sdpa_params;

/* Rotary embedding; one thread per (.., head) row over head_dim. sin_sign is
 * +1 forward, -1 backward (transpose rotation). */
typedef struct gd_metal_rope_params {
    int rows;        /* numel / head_dim */
    int heads;       /* size of the heads axis */
    int head_dim;
    int n_dims;      /* rotary dims (even, <= head_dim) */
    int interleaved;
    float theta;
    float sin_sign;
} gd_metal_rope_params;

#endif /* GRADIENTS_METAL_KERNEL_TYPES_H */
