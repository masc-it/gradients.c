#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define GD_NEW_OP_NAME_MAX 96U
#define GD_NEW_OP_PATH_MAX 4096U
#define GD_NEW_OP_CONTENT_MAX 32768U
#define GD_NEW_OP_NOTES_MAX 2048U

typedef struct gd_new_op_options {
    const char *op;
    bool binary;
    bool no_backend;
    bool f16_only;
    bool f16_f32_accum;
    bool save_stats;
    bool reduction;
} gd_new_op_options;

static bool gd_valid_op_name(const char *name)
{
    size_t i;
    bool prev_underscore = false;
    if (name == NULL || name[0] == '\0' || !islower((unsigned char)name[0])) {
        return false;
    }
    for (i = 0U; name[i] != '\0'; ++i) {
        unsigned char c = (unsigned char)name[i];
        if (islower(c) || isdigit(c)) {
            prev_underscore = false;
            continue;
        }
        if (c == '_') {
            if (prev_underscore || name[i + 1U] == '\0') {
                return false;
            }
            prev_underscore = true;
            continue;
        }
        return false;
    }
    return i < GD_NEW_OP_NAME_MAX;
}

static void gd_upper_from_name(const char *name, char *out, size_t out_size)
{
    size_t i;
    if (out == NULL || out_size == 0U) {
        return;
    }
    for (i = 0U; name != NULL && name[i] != '\0' && i + 1U < out_size; ++i) {
        out[i] = (char)toupper((unsigned char)name[i]);
    }
    out[i] = '\0';
}

static bool gd_path_exists(const char *path)
{
    struct stat st;
    return path != NULL && stat(path, &st) == 0;
}

static bool gd_dir_exists(const char *path)
{
    struct stat st;
    return path != NULL && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int gd_mkdir_if_missing(const char *path)
{
    if (gd_dir_exists(path)) {
        return 0;
    }
    if (mkdir(path, 0777) != 0) {
        fprintf(stderr, "gradients-new-op: mkdir %s: %s\n", path, strerror(errno));
        return 1;
    }
    printf("[create] %s/\n", path);
    return 0;
}

static int gd_write_new_file(const char *path, const char *content)
{
    FILE *f;
    size_t len;
    if (gd_path_exists(path)) {
        printf("[exists] %s\n", path);
        return 0;
    }
    f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "gradients-new-op: write %s: %s\n", path, strerror(errno));
        return 1;
    }
    len = strlen(content);
    if (fwrite(content, 1U, len, f) != len) {
        fprintf(stderr, "gradients-new-op: short write %s\n", path);
        fclose(f);
        return 1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "gradients-new-op: close %s: %s\n", path, strerror(errno));
        return 1;
    }
    printf("[create] %s\n", path);
    return 0;
}

static bool gd_make_op_dir_path(char *out, size_t out_size, const char *op)
{
    int n = snprintf(out, out_size, "src/ops/%s", op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_op_file_path(char *out,
                                 size_t out_size,
                                 const char *op,
                                 const char *prefix,
                                 const char *suffix)
{
    int n = snprintf(out, out_size, "src/ops/%s/%s%s%s", op, prefix, op, suffix);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_op_named_path(char *out, size_t out_size, const char *op, const char *filename)
{
    int n = snprintf(out, out_size, "src/ops/%s/%s", op, filename);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_fixed_append(char *out, size_t out_size, size_t *len, const char *text)
{
    size_t text_len;
    if (out == NULL || len == NULL || text == NULL || out_size == 0U) {
        return false;
    }
    text_len = strlen(text);
    if (*len >= out_size || text_len >= out_size - *len) {
        return false;
    }
    memcpy(out + *len, text, text_len + 1U);
    *len += text_len;
    return true;
}

static bool gd_fixed_append_note(char *out,
                                 size_t out_size,
                                 size_t *len,
                                 const char *indent,
                                 const char *text)
{
    return gd_fixed_append(out, out_size, len, indent) &&
           gd_fixed_append(out, out_size, len, text) &&
           gd_fixed_append(out, out_size, len, "\n");
}

static bool gd_make_option_notes(char *out,
                                 size_t out_size,
                                 const gd_new_op_options *opts,
                                 const char *indent)
{
    size_t len = 0U;
    if (out == NULL || opts == NULL || indent == NULL || out_size == 0U) {
        return false;
    }
    out[0] = '\0';
    if (!gd_fixed_append_note(out,
                              out_size,
                              &len,
                              indent,
                              "See docs/guides/metal_tips.md before implementing Metal hot paths.")) {
        return false;
    }
    if (opts->no_backend &&
        !gd_fixed_append_note(out,
                              out_size,
                              &len,
                              indent,
                              "Custom backend mode: generated backend stubs are omitted; add custom backend declarations/PSOs manually.")) {
        return false;
    }
    if (opts->f16_only &&
        !gd_fixed_append_note(out,
                              out_size,
                              &len,
                              indent,
                              "F16-only: reject other dtypes in core/host validation and keep kernels dtype-specialized.")) {
        return false;
    }
    if (opts->f16_f32_accum &&
        !gd_fixed_append_note(out,
                              out_size,
                              &len,
                              indent,
                              "F16+FP32 accumulation: keep reductions/normalization in float and store stats/losses as F32.")) {
        return false;
    }
    if (opts->save_stats &&
        !gd_fixed_append_note(out,
                              out_size,
                              &len,
                              indent,
                              "Save-stats: save compact forward tensors with gd_autograd_record(..., saved, n_saved) for fast backward.")) {
        return false;
    }
    if (opts->reduction &&
        !gd_fixed_append_note(out,
                              out_size,
                              &len,
                              indent,
                              "Reduction: use SIMD reductions first, then threadgroup reductions, with shape-adaptive simdgroup counts.")) {
        return false;
    }
    return true;
}

static bool gd_make_def_content(char *out,
                                size_t out_size,
                                const gd_new_op_options *opts)
{
    const char *shape;
    int n;
    if (out == NULL || opts == NULL) {
        return false;
    }
    shape = opts->binary ? "binary" : "unary";
    if (opts->no_backend) {
        n = snprintf(out,
                     out_size,
                     "# Generated op metadata for %s.\n"
                     "# api shape controls generated public stubs.\n"
                     "# backend omitted: implement custom backend declarations/dispatch manually.\n"
                     "api=%s\n",
                     opts->op,
                     shape);
    } else {
        n = snprintf(out,
                     out_size,
                     "# Generated op metadata for %s.\n"
                     "# api/backend shape controls generated public/backend stubs.\n"
                     "api=%s\n"
                     "backend=%s\n",
                     opts->op,
                     shape,
                     shape);
    }
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_core_content(char *out,
                                 size_t out_size,
                                 const gd_new_op_options *opts,
                                 const char *upper)
{
    char notes[GD_NEW_OP_NOTES_MAX];
    int n;
    if (!gd_make_option_notes(notes, sizeof(notes), opts, "   ")) {
        return false;
    }
    n = snprintf(out,
                 out_size,
                 "#include <gradients/ops.h>\n"
                 "\n"
                 "#include \"../op_common.h\"\n"
                 "\n"
                 "/* Scaffold for the '%s' op.\n"
                 "\n"
                 "   Next steps:\n"
                 "   1. Confirm generated public declaration in include/gradients/ops_generated.h.\n"
                 "   2. Replace this anchor with gd_%s(...) validation/allocation/backend dispatch.\n"
                 "   3. Record the op with gd_autograd_record(ctx, GD_OP_%s, ...).\n"
                 "   4. Add backend implementations/tests/probes as needed.\n"
                 "\n"
                 "%s"
                 "*/\n"
                 "typedef int gd_%s_core_scaffold_anchor;\n",
                 opts->op,
                 opts->op,
                 upper,
                 notes,
                 opts->op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_autograd_content(char *out,
                                     size_t out_size,
                                     const gd_new_op_options *opts,
                                     const char *upper)
{
    char notes[GD_NEW_OP_NOTES_MAX];
    int n;
    if (!gd_make_option_notes(notes, sizeof(notes), opts, " * ")) {
        return false;
    }
    n = snprintf(out,
                 out_size,
                 "#include \"../autograd_impl.h\"\n"
                 "\n"
                 "/* Backward implementation notes:\n"
                 "%s"
                 " */\n"
                 "static gd_status gd_%s_autograd_backward(gd_bwd_ctx *bwd,\n"
                 "                                            const gd_tape_node *node)\n"
                 "{\n"
                 "    (void)bwd;\n"
                 "    (void)node;\n"
                 "    return GD_ERR_UNSUPPORTED;\n"
                 "}\n"
                 "\n"
                 "const gd_autograd_rule gd_bwd_rule_%s = {\n"
                 "    .kind = GD_OP_%s,\n"
                 "    .name = \"%s\",\n"
                 "    .backward = gd_%s_autograd_backward,\n"
                 "};\n",
                 notes,
                 opts->op,
                 opts->op,
                 upper,
                 opts->op,
                 opts->op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_metal_content_custom(char *out,
                                         size_t out_size,
                                         const gd_new_op_options *opts)
{
    char notes[GD_NEW_OP_NOTES_MAX];
    int n;
    if (!gd_make_option_notes(notes, sizeof(notes), opts, "   ")) {
        return false;
    }
    n = snprintf(out,
                 out_size,
                 "#include \"../../backends/metal/metal_backend_internal.h\"\n"
                 "#include \"metal_%s_types.h\"\n"
                 "\n"
                 "/* Custom Metal backend capsule for '%s'.\n"
                 "\n"
                 "   backend= is omitted in op_%s.def. Add custom backend declarations\n"
                 "   to src/core/backend.h and PSO creation/release in\n"
                 "   src/backends/metal/backend_metal.m when implementing this op.\n"
                 "\n"
                 "%s"
                 "*/\n"
                 "typedef int gd_%s_metal_scaffold_anchor;\n",
                 opts->op,
                 opts->op,
                 opts->op,
                 notes,
                 opts->op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_metal_content_binary(char *out,
                                         size_t out_size,
                                         const gd_new_op_options *opts,
                                         const char *upper)
{
    char notes[GD_NEW_OP_NOTES_MAX];
    int n;
    if (!gd_make_option_notes(notes, sizeof(notes), opts, " * ")) {
        return false;
    }
    n = snprintf(out,
                 out_size,
                 "#include \"../../backends/metal/metal_backend_internal.h\"\n"
                 "#include \"metal_%s_types.h\"\n"
                 "\n"
                 "/* Metal backend scaffold for '%s'.\n"
                 "%s"
                 " * TODO: validate views, zero-init ABI args, encode gd_%s_kernel /\n"
                 " * gd_%s_bcast_kernel / gd_%s_row_bcast_kernel, and dispatch inside\n"
                 " * gd_metal_command_for_op(). Return GD_ERR_UNSUPPORTED until done.\n"
                 " */\n"
                 "static id<MTLComputePipelineState> gd_%s_pso(gd_backend *backend)\n"
                 "{\n"
                 "    return (__bridge id<MTLComputePipelineState>)backend->binary_pso[GD_OP_%s];\n"
                 "}\n"
                 "\n"
                 "static id<MTLComputePipelineState> gd_%s_bcast_pso(gd_backend *backend)\n"
                 "{\n"
                 "    return (__bridge id<MTLComputePipelineState>)backend->binary_bcast_pso[GD_OP_%s];\n"
                 "}\n"
                 "\n"
                 "static id<MTLComputePipelineState> gd_%s_row_bcast_pso(gd_backend *backend)\n"
                 "{\n"
                 "    return (__bridge id<MTLComputePipelineState>)backend->binary_row_bcast_pso[GD_OP_%s];\n"
                 "}\n"
                 "\n"
                 "gd_status gd_backend_%s(gd_backend *backend,\n"
                 "                        const gd_backend_tensor_view *x,\n"
                 "                        const gd_backend_tensor_view *y,\n"
                 "                        const gd_backend_tensor_view *out)\n"
                 "{\n"
                 "    if (backend == NULL || x == NULL || y == NULL || out == NULL) {\n"
                 "        return GD_ERR_INVALID_ARGUMENT;\n"
                 "    }\n"
                 "    (void)gd_%s_pso(backend);\n"
                 "    (void)gd_%s_bcast_pso(backend);\n"
                 "    (void)gd_%s_row_bcast_pso(backend);\n"
                 "    (void)x;\n"
                 "    (void)y;\n"
                 "    (void)out;\n"
                 "    return GD_ERR_UNSUPPORTED;\n"
                 "}\n",
                 opts->op,
                 opts->op,
                 notes,
                 opts->op,
                 opts->op,
                 opts->op,
                 opts->op,
                 upper,
                 opts->op,
                 upper,
                 opts->op,
                 upper,
                 opts->op,
                 opts->op,
                 opts->op,
                 opts->op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_metal_content_unary(char *out,
                                        size_t out_size,
                                        const gd_new_op_options *opts,
                                        const char *upper)
{
    char notes[GD_NEW_OP_NOTES_MAX];
    int n;
    if (!gd_make_option_notes(notes, sizeof(notes), opts, " * ")) {
        return false;
    }
    n = snprintf(out,
                 out_size,
                 "#include \"../../backends/metal/metal_backend_internal.h\"\n"
                 "#include \"metal_%s_types.h\"\n"
                 "\n"
                 "/* Metal backend scaffold for '%s'.\n"
                 "%s"
                 " * TODO: validate views, zero-init gd_metal_%s_args, encode kernels,\n"
                 " * and dispatch inside gd_metal_command_for_op(). Return\n"
                 " * GD_ERR_UNSUPPORTED until implemented.\n"
                 " */\n"
                 "static id<MTLComputePipelineState> gd_%s_pso(gd_backend *backend)\n"
                 "{\n"
                 "    return (__bridge id<MTLComputePipelineState>)backend->unary_pso[GD_OP_%s];\n"
                 "}\n"
                 "\n"
                 "static id<MTLComputePipelineState> gd_%s_backward_pso(gd_backend *backend)\n"
                 "{\n"
                 "    return (__bridge id<MTLComputePipelineState>)backend->unary_backward_pso[GD_OP_%s];\n"
                 "}\n"
                 "\n"
                 "gd_status gd_backend_%s(gd_backend *backend,\n"
                 "                        const gd_backend_tensor_view *x,\n"
                 "                        const gd_backend_tensor_view *y)\n"
                 "{\n"
                 "    if (backend == NULL || x == NULL || y == NULL) {\n"
                 "        return GD_ERR_INVALID_ARGUMENT;\n"
                 "    }\n"
                 "    (void)gd_%s_pso(backend);\n"
                 "    (void)x;\n"
                 "    (void)y;\n"
                 "    return GD_ERR_UNSUPPORTED;\n"
                 "}\n"
                 "\n"
                 "gd_status gd_backend_%s_backward(gd_backend *backend,\n"
                 "                                 const gd_backend_tensor_view *x,\n"
                 "                                 const gd_backend_tensor_view *grad_out,\n"
                 "                                 const gd_backend_tensor_view *grad_x)\n"
                 "{\n"
                 "    if (backend == NULL || x == NULL || grad_out == NULL || grad_x == NULL) {\n"
                 "        return GD_ERR_INVALID_ARGUMENT;\n"
                 "    }\n"
                 "    (void)gd_%s_backward_pso(backend);\n"
                 "    (void)x;\n"
                 "    (void)grad_out;\n"
                 "    (void)grad_x;\n"
                 "    return GD_ERR_UNSUPPORTED;\n"
                 "}\n",
                 opts->op,
                 opts->op,
                 notes,
                 opts->op,
                 opts->op,
                 upper,
                 opts->op,
                 upper,
                 opts->op,
                 opts->op,
                 opts->op,
                 opts->op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_metal_content(char *out,
                                  size_t out_size,
                                  const gd_new_op_options *opts,
                                  const char *upper)
{
    if (opts == NULL) {
        return false;
    }
    if (opts->no_backend) {
        return gd_make_metal_content_custom(out, out_size, opts);
    }
    if (opts->binary) {
        return gd_make_metal_content_binary(out, out_size, opts, upper);
    }
    return gd_make_metal_content_unary(out, out_size, opts, upper);
}

static bool gd_make_metal_types_content(char *out,
                                        size_t out_size,
                                        const gd_new_op_options *opts,
                                        const char *upper)
{
    char notes[GD_NEW_OP_NOTES_MAX];
    int n;
    if (out == NULL || opts == NULL || upper == NULL) {
        return false;
    }
    if (!gd_make_option_notes(notes, sizeof(notes), opts, " * ")) {
        return false;
    }
    if (opts->binary) {
        n = snprintf(out,
                     out_size,
                     "#ifndef GD_OP_%s_METAL_TYPES_H\n"
                     "#define GD_OP_%s_METAL_TYPES_H\n"
                     "\n"
                     "/* Binary elementwise ops use the shared binary Metal ABI.\n"
                     "%s"
                     " */\n"
                     "#include \"../_shared/binary/metal_binary_types.h\"\n"
                     "\n"
                     "#endif /* GD_OP_%s_METAL_TYPES_H */\n",
                     upper,
                     upper,
                     notes,
                     upper);
        return n >= 0 && (size_t)n < out_size;
    }
    if (opts->f16_only) {
        n = snprintf(out,
                     out_size,
                     "#ifndef GD_OP_%s_METAL_TYPES_H\n"
                     "#define GD_OP_%s_METAL_TYPES_H\n"
                     "\n"
                     "/* Op-local Metal ABI types for %s. Keep host/Metal layouts in sync.\n"
                     "%s"
                     " */\n"
                     "#include \"../../backends/metal/metal_abi.h\"\n"
                     "\n"
                     "typedef struct gd_metal_%s_args {\n"
                     "    gd_metal_u64 x_offset;\n"
                     "    gd_metal_u64 y_offset;\n"
                     "    gd_metal_u64 grad_offset;\n"
                     "    gd_metal_u64 count;\n"
                     "} gd_metal_%s_args;\n"
                     "\n"
                     "#ifndef __METAL_VERSION__\n"
                     "_Static_assert(sizeof(gd_metal_%s_args) == 32U, \"gd_metal_%s_args ABI mismatch\");\n"
                     "#endif\n"
                     "\n"
                     "#endif /* GD_OP_%s_METAL_TYPES_H */\n",
                     upper,
                     upper,
                     opts->op,
                     notes,
                     opts->op,
                     opts->op,
                     opts->op,
                     opts->op,
                     upper);
        return n >= 0 && (size_t)n < out_size;
    }
    n = snprintf(out,
                 out_size,
                 "#ifndef GD_OP_%s_METAL_TYPES_H\n"
                 "#define GD_OP_%s_METAL_TYPES_H\n"
                 "\n"
                 "/* Op-local Metal ABI types for %s. Keep host/Metal layouts in sync.\n"
                 "%s"
                 " */\n"
                 "#include \"../../backends/metal/metal_abi.h\"\n"
                 "\n"
                 "typedef struct gd_metal_%s_args {\n"
                 "    gd_metal_u64 x_offset;\n"
                 "    gd_metal_u64 y_offset;\n"
                 "    gd_metal_u64 grad_offset;\n"
                 "    gd_metal_u64 count;\n"
                 "    gd_metal_u32 dtype;\n"
                 "    gd_metal_u32 pad0;\n"
                 "} gd_metal_%s_args;\n"
                 "\n"
                 "#ifndef __METAL_VERSION__\n"
                 "_Static_assert(sizeof(gd_metal_%s_args) == 40U, \"gd_metal_%s_args ABI mismatch\");\n"
                 "#endif\n"
                 "\n"
                 "#endif /* GD_OP_%s_METAL_TYPES_H */\n",
                 upper,
                 upper,
                 opts->op,
                 notes,
                 opts->op,
                 opts->op,
                 opts->op,
                 opts->op,
                 upper);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_metal_kernel_content(char *out,
                                         size_t out_size,
                                         const gd_new_op_options *opts)
{
    char notes[GD_NEW_OP_NOTES_MAX];
    int n;
    if (out == NULL || opts == NULL) {
        return false;
    }
    if (!gd_make_option_notes(notes, sizeof(notes), opts, "   ")) {
        return false;
    }
    if (opts->binary) {
        n = snprintf(out,
                     out_size,
                     "#include <metal_stdlib>\n"
                     "#include \"metal_%s_types.h\"\n"
                     "#include \"../_shared/binary/metal_binary_metal.h\"\n"
                     "\n"
                     "using namespace metal;\n"
                     "\n"
                     "/* Scaffold for the '%s' binary op-local Metal kernels.\n"
                     "\n"
                     "   The generated Metal PSO glue expects binary ops to export:\n"
                     "     gd_%s_kernel\n"
                     "     gd_%s_bcast_kernel\n"
                     "     gd_%s_row_bcast_kernel\n"
                     "\n"
                     "%s"
                     "*/\n"
                     "kernel void gd_%s_kernel(device const uchar *x [[buffer(0)]],\n"
                     "                         device const uchar *y [[buffer(1)]],\n"
                     "                         device uchar *out [[buffer(2)]],\n"
                     "                         constant gd_metal_binary_args &args [[buffer(3)]],\n"
                     "                         uint gid [[thread_position_in_grid]])\n"
                     "{\n"
                     "    (void)x;\n"
                     "    (void)y;\n"
                     "    (void)out;\n"
                     "    (void)args;\n"
                     "    (void)gid;\n"
                     "}\n"
                     "\n"
                     "kernel void gd_%s_bcast_kernel(device const uchar *x [[buffer(0)]],\n"
                     "                               device const uchar *y [[buffer(1)]],\n"
                     "                               device uchar *out [[buffer(2)]],\n"
                     "                               constant gd_metal_binary_bcast_args &args [[buffer(3)]],\n"
                     "                               uint gid [[thread_position_in_grid]])\n"
                     "{\n"
                     "    (void)x;\n"
                     "    (void)y;\n"
                     "    (void)out;\n"
                     "    (void)args;\n"
                     "    (void)gid;\n"
                     "}\n"
                     "\n"
                     "kernel void gd_%s_row_bcast_kernel(device const uchar *x [[buffer(0)]],\n"
                     "                                   device const uchar *y [[buffer(1)]],\n"
                     "                                   device uchar *out [[buffer(2)]],\n"
                     "                                   constant gd_metal_binary_bcast_args &args [[buffer(3)]],\n"
                     "                                   uint2 gid [[thread_position_in_grid]])\n"
                     "{\n"
                     "    (void)x;\n"
                     "    (void)y;\n"
                     "    (void)out;\n"
                     "    (void)args;\n"
                     "    (void)gid;\n"
                     "}\n",
                     opts->op,
                     opts->op,
                     opts->op,
                     opts->op,
                     opts->op,
                     notes,
                     opts->op,
                     opts->op,
                     opts->op);
        return n >= 0 && (size_t)n < out_size;
    }
    n = snprintf(out,
                 out_size,
                 "#include <metal_stdlib>\n"
                 "#include \"metal_%s_types.h\"\n"
                 "\n"
                 "using namespace metal;\n"
                 "\n"
                 "/* Scaffold for the '%s' op-local Metal kernels.\n"
                 "\n"
                 "   The generated Metal PSO glue expects unary ops to export:\n"
                 "     gd_%s_kernel\n"
                 "     gd_%s_backward_kernel\n"
                 "\n"
                 "%s"
                 "*/\n"
                 "kernel void gd_%s_kernel(device const uchar *x [[buffer(0)]],\n"
                 "                         device uchar *y [[buffer(1)]],\n"
                 "                         constant gd_metal_%s_args &args [[buffer(2)]],\n"
                 "                         uint gid [[thread_position_in_grid]])\n"
                 "{\n"
                 "    (void)x;\n"
                 "    (void)y;\n"
                 "    (void)args;\n"
                 "    (void)gid;\n"
                 "}\n"
                 "\n"
                 "kernel void gd_%s_backward_kernel(device const uchar *x [[buffer(0)]],\n"
                 "                                  device const uchar *grad_out [[buffer(1)]],\n"
                 "                                  device uchar *grad_x [[buffer(2)]],\n"
                 "                                  constant gd_metal_%s_args &args [[buffer(3)]],\n"
                 "                                  uint gid [[thread_position_in_grid]])\n"
                 "{\n"
                 "    (void)x;\n"
                 "    (void)grad_out;\n"
                 "    (void)grad_x;\n"
                 "    (void)args;\n"
                 "    (void)gid;\n"
                 "}\n",
                 opts->op,
                 opts->op,
                 opts->op,
                 opts->op,
                 notes,
                 opts->op,
                 opts->op,
                 opts->op,
                 opts->op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_fwd_py_content(char *out, size_t out_size, const char *op)
{
    int n = snprintf(out,
                     out_size,
                     "# /// script\n"
                     "# requires-python = \">=3.11\"\n"
                     "# dependencies = [\"torch\", \"numpy\"]\n"
                     "# ///\n"
                     "\n"
                     "\"\"\"Forward correctness harness template for gd_%s.\n"
                     "\n"
                     "Fill in a C runner that calls gd_%s, then compare its output\n"
                     "against a PyTorch reference. Run from the repository root with:\n"
                     "\n"
                     "    uv run src/ops/%s/fwd.py\n"
                     "\"\"\"\n"
                     "\n"
                     "from __future__ import annotations\n"
                     "\n"
                     "import subprocess\n"
                     "from pathlib import Path\n"
                     "\n"
                     "import numpy as np\n"
                     "import torch\n"
                     "\n"
                     "\n"
                     "def repo_root() -> Path:\n"
                     "    return Path(__file__).resolve().parents[3]\n"
                     "\n"
                     "\n"
                     "def build_library(root: Path) -> None:\n"
                     "    subprocess.run([\"make\", \"build\"], cwd=root, check=True)\n"
                     "\n"
                     "\n"
                     "def main() -> None:\n"
                     "    root = repo_root()\n"
                     "    build_library(root)\n"
                     "    _ = (np, torch)\n"
                     "    print(\"TODO: implement gd_%s forward PyTorch comparison\")\n"
                     "\n"
                     "\n"
                     "if __name__ == \"__main__\":\n"
                     "    main()\n",
                     op,
                     op,
                     op,
                     op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_bwd_py_content(char *out, size_t out_size, const char *op)
{
    int n = snprintf(out,
                     out_size,
                     "# /// script\n"
                     "# requires-python = \">=3.11\"\n"
                     "# dependencies = [\"torch\", \"numpy\"]\n"
                     "# ///\n"
                     "\n"
                     "\"\"\"Backward/autograd correctness harness template for gd_%s.\n"
                     "\n"
                     "Fill in a C runner that records gd_%s on the autograd tape, calls\n"
                     "gd_backward or gd_backward_many, then compare gradients against\n"
                     "PyTorch autograd. Run from the repository root with:\n"
                     "\n"
                     "    uv run src/ops/%s/bwd.py\n"
                     "\"\"\"\n"
                     "\n"
                     "from __future__ import annotations\n"
                     "\n"
                     "import subprocess\n"
                     "from pathlib import Path\n"
                     "\n"
                     "import numpy as np\n"
                     "import torch\n"
                     "\n"
                     "\n"
                     "def repo_root() -> Path:\n"
                     "    return Path(__file__).resolve().parents[3]\n"
                     "\n"
                     "\n"
                     "def build_library(root: Path) -> None:\n"
                     "    subprocess.run([\"make\", \"build\"], cwd=root, check=True)\n"
                     "\n"
                     "\n"
                     "def main() -> None:\n"
                     "    root = repo_root()\n"
                     "    build_library(root)\n"
                     "    _ = (np, torch)\n"
                     "    print(\"TODO: implement gd_%s backward PyTorch/autograd comparison\")\n"
                     "\n"
                     "\n"
                     "if __name__ == \"__main__\":\n"
                     "    main()\n",
                     op,
                     op,
                     op,
                     op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_readme_content(char *out,
                                   size_t out_size,
                                   const gd_new_op_options *opts)
{
    char notes[GD_NEW_OP_NOTES_MAX];
    int n;
    if (!gd_make_option_notes(notes, sizeof(notes), opts, "- ")) {
        return false;
    }
    n = snprintf(out,
                 out_size,
                 "# %s\n"
                 "\n"
                 "Generated op capsule scaffold. Before implementing Metal hot paths, read\n"
                 "[Metal performance tips](../../../docs/guides/metal_tips.md).\n"
                 "\n"
                 "Scaffold notes:\n"
                 "\n"
                 "%s"
                 "\n"
                 "Checklist:\n"
                 "\n"
                 "- [ ] Public API generated in `include/gradients/ops_generated.h`\n"
                 "- [ ] Forward validation/allocation/recording in `core_%s.c`\n"
                 "- [ ] Backend dispatch in `metal_%s.m`\n"
                 "- [ ] Op-local Metal ABI/kernel implementation in `metal_%s_types.h` / `metal_%s.metal`\n"
                 "- [ ] Backward rule in `autograd_%s.c`\n"
                 "- [ ] Forward PyTorch harness in `fwd.py`\n"
                 "- [ ] Backward PyTorch harness in `bwd.py`\n"
                 "- [ ] C tests under `tests/`\n"
                 "- [ ] Perf probe or benchmark for hot shapes\n",
                 opts->op,
                 notes,
                 opts->op,
                 opts->op,
                 opts->op,
                 opts->op,
                 opts->op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_sibling_tool_path(const char *argv0, const char *tool, char *out, size_t out_size)
{
    const char *slash;
    size_t dir_len;
    int n;
    if (argv0 == NULL || tool == NULL || out == NULL || out_size == 0U) {
        return false;
    }
    slash = strrchr(argv0, '/');
    if (slash == NULL) {
        n = snprintf(out, out_size, "%s", tool);
        return n >= 0 && (size_t)n < out_size;
    }
    dir_len = (size_t)(slash - argv0);
    if (dir_len + 1U + strlen(tool) + 1U > out_size) {
        return false;
    }
    memcpy(out, argv0, dir_len);
    out[dir_len] = '/';
    snprintf(out + dir_len + 1U, out_size - dir_len - 1U, "%s", tool);
    return true;
}

static bool gd_registry_stamp_path(const char *argv0, char *out, size_t out_size)
{
    const char *last_slash;
    const char *prev_slash;
    size_t build_len;
    int n;
    if (argv0 == NULL || out == NULL || out_size == 0U) {
        return false;
    }
    last_slash = strrchr(argv0, '/');
    if (last_slash == NULL) {
        n = snprintf(out, out_size, "build/.ops-registry");
        return n >= 0 && (size_t)n < out_size;
    }
    prev_slash = last_slash;
    while (prev_slash > argv0 && prev_slash[-1] != '/') {
        prev_slash -= 1;
    }
    if (prev_slash == argv0) {
        n = snprintf(out, out_size, "build/.ops-registry");
        return n >= 0 && (size_t)n < out_size;
    }
    build_len = (size_t)(prev_slash - argv0 - 1);
    if (build_len + strlen("/.ops-registry") + 1U > out_size) {
        return false;
    }
    memcpy(out, argv0, build_len);
    snprintf(out + build_len, out_size - build_len, "/.ops-registry");
    return true;
}

static int gd_run_gen_ops(const char *argv0)
{
    char gen_path[GD_NEW_OP_PATH_MAX];
    char stamp_path[GD_NEW_OP_PATH_MAX];
    char cmd[(GD_NEW_OP_PATH_MAX * 2U) + 32U];
    int n;
    int rc;
    if (!gd_sibling_tool_path(argv0, "gen_ops", gen_path, sizeof(gen_path)) ||
        !gd_registry_stamp_path(argv0, stamp_path, sizeof(stamp_path))) {
        fprintf(stderr, "gradients-new-op: failed to resolve registry tool paths\n");
        return 1;
    }
    if (!gd_path_exists(gen_path)) {
        fprintf(stderr, "gradients-new-op: missing %s; run `make tools` first\n", gen_path);
        return 1;
    }
    n = snprintf(cmd, sizeof(cmd), "\"%s\" --stamp \"%s\"", gen_path, stamp_path);
    if (n < 0 || (size_t)n >= sizeof(cmd)) {
        fprintf(stderr, "gradients-new-op: gen_ops path too long\n");
        return 1;
    }
    fflush(stdout);
    rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "gradients-new-op: registry generation failed\n");
        return 1;
    }
    return 0;
}

static void gd_print_usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [options] <snake_case_op_name>\n"
            "\n"
            "Options:\n"
            "  --binary          scaffold binary public API/backend shape (default unary)\n"
            "  --unary           scaffold unary public API/backend shape\n"
            "  --no-backend      omit generated backend= metadata for custom backend entry points\n"
            "  --custom          alias for --no-backend\n"
            "  --f16-only        annotate/scaffold for F16-specialized hot kernels\n"
            "  --f16-f32-accum   annotate/scaffold for F16 inputs with FP32 accumulation\n"
            "  --save-stats      annotate/scaffold for saved forward stats used by backward\n"
            "  --reduction       annotate/scaffold for shape-adaptive reduction kernels\n"
            "  --help            show this help\n",
            argv0);
}

static int gd_parse_args(int argc, char **argv, gd_new_op_options *opts)
{
    int i;
    if (opts == NULL) {
        return 2;
    }
    memset(opts, 0, sizeof(*opts));
    for (i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            return 3;
        }
        if (strcmp(arg, "--binary") == 0) {
            opts->binary = true;
            continue;
        }
        if (strcmp(arg, "--unary") == 0) {
            opts->binary = false;
            continue;
        }
        if (strcmp(arg, "--no-backend") == 0 || strcmp(arg, "--custom") == 0) {
            opts->no_backend = true;
            continue;
        }
        if (strcmp(arg, "--f16-only") == 0) {
            opts->f16_only = true;
            continue;
        }
        if (strcmp(arg, "--f16-f32-accum") == 0) {
            opts->f16_f32_accum = true;
            continue;
        }
        if (strcmp(arg, "--save-stats") == 0) {
            opts->save_stats = true;
            continue;
        }
        if (strcmp(arg, "--reduction") == 0) {
            opts->reduction = true;
            continue;
        }
        if (arg[0] == '-') {
            fprintf(stderr, "gradients-new-op: unknown option '%s'\n", arg);
            return 2;
        }
        if (opts->op != NULL) {
            fprintf(stderr, "gradients-new-op: multiple op names provided ('%s' and '%s')\n", opts->op, arg);
            return 2;
        }
        opts->op = arg;
    }
    if (opts->op == NULL) {
        fprintf(stderr, "gradients-new-op: missing op name\n");
        return 2;
    }
    if (!gd_valid_op_name(opts->op)) {
        fprintf(stderr,
                "gradients-new-op: invalid op name '%s' (use snake_case: [a-z][a-z0-9_]*, no trailing/double underscores)\n",
                opts->op);
        return 2;
    }
    return 0;
}

int main(int argc, char **argv)
{
    gd_new_op_options opts;
    int parse_rc;
    char upper[GD_NEW_OP_NAME_MAX];
    char op_dir[GD_NEW_OP_PATH_MAX];
    char def_path[GD_NEW_OP_PATH_MAX];
    char core_path[GD_NEW_OP_PATH_MAX];
    char autograd_path[GD_NEW_OP_PATH_MAX];
    char metal_path[GD_NEW_OP_PATH_MAX];
    char metal_types_path[GD_NEW_OP_PATH_MAX];
    char metal_kernel_path[GD_NEW_OP_PATH_MAX];
    char fwd_path[GD_NEW_OP_PATH_MAX];
    char bwd_path[GD_NEW_OP_PATH_MAX];
    char readme_path[GD_NEW_OP_PATH_MAX];
    char content[GD_NEW_OP_CONTENT_MAX];

    parse_rc = gd_parse_args(argc, argv, &opts);
    if (parse_rc == 3) {
        gd_print_usage(argv[0]);
        return 0;
    }
    if (parse_rc != 0) {
        gd_print_usage(argv[0]);
        return parse_rc;
    }
    gd_upper_from_name(opts.op, upper, sizeof(upper));
    if (!gd_make_op_dir_path(op_dir, sizeof(op_dir), opts.op) ||
        !gd_make_op_file_path(def_path, sizeof(def_path), opts.op, "op_", ".def") ||
        !gd_make_op_file_path(core_path, sizeof(core_path), opts.op, "core_", ".c") ||
        !gd_make_op_file_path(autograd_path, sizeof(autograd_path), opts.op, "autograd_", ".c") ||
        !gd_make_op_file_path(metal_path, sizeof(metal_path), opts.op, "metal_", ".m") ||
        !gd_make_op_file_path(metal_types_path, sizeof(metal_types_path), opts.op, "metal_", "_types.h") ||
        !gd_make_op_file_path(metal_kernel_path, sizeof(metal_kernel_path), opts.op, "metal_", ".metal") ||
        !gd_make_op_named_path(fwd_path, sizeof(fwd_path), opts.op, "fwd.py") ||
        !gd_make_op_named_path(bwd_path, sizeof(bwd_path), opts.op, "bwd.py") ||
        !gd_make_op_named_path(readme_path, sizeof(readme_path), opts.op, "README.md")) {
        fprintf(stderr, "gradients-new-op: generated path too long\n");
        return 1;
    }
    if (gd_mkdir_if_missing("src/ops") != 0 || gd_mkdir_if_missing(op_dir) != 0) {
        return 1;
    }
    if (!gd_make_def_content(content, sizeof(content), &opts) ||
        gd_write_new_file(def_path, content) != 0) {
        return 1;
    }
    if (!gd_make_core_content(content, sizeof(content), &opts, upper) ||
        gd_write_new_file(core_path, content) != 0) {
        return 1;
    }
    if (!gd_make_autograd_content(content, sizeof(content), &opts, upper) ||
        gd_write_new_file(autograd_path, content) != 0) {
        return 1;
    }
    if (!gd_make_metal_content(content, sizeof(content), &opts, upper) ||
        gd_write_new_file(metal_path, content) != 0) {
        return 1;
    }
    if (!gd_make_metal_types_content(content, sizeof(content), &opts, upper) ||
        gd_write_new_file(metal_types_path, content) != 0) {
        return 1;
    }
    if (!gd_make_metal_kernel_content(content, sizeof(content), &opts) ||
        gd_write_new_file(metal_kernel_path, content) != 0) {
        return 1;
    }
    if (!gd_make_fwd_py_content(content, sizeof(content), opts.op) ||
        gd_write_new_file(fwd_path, content) != 0) {
        return 1;
    }
    if (!gd_make_bwd_py_content(content, sizeof(content), opts.op) ||
        gd_write_new_file(bwd_path, content) != 0) {
        return 1;
    }
    if (!gd_make_readme_content(content, sizeof(content), &opts) ||
        gd_write_new_file(readme_path, content) != 0) {
        return 1;
    }
    if (gd_run_gen_ops(argv[0]) != 0) {
        return 1;
    }
    printf("[done] registered op '%s' as GD_OP_%s\n", opts.op, upper);
    return 0;
}
