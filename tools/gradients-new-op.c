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
#define GD_NEW_OP_CONTENT_MAX 16384U

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

static bool gd_make_def_content(char *out, size_t out_size, const char *op)
{
    int n = snprintf(out,
                     out_size,
                     "# Generated op metadata for %s.\n"
                     "# api/backend shape controls generated public/backend stubs.\n"
                     "api=unary\n"
                     "backend=unary\n",
                     op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_core_content(char *out, size_t out_size, const char *op, const char *upper)
{
    int n = snprintf(out,
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
                     "*/\n"
                     "typedef int gd_%s_core_scaffold_anchor;\n",
                     op,
                     op,
                     upper,
                     op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_autograd_content(char *out, size_t out_size, const char *op, const char *upper)
{
    int n = snprintf(out,
                     out_size,
                     "#include \"../autograd_impl.h\"\n"
                     "\n"
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
                     op,
                     op,
                     upper,
                     op,
                     op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_metal_content(char *out, size_t out_size, const char *op)
{
    int n = snprintf(out,
                     out_size,
                     "#include \"../../backends/metal/metal_backend_internal.h\"\n"
                     "#include \"metal_%s_types.h\"\n"
                     "\n"
                     "/* Scaffold for the '%s' Metal backend capsule.\n"
                     "\n"
                     "   Add backend entry points declared in src/core/backend_generated.h here,\n"
                     "   bind kernels from metal_%s.metal, and keep hot paths allocation-free.\n"
                     "*/\n"
                     "typedef int gd_%s_metal_scaffold_anchor;\n",
                     op,
                     op,
                     op,
                     op);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_metal_types_content(char *out, size_t out_size, const char *op, const char *upper)
{
    int n = snprintf(out,
                     out_size,
                     "#ifndef GD_OP_%s_METAL_TYPES_H\n"
                     "#define GD_OP_%s_METAL_TYPES_H\n"
                     "\n"
                     "/* Op-local Metal ABI types for %s. Keep host/Metal layouts in sync. */\n"
                     "\n"
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
                     op,
                     op,
                     op,
                     op,
                     op,
                     upper);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_make_metal_kernel_content(char *out, size_t out_size, const char *op)
{
    int n = snprintf(out,
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
                     op,
                     op,
                     op,
                     op,
                     op,
                     op,
                     op,
                     op);
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

static bool gd_make_readme_content(char *out, size_t out_size, const char *op)
{
    int n = snprintf(out,
                     out_size,
                     "# %s\n"
                     "\n"
                     "Generated op capsule scaffold.\n"
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
                     "- [ ] C tests under `tests/`\n",
                     op,
                     op,
                     op,
                     op,
                     op,
                     op);
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

int main(int argc, char **argv)
{
    const char *op;
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

    if (argc != 2) {
        fprintf(stderr, "usage: %s <snake_case_op_name>\n", argv[0]);
        return 2;
    }
    op = argv[1];
    if (!gd_valid_op_name(op)) {
        fprintf(stderr,
                "gradients-new-op: invalid op name '%s' (use snake_case: [a-z][a-z0-9_]*, no trailing/double underscores)\n",
                op);
        return 2;
    }
    gd_upper_from_name(op, upper, sizeof(upper));
    if (!gd_make_op_dir_path(op_dir, sizeof(op_dir), op) ||
        !gd_make_op_file_path(def_path, sizeof(def_path), op, "op_", ".def") ||
        !gd_make_op_file_path(core_path, sizeof(core_path), op, "core_", ".c") ||
        !gd_make_op_file_path(autograd_path, sizeof(autograd_path), op, "autograd_", ".c") ||
        !gd_make_op_file_path(metal_path, sizeof(metal_path), op, "metal_", ".m") ||
        !gd_make_op_file_path(metal_types_path, sizeof(metal_types_path), op, "metal_", "_types.h") ||
        !gd_make_op_file_path(metal_kernel_path, sizeof(metal_kernel_path), op, "metal_", ".metal") ||
        !gd_make_op_named_path(fwd_path, sizeof(fwd_path), op, "fwd.py") ||
        !gd_make_op_named_path(bwd_path, sizeof(bwd_path), op, "bwd.py") ||
        !gd_make_op_named_path(readme_path, sizeof(readme_path), op, "README.md")) {
        fprintf(stderr, "gradients-new-op: generated path too long\n");
        return 1;
    }
    if (gd_mkdir_if_missing("src/ops") != 0 || gd_mkdir_if_missing(op_dir) != 0) {
        return 1;
    }
    if (!gd_make_def_content(content, sizeof(content), op) ||
        gd_write_new_file(def_path, content) != 0) {
        return 1;
    }
    if (!gd_make_core_content(content, sizeof(content), op, upper) ||
        gd_write_new_file(core_path, content) != 0) {
        return 1;
    }
    if (!gd_make_autograd_content(content, sizeof(content), op, upper) ||
        gd_write_new_file(autograd_path, content) != 0) {
        return 1;
    }
    if (!gd_make_metal_content(content, sizeof(content), op) ||
        gd_write_new_file(metal_path, content) != 0) {
        return 1;
    }
    if (!gd_make_metal_types_content(content, sizeof(content), op, upper) ||
        gd_write_new_file(metal_types_path, content) != 0) {
        return 1;
    }
    if (!gd_make_metal_kernel_content(content, sizeof(content), op) ||
        gd_write_new_file(metal_kernel_path, content) != 0) {
        return 1;
    }
    if (!gd_make_fwd_py_content(content, sizeof(content), op) ||
        gd_write_new_file(fwd_path, content) != 0) {
        return 1;
    }
    if (!gd_make_bwd_py_content(content, sizeof(content), op) ||
        gd_write_new_file(bwd_path, content) != 0) {
        return 1;
    }
    if (!gd_make_readme_content(content, sizeof(content), op) ||
        gd_write_new_file(readme_path, content) != 0) {
        return 1;
    }
    if (gd_run_gen_ops(argv[0]) != 0) {
        return 1;
    }
    printf("[done] registered op '%s' as GD_OP_%s\n", op, upper);
    return 0;
}
