#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define GD_GEN_MAX_OPS 512U
#define GD_GEN_NAME_MAX 96U
#define GD_GEN_PATH_MAX 4096U

#define GD_OPS_DIR "src/ops"
#define GD_OP_KIND_PATH "src/ops/op_kind.h"
#define GD_OP_REGISTRY_PATH "src/ops/op_registry.c"
#define GD_OPS_GENERATED_PATH "include/gradients/ops_generated.h"
#define GD_BACKEND_GENERATED_PATH "src/core/backend_generated.h"
#define GD_NULL_BACKEND_GENERATED_PATH "src/backends/null/backend_generated.c"
#define GD_METAL_OPS_GENERATED_PATH "src/backends/metal/metal_ops_generated.inc"

typedef struct gd_gen_op {
    char name[GD_GEN_NAME_MAX];
    char upper[GD_GEN_NAME_MAX];
    bool has_core;
    bool has_autograd;
    bool api_unary;
    bool api_binary;
    bool api_dropout;
    bool backend_unary;
    bool backend_binary;
    uint32_t old_id;
    uint32_t new_id;
} gd_gen_op;

typedef struct gd_gen_ops {
    gd_gen_op items[GD_GEN_MAX_OPS];
    uint32_t count;
} gd_gen_ops;

static uint32_t gd_generated_changed_count;

static bool gd_path_join(char *out, size_t out_size, const char *a, const char *b)
{
    int n;
    if (out == NULL || a == NULL || b == NULL || out_size == 0U) {
        return false;
    }
    n = snprintf(out, out_size, "%s/%s", a, b);
    return n >= 0 && (size_t)n < out_size;
}

static bool gd_file_exists(const char *path)
{
    struct stat st;
    return path != NULL && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static bool gd_dir_exists(const char *path)
{
    struct stat st;
    return path != NULL && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static bool gd_snprintf_ok(char *out, size_t out_size, const char *fmt, const char *a, const char *b)
{
    int n;
    if (out == NULL || fmt == NULL || out_size == 0U) {
        return false;
    }
    n = snprintf(out, out_size, fmt, a, b);
    return n >= 0 && (size_t)n < out_size;
}

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
    return i < GD_GEN_NAME_MAX;
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

static void gd_lower_from_upper(const char *upper, char *out, size_t out_size)
{
    size_t i;
    if (out == NULL || out_size == 0U) {
        return;
    }
    for (i = 0U; upper != NULL && upper[i] != '\0' && i + 1U < out_size; ++i) {
        out[i] = (char)tolower((unsigned char)upper[i]);
    }
    out[i] = '\0';
}

static gd_gen_op *gd_find_op(gd_gen_ops *ops, const char *name)
{
    uint32_t i;
    if (ops == NULL || name == NULL) {
        return NULL;
    }
    for (i = 0U; i < ops->count; ++i) {
        if (strcmp(ops->items[i].name, name) == 0) {
            return &ops->items[i];
        }
    }
    return NULL;
}

static gd_gen_op *gd_add_op(gd_gen_ops *ops, const char *name)
{
    gd_gen_op *op;
    if (ops == NULL || name == NULL || ops->count >= GD_GEN_MAX_OPS) {
        return NULL;
    }
    op = &ops->items[ops->count];
    memset(op, 0, sizeof(*op));
    snprintf(op->name, sizeof(op->name), "%s", name);
    gd_upper_from_name(name, op->upper, sizeof(op->upper));
    ops->count += 1U;
    return op;
}

static char *gd_trim(char *s)
{
    char *end;
    if (s == NULL) {
        return NULL;
    }
    while (isspace((unsigned char)*s)) {
        s += 1;
    }
    end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) {
        end -= 1;
    }
    *end = '\0';
    return s;
}

static void gd_parse_op_def(gd_gen_op *op, const char *path)
{
    FILE *f;
    char line[256];
    if (op == NULL || path == NULL) {
        return;
    }
    f = fopen(path, "r");
    if (f == NULL) {
        return;
    }
    while (fgets(line, sizeof(line), f) != NULL) {
        char *eq;
        char *key;
        char *value;
        char *comment = strchr(line, '#');
        if (comment != NULL) {
            *comment = '\0';
        }
        eq = strchr(line, '=');
        if (eq == NULL) {
            continue;
        }
        *eq = '\0';
        key = gd_trim(line);
        value = gd_trim(eq + 1);
        if (strcmp(key, "api") == 0) {
            if (strcmp(value, "unary") == 0) {
                op->api_unary = true;
            } else if (strcmp(value, "binary") == 0) {
                op->api_binary = true;
            } else if (strcmp(value, "dropout") == 0) {
                op->api_dropout = true;
            }
        } else if (strcmp(key, "backend") == 0) {
            if (strcmp(value, "unary") == 0) {
                op->backend_unary = true;
            } else if (strcmp(value, "binary") == 0) {
                op->backend_binary = true;
            }
        }
    }
    fclose(f);
}

static int gd_scan_ops(gd_gen_ops *ops)
{
    DIR *dir;
    struct dirent *entry;
    if (ops == NULL) {
        return 1;
    }
    dir = opendir(GD_OPS_DIR);
    if (dir == NULL) {
        fprintf(stderr, "gen_ops: failed to open %s: %s\n", GD_OPS_DIR, strerror(errno));
        return 1;
    }
    while ((entry = readdir(dir)) != NULL) {
        char op_dir[GD_GEN_PATH_MAX];
        char path[GD_GEN_PATH_MAX];
        gd_gen_op *op;
        if (entry->d_name[0] == '.') {
            continue;
        }
        if (!gd_valid_op_name(entry->d_name)) {
            continue;
        }
        if (!gd_path_join(op_dir, sizeof(op_dir), GD_OPS_DIR, entry->d_name) ||
            !gd_dir_exists(op_dir)) {
            continue;
        }
        op = gd_add_op(ops, entry->d_name);
        if (op == NULL) {
            closedir(dir);
            fprintf(stderr, "gen_ops: too many ops\n");
            return 1;
        }
        if (!gd_snprintf_ok(path, sizeof(path), "%s/core_%s.c", op_dir, op->name)) {
            closedir(dir);
            fprintf(stderr, "gen_ops: path too long for %s\n", op->name);
            return 1;
        }
        op->has_core = gd_file_exists(path);
        if (!gd_snprintf_ok(path, sizeof(path), "%s/autograd_%s.c", op_dir, op->name)) {
            closedir(dir);
            fprintf(stderr, "gen_ops: path too long for %s\n", op->name);
            return 1;
        }
        op->has_autograd = gd_file_exists(path);
        if (!gd_snprintf_ok(path, sizeof(path), "%s/op_%s.def", op_dir, op->name)) {
            closedir(dir);
            fprintf(stderr, "gen_ops: path too long for %s\n", op->name);
            return 1;
        }
        gd_parse_op_def(op, path);
        if (!op->has_core && !op->has_autograd) {
            ops->count -= 1U;
        }
    }
    closedir(dir);
    return 0;
}

static void gd_apply_existing_order(gd_gen_ops *ops)
{
    FILE *f;
    char line[512];
    if (ops == NULL) {
        return;
    }
    f = fopen(GD_OP_KIND_PATH, "r");
    if (f == NULL) {
        return;
    }
    while (fgets(line, sizeof(line), f) != NULL) {
        char *p = strstr(line, "GD_OP_");
        while (p != NULL) {
            char token[GD_GEN_NAME_MAX];
            char lower[GD_GEN_NAME_MAX];
            size_t len = 0U;
            char *q = p + strlen("GD_OP_");
            while ((isupper((unsigned char)q[len]) || isdigit((unsigned char)q[len]) || q[len] == '_') &&
                   len + 1U < sizeof(token)) {
                len += 1U;
            }
            if (len > 0U) {
                uint32_t value = 0U;
                char *eq;
                memcpy(token, q, len);
                token[len] = '\0';
                gd_lower_from_upper(token, lower, sizeof(lower));
                if (strcmp(lower, "invalid") != 0 && strcmp(lower, "count") != 0) {
                    gd_gen_op *op = gd_find_op(ops, lower);
                    if (op != NULL) {
                        eq = strchr(q + len, '=');
                        if (eq != NULL) {
                            unsigned long parsed = strtoul(eq + 1, NULL, 10);
                            if (parsed > 0UL && parsed <= UINT32_MAX) {
                                value = (uint32_t)parsed;
                            }
                        }
                        if (value != 0U) {
                            op->old_id = value;
                        }
                    }
                }
            }
            p = strstr(q + len, "GD_OP_");
        }
    }
    fclose(f);
}

static int gd_compare_ops(const void *a, const void *b)
{
    const gd_gen_op *oa = (const gd_gen_op *)a;
    const gd_gen_op *ob = (const gd_gen_op *)b;
    if (oa->old_id != 0U && ob->old_id != 0U) {
        if (oa->old_id < ob->old_id) {
            return -1;
        }
        if (oa->old_id > ob->old_id) {
            return 1;
        }
        return strcmp(oa->name, ob->name);
    }
    if (oa->old_id != 0U) {
        return -1;
    }
    if (ob->old_id != 0U) {
        return 1;
    }
    return strcmp(oa->name, ob->name);
}

static void gd_assign_new_ids(gd_gen_ops *ops)
{
    uint32_t i;
    qsort(ops->items, ops->count, sizeof(ops->items[0]), gd_compare_ops);
    for (i = 0U; i < ops->count; ++i) {
        ops->items[i].new_id = i + 1U;
    }
}

static bool gd_append(char **buf, size_t *len, size_t *cap, const char *text)
{
    size_t need;
    size_t text_len;
    char *grown;
    if (buf == NULL || len == NULL || cap == NULL || text == NULL) {
        return false;
    }
    text_len = strlen(text);
    if (*len > SIZE_MAX - text_len - 1U) {
        return false;
    }
    need = *len + text_len + 1U;
    if (need > *cap) {
        size_t new_cap = *cap == 0U ? 4096U : *cap;
        while (new_cap < need) {
            if (new_cap > SIZE_MAX / 2U) {
                return false;
            }
            new_cap *= 2U;
        }
        grown = (char *)realloc(*buf, new_cap);
        if (grown == NULL) {
            return false;
        }
        *buf = grown;
        *cap = new_cap;
    }
    memcpy(*buf + *len, text, text_len + 1U);
    *len += text_len;
    return true;
}

static bool gd_appendf(char **buf, size_t *len, size_t *cap, const char *fmt, const char *a, uint32_t b)
{
    char tmp[512];
    int n = snprintf(tmp, sizeof(tmp), fmt, a, b);
    if (n < 0 || (size_t)n >= sizeof(tmp)) {
        return false;
    }
    return gd_append(buf, len, cap, tmp);
}

static bool gd_read_file(const char *path, char **out, size_t *out_len)
{
    FILE *f;
    long size;
    char *buf;
    size_t nread;
    if (out == NULL || out_len == NULL) {
        return false;
    }
    *out = NULL;
    *out_len = 0U;
    f = fopen(path, "rb");
    if (f == NULL) {
        return false;
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        fclose(f);
        return false;
    }
    size = ftell(f);
    if (size < 0L) {
        fclose(f);
        return false;
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        fclose(f);
        return false;
    }
    buf = (char *)malloc((size_t)size + 1U);
    if (buf == NULL) {
        fclose(f);
        return false;
    }
    nread = fread(buf, 1U, (size_t)size, f);
    fclose(f);
    if (nread != (size_t)size) {
        free(buf);
        return false;
    }
    buf[nread] = '\0';
    *out = buf;
    *out_len = nread;
    return true;
}

static int gd_write_if_changed(const char *path, const char *content, size_t len)
{
    char *old = NULL;
    size_t old_len = 0U;
    FILE *f;
    bool same = false;
    if (gd_read_file(path, &old, &old_len)) {
        same = old_len == len && memcmp(old, content, len) == 0;
    }
    free(old);
    if (same) {
        printf("[ops-registry] unchanged %s\n", path);
        return 0;
    }
    f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "gen_ops: failed to write %s: %s\n", path, strerror(errno));
        return 1;
    }
    if (fwrite(content, 1U, len, f) != len) {
        fprintf(stderr, "gen_ops: short write %s\n", path);
        fclose(f);
        return 1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "gen_ops: failed to close %s: %s\n", path, strerror(errno));
        return 1;
    }
    gd_generated_changed_count += 1U;
    printf("[ops-registry] generated %s\n", path);
    return 0;
}

static int gd_touch_file(const char *path)
{
    FILE *f;
    if (path == NULL) {
        return 1;
    }
    f = fopen(path, "ab");
    if (f == NULL) {
        fprintf(stderr, "gen_ops: failed to touch %s: %s\n", path, strerror(errno));
        return 1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "gen_ops: failed to close %s: %s\n", path, strerror(errno));
        return 1;
    }
    return 0;
}

static int gd_generate_op_kind(const gd_gen_ops *ops)
{
    char *buf = NULL;
    size_t len = 0U;
    size_t cap = 0U;
    uint32_t i;
    bool ok = true;
    ok = ok && gd_append(&buf, &len, &cap,
                         "#ifndef GD_OP_KIND_H\n"
                         "#define GD_OP_KIND_H\n\n"
                         "/* @generated by tools/gen_ops.c; do not edit. */\n"
                         "typedef enum gd_op_kind {\n"
                         "    GD_OP_INVALID = 0,\n");
    for (i = 0U; ok && i < ops->count; ++i) {
        ok = gd_appendf(&buf, &len, &cap, "    GD_OP_%s = %u,\n", ops->items[i].upper, ops->items[i].new_id);
    }
    if (ok) {
        char tmp[128];
        int n = snprintf(tmp, sizeof(tmp), "    GD_OP_COUNT = %u,\n", ops->count + 1U);
        ok = n >= 0 && (size_t)n < sizeof(tmp) && gd_append(&buf, &len, &cap, tmp);
    }
    ok = ok && gd_append(&buf, &len, &cap, "} gd_op_kind;\n\n#endif /* GD_OP_KIND_H */\n");
    if (!ok) {
        free(buf);
        fprintf(stderr, "gen_ops: failed to render op_kind.h\n");
        return 1;
    }
    i = (uint32_t)gd_write_if_changed(GD_OP_KIND_PATH, buf, len);
    free(buf);
    return (int)i;
}

static int gd_generate_public_ops(const gd_gen_ops *ops)
{
    char *buf = NULL;
    size_t len = 0U;
    size_t cap = 0U;
    uint32_t i;
    bool ok = true;
    ok = ok && gd_append(&buf, &len, &cap,
                         "#ifndef GRADIENTS_OPS_GENERATED_H\n"
                         "#define GRADIENTS_OPS_GENERATED_H\n\n"
                         "/* @generated by tools/gen_ops.c; do not edit. */\n\n"
                         "#include <gradients/status.h>\n"
                         "#include <gradients/tensor.h>\n\n"
                         "#ifdef __cplusplus\n"
                         "extern \"C\" {\n"
                         "#endif\n\n"
                         "/* Direct backward helpers accept NULL grad_* output pointers to omit gradients.\n"
                         " * If every grad_* output is NULL, helpers validate inputs/grad_out and return\n"
                         " * GD_OK without enqueueing backward work. Requesting gradients for\n"
                         " * non-differentiable inputs still returns GD_ERR_UNSUPPORTED. */\n\n");
    for (i = 0U; ok && i < ops->count; ++i) {
        if (ops->items[i].api_unary) {
            char tmp[1024];
            int n = snprintf(tmp,
                             sizeof(tmp),
                             "gd_status gd_%s(gd_context *ctx,\n"
                             "                  const gd_tensor *x,\n"
                             "                  gd_tensor *out);\n\n"
                             "gd_status gd_%s_backward(gd_context *ctx,\n"
                             "                           const gd_tensor *x,\n"
                             "                           const gd_tensor *grad_out,\n"
                             "                           gd_tensor *grad_x);\n\n",
                             ops->items[i].name,
                             ops->items[i].name);
            ok = n >= 0 && (size_t)n < sizeof(tmp) && gd_append(&buf, &len, &cap, tmp);
        }
        if (ok && ops->items[i].api_binary) {
            char tmp[1400];
            int n = snprintf(tmp,
                             sizeof(tmp),
                             "gd_status gd_%s(gd_context *ctx,\n"
                             "                  const gd_tensor *x,\n"
                             "                  const gd_tensor *y,\n"
                             "                  gd_tensor *out);\n\n"
                             "gd_status gd_%s_backward(gd_context *ctx,\n"
                             "                           const gd_tensor *x,\n"
                             "                           const gd_tensor *y,\n"
                             "                           const gd_tensor *grad_out,\n"
                             "                           gd_tensor *grad_x,\n"
                             "                           gd_tensor *grad_y);\n\n",
                             ops->items[i].name,
                             ops->items[i].name);
            ok = n >= 0 && (size_t)n < sizeof(tmp) && gd_append(&buf, &len, &cap, tmp);
        }
        if (ok && ops->items[i].api_dropout) {
            char tmp[1600];
            int n = snprintf(tmp,
                             sizeof(tmp),
                             "gd_status gd_%s(gd_context *ctx,\n"
                             "                  const gd_tensor *x,\n"
                             "                  float p,\n"
                             "                  bool training,\n"
                             "                  uint64_t seed,\n"
                             "                  gd_tensor *out);\n\n"
                             "gd_status gd_%s_backward(gd_context *ctx,\n"
                             "                           const gd_tensor *x,\n"
                             "                           const gd_tensor *grad_out,\n"
                             "                           float p,\n"
                             "                           uint64_t seed,\n"
                             "                           gd_tensor *grad_x);\n\n",
                             ops->items[i].name,
                             ops->items[i].name);
            ok = n >= 0 && (size_t)n < sizeof(tmp) && gd_append(&buf, &len, &cap, tmp);
        }
    }
    ok = ok && gd_append(&buf, &len, &cap,
                         "#ifdef __cplusplus\n"
                         "}\n"
                         "#endif\n\n"
                         "#endif /* GRADIENTS_OPS_GENERATED_H */\n");
    if (!ok) {
        free(buf);
        fprintf(stderr, "gen_ops: failed to render ops_generated.h\n");
        return 1;
    }
    i = (uint32_t)gd_write_if_changed(GD_OPS_GENERATED_PATH, buf, len);
    free(buf);
    return (int)i;
}

static int gd_generate_backend_header(const gd_gen_ops *ops)
{
    char *buf = NULL;
    size_t len = 0U;
    size_t cap = 0U;
    uint32_t i;
    bool ok = true;
    ok = ok && gd_append(&buf, &len, &cap,
                         "#ifndef GD_CORE_BACKEND_GENERATED_H\n"
                         "#define GD_CORE_BACKEND_GENERATED_H\n\n"
                         "/* @generated by tools/gen_ops.c; do not edit. */\n\n");
    for (i = 0U; ok && i < ops->count; ++i) {
        if (ops->items[i].backend_unary) {
            char tmp[1024];
            int n = snprintf(tmp,
                             sizeof(tmp),
                             "gd_status gd_backend_%s(gd_backend *backend,\n"
                             "                          const gd_backend_tensor_view *x,\n"
                             "                          const gd_backend_tensor_view *y);\n"
                             "gd_status gd_backend_%s_backward(gd_backend *backend,\n"
                             "                                   const gd_backend_tensor_view *x,\n"
                             "                                   const gd_backend_tensor_view *grad_out,\n"
                             "                                   const gd_backend_tensor_view *grad_x);\n\n",
                             ops->items[i].name,
                             ops->items[i].name);
            ok = n >= 0 && (size_t)n < sizeof(tmp) && gd_append(&buf, &len, &cap, tmp);
        }
        if (ok && ops->items[i].backend_binary) {
            char tmp[1024];
            int n = snprintf(tmp,
                             sizeof(tmp),
                             "gd_status gd_backend_%s(gd_backend *backend,\n"
                             "                          const gd_backend_tensor_view *x,\n"
                             "                          const gd_backend_tensor_view *y,\n"
                             "                          const gd_backend_tensor_view *out);\n\n",
                             ops->items[i].name);
            ok = n >= 0 && (size_t)n < sizeof(tmp) && gd_append(&buf, &len, &cap, tmp);
        }
    }
    ok = ok && gd_append(&buf, &len, &cap, "#endif /* GD_CORE_BACKEND_GENERATED_H */\n");
    if (!ok) {
        free(buf);
        fprintf(stderr, "gen_ops: failed to render backend_generated.h\n");
        return 1;
    }
    i = (uint32_t)gd_write_if_changed(GD_BACKEND_GENERATED_PATH, buf, len);
    free(buf);
    return (int)i;
}

static int gd_generate_null_backend(const gd_gen_ops *ops)
{
    char *buf = NULL;
    size_t len = 0U;
    size_t cap = 0U;
    uint32_t i;
    bool ok = true;
    ok = ok && gd_append(&buf, &len, &cap,
                         "#include \"../../core/backend.h\"\n\n"
                         "/* @generated by tools/gen_ops.c; do not edit. */\n\n");
    for (i = 0U; ok && i < ops->count; ++i) {
        if (ops->items[i].backend_unary) {
            char tmp[1600];
            int n = snprintf(tmp,
                             sizeof(tmp),
                             "gd_status gd_backend_%s(gd_backend *backend,\n"
                             "                          const gd_backend_tensor_view *x,\n"
                             "                          const gd_backend_tensor_view *y)\n"
                             "{\n"
                             "    (void)backend;\n"
                             "    (void)x;\n"
                             "    (void)y;\n"
                             "    return GD_ERR_UNSUPPORTED;\n"
                             "}\n\n"
                             "gd_status gd_backend_%s_backward(gd_backend *backend,\n"
                             "                                   const gd_backend_tensor_view *x,\n"
                             "                                   const gd_backend_tensor_view *grad_out,\n"
                             "                                   const gd_backend_tensor_view *grad_x)\n"
                             "{\n"
                             "    (void)backend;\n"
                             "    (void)x;\n"
                             "    (void)grad_out;\n"
                             "    (void)grad_x;\n"
                             "    return GD_ERR_UNSUPPORTED;\n"
                             "}\n\n",
                             ops->items[i].name,
                             ops->items[i].name);
            ok = n >= 0 && (size_t)n < sizeof(tmp) && gd_append(&buf, &len, &cap, tmp);
        }
        if (ok && ops->items[i].backend_binary) {
            char tmp[1200];
            int n = snprintf(tmp,
                             sizeof(tmp),
                             "gd_status gd_backend_%s(gd_backend *backend,\n"
                             "                          const gd_backend_tensor_view *x,\n"
                             "                          const gd_backend_tensor_view *y,\n"
                             "                          const gd_backend_tensor_view *out)\n"
                             "{\n"
                             "    (void)backend;\n"
                             "    (void)x;\n"
                             "    (void)y;\n"
                             "    (void)out;\n"
                             "    return GD_ERR_UNSUPPORTED;\n"
                             "}\n\n",
                             ops->items[i].name);
            ok = n >= 0 && (size_t)n < sizeof(tmp) && gd_append(&buf, &len, &cap, tmp);
        }
    }
    if (!ok) {
        free(buf);
        fprintf(stderr, "gen_ops: failed to render null backend stubs\n");
        return 1;
    }
    i = (uint32_t)gd_write_if_changed(GD_NULL_BACKEND_GENERATED_PATH, buf, len);
    free(buf);
    return (int)i;
}

static int gd_generate_metal_ops(const gd_gen_ops *ops)
{
    char *buf = NULL;
    size_t len = 0U;
    size_t cap = 0U;
    uint32_t i;
    bool ok = true;
    ok = ok && gd_append(&buf, &len, &cap,
                         "/* @generated by tools/gen_ops.c; do not edit.\n"
                         "   Included inside gd_metal_pipeline_specs[]. */\n");
    for (i = 0U; ok && i < ops->count; ++i) {
        if (ops->items[i].backend_unary) {
            char tmp[512];
            int n = snprintf(tmp,
                             sizeof(tmp),
                             "    GD_METAL_PIPELINE_INDEX(\"gd_%s_kernel\", unary_pso, GD_OP_%s),\n"
                             "    GD_METAL_PIPELINE_INDEX(\"gd_%s_backward_kernel\", "
                             "unary_backward_pso, GD_OP_%s),\n",
                             ops->items[i].name,
                             ops->items[i].upper,
                             ops->items[i].name,
                             ops->items[i].upper);
            ok = n >= 0 && (size_t)n < sizeof(tmp) && gd_append(&buf, &len, &cap, tmp);
        }
        if (ok && ops->items[i].backend_binary) {
            char tmp[768];
            int n = snprintf(tmp,
                             sizeof(tmp),
                             "    GD_METAL_PIPELINE_INDEX(\"gd_%s_kernel\", binary_pso, GD_OP_%s),\n"
                             "    GD_METAL_PIPELINE_INDEX(\"gd_%s_bcast_kernel\", "
                             "binary_bcast_pso, GD_OP_%s),\n"
                             "    GD_METAL_PIPELINE_INDEX(\"gd_%s_row_bcast_kernel\", "
                             "binary_row_bcast_pso, GD_OP_%s),\n",
                             ops->items[i].name,
                             ops->items[i].upper,
                             ops->items[i].name,
                             ops->items[i].upper,
                             ops->items[i].name,
                             ops->items[i].upper);
            ok = n >= 0 && (size_t)n < sizeof(tmp) && gd_append(&buf, &len, &cap, tmp);
        }
    }
    if (!ok) {
        free(buf);
        fprintf(stderr, "gen_ops: failed to render metal op pipeline table\n");
        return 1;
    }
    i = (uint32_t)gd_write_if_changed(GD_METAL_OPS_GENERATED_PATH, buf, len);
    free(buf);
    return (int)i;
}

static int gd_generate_registry(const gd_gen_ops *ops)
{
    char *buf = NULL;
    size_t len = 0U;
    size_t cap = 0U;
    uint32_t i;
    bool ok = true;
    ok = ok && gd_append(&buf, &len, &cap,
                         "#include \"autograd_impl.h\"\n\n"
                         "/* @generated by tools/gen_ops.c; do not edit. */\n");
    for (i = 0U; ok && i < ops->count; ++i) {
        if (ops->items[i].has_autograd) {
            ok = gd_appendf(&buf, &len, &cap,
                            "extern const gd_autograd_rule gd_bwd_rule_%s;\n",
                            ops->items[i].name,
                            0U);
        }
    }
    ok = ok && gd_append(&buf, &len, &cap,
                         "\nstatic const gd_autograd_rule *const gd_bwd_rules[GD_OP_COUNT] = {\n");
    for (i = 0U; ok && i < ops->count; ++i) {
        if (ops->items[i].has_autograd) {
            char tmp[512];
            int n = snprintf(tmp, sizeof(tmp),
                             "    [GD_OP_%s] = &gd_bwd_rule_%s,\n",
                             ops->items[i].upper,
                             ops->items[i].name);
            ok = n >= 0 && (size_t)n < sizeof(tmp) && gd_append(&buf, &len, &cap, tmp);
        }
    }
    ok = ok && gd_append(&buf, &len, &cap,
                         "};\n\n"
                         "const gd_autograd_rule *gd_autograd_rule_for(gd_op_kind kind)\n"
                         "{\n"
                         "    if (kind <= GD_OP_INVALID || kind >= GD_OP_COUNT) {\n"
                         "        return NULL;\n"
                         "    }\n"
                         "    return gd_bwd_rules[kind];\n"
                         "}\n");
    if (!ok) {
        free(buf);
        fprintf(stderr, "gen_ops: failed to render op_registry.c\n");
        return 1;
    }
    i = (uint32_t)gd_write_if_changed(GD_OP_REGISTRY_PATH, buf, len);
    free(buf);
    return (int)i;
}

int main(int argc, char **argv)
{
    const char *stamp_path = NULL;
    gd_gen_ops ops;
    if (argc == 3 && strcmp(argv[1], "--stamp") == 0) {
        stamp_path = argv[2];
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [--stamp PATH]\n", argv[0]);
        return 2;
    }
    memset(&ops, 0, sizeof(ops));
    if (gd_scan_ops(&ops) != 0) {
        return 1;
    }
    gd_apply_existing_order(&ops);
    gd_assign_new_ids(&ops);
    if (gd_generate_op_kind(&ops) != 0) {
        return 1;
    }
    if (gd_generate_public_ops(&ops) != 0) {
        return 1;
    }
    if (gd_generate_backend_header(&ops) != 0) {
        return 1;
    }
    if (gd_generate_null_backend(&ops) != 0) {
        return 1;
    }
    if (gd_generate_metal_ops(&ops) != 0) {
        return 1;
    }
    if (gd_generate_registry(&ops) != 0) {
        return 1;
    }
    if (stamp_path != NULL) {
        if (gd_generated_changed_count != 0U || !gd_file_exists(stamp_path)) {
            if (gd_touch_file(stamp_path) != 0) {
                return 1;
            }
            printf("[ops-registry] touched %s\n", stamp_path);
        } else {
            printf("[ops-registry] stamp unchanged %s\n", stamp_path);
        }
    }
    printf("[ops-registry] ops=%u\n", ops.count);
    return 0;
}
