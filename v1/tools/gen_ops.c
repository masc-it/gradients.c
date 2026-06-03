/*
 * gradients.c operator registry generator.
 *
 * This tool intentionally reads filenames only.  Makefile passes discovered
 * capsule files; file names define op kind, registry symbols, and coverage.
 */

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum gen_category {
    GEN_NONE = 0,
    GEN_CORE,
    GEN_GRAD,
    GEN_CPU,
    GEN_METAL,
    GEN_METAL_SHADER
} gen_category;

typedef struct gen_file {
    char *path;
    char *op;
    char *role;
    char *name;
    char *kind_enum;
    char *symbol_suffix;
} gen_file;

typedef struct gen_vec {
    gen_file *items;
    size_t count;
    size_t cap;
} gen_vec;

typedef struct str_vec {
    char **items;
    size_t count;
    size_t cap;
} str_vec;

typedef struct gen_inputs {
    char *out_dir;
    gen_vec core;
    gen_vec grad;
    gen_vec cpu;
    gen_vec metal;
    gen_vec metal_shader;
} gen_inputs;

static void die_errno(const char *what, const char *path)
{
    if (path != NULL) {
        (void)fprintf(stderr, "gen_ops: %s %s: %s\n", what, path, strerror(errno));
    } else {
        (void)fprintf(stderr, "gen_ops: %s: %s\n", what, strerror(errno));
    }
    exit(1);
}

static void die_msg(const char *msg)
{
    (void)fprintf(stderr, "gen_ops: %s\n", msg);
    exit(1);
}

static void die_path(const char *msg, const char *path)
{
    (void)fprintf(stderr, "gen_ops: %s: %s\n", msg, path);
    exit(1);
}

static void *xmalloc(size_t size)
{
    void *ptr = malloc(size == 0u ? 1u : size);

    if (ptr == NULL) {
        die_errno("malloc", NULL);
    }
    return ptr;
}

static void *xrealloc(void *ptr, size_t size)
{
    void *next = realloc(ptr, size == 0u ? 1u : size);

    if (next == NULL) {
        die_errno("realloc", NULL);
    }
    return next;
}

static char *xstrdup(const char *s)
{
    size_t len = strlen(s);
    char *out = xmalloc(len + 1u);

    (void)memcpy(out, s, len + 1u);
    return out;
}

static char *copy_range(const char *s, size_t len)
{
    char *out = xmalloc(len + 1u);

    if (len > 0u) {
        (void)memcpy(out, s, len);
    }
    out[len] = '\0';
    return out;
}

static char *concat2(const char *a, const char *b)
{
    size_t alen = strlen(a);
    size_t blen = strlen(b);
    char *out = xmalloc(alen + blen + 1u);

    (void)memcpy(out, a, alen);
    (void)memcpy(out + alen, b, blen + 1u);
    return out;
}

static char *concat3(const char *a, const char *b, const char *c)
{
    size_t alen = strlen(a);
    size_t blen = strlen(b);
    size_t clen = strlen(c);
    char *out = xmalloc(alen + blen + clen + 1u);

    (void)memcpy(out, a, alen);
    (void)memcpy(out + alen, b, blen);
    (void)memcpy(out + alen + blen, c, clen + 1u);
    return out;
}

static bool starts_with(const char *s, const char *prefix)
{
    size_t n = strlen(prefix);

    return strncmp(s, prefix, n) == 0;
}

static bool ends_with(const char *s, const char *suffix)
{
    size_t slen = strlen(s);
    size_t tlen = strlen(suffix);

    if (slen < tlen) {
        return false;
    }
    return strcmp(s + slen - tlen, suffix) == 0;
}

static const char *path_basename(const char *path)
{
    const char *slash = strrchr(path, '/');

    return slash == NULL ? path : slash + 1;
}

static char *path_parent_basename(const char *path)
{
    const char *slash = strrchr(path, '/');
    const char *prev = NULL;

    if (slash == NULL || slash == path) {
        return NULL;
    }

    prev = slash - 1;
    while (prev > path && *prev != '/') {
        prev--;
    }
    if (*prev == '/') {
        prev++;
    }
    if (prev >= slash) {
        return NULL;
    }
    return copy_range(prev, (size_t)(slash - prev));
}

static bool is_snake_name(const char *s)
{
    size_t i;
    size_t len = strlen(s);
    bool prev_underscore = false;

    if (len == 0u || s[0] == '_' || s[len - 1u] == '_') {
        return false;
    }
    for (i = 0u; i < len; i++) {
        char c = s[i];

        if (c == '_') {
            if (prev_underscore) {
                return false;
            }
            prev_underscore = true;
            continue;
        }
        prev_underscore = false;
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) {
            return false;
        }
    }
    return true;
}

static char upper_ascii(char c)
{
    if (c >= 'a' && c <= 'z') {
        return (char)(c - ('a' - 'A'));
    }
    return c;
}

static char *snake_to_upper(const char *s)
{
    size_t i;
    size_t len = strlen(s);
    char *out = xmalloc(len + 1u);

    for (i = 0u; i < len; i++) {
        out[i] = upper_ascii(s[i]);
    }
    out[len] = '\0';
    return out;
}

static bool role_is_fwd(const char *role)
{
    return strcmp(role, "fwd") == 0;
}

static char *kind_name_from_parts(const char *op, const char *role)
{
    if (role == NULL || role_is_fwd(role)) {
        return xstrdup(op);
    }
    return concat3(op, "_", role);
}

static char *kind_enum_from_name(const char *name)
{
    char *upper = snake_to_upper(name);
    char *out = concat2("_GD_OP_", upper);

    free(upper);
    return out;
}

static void gen_file_clear(gen_file *file)
{
    free(file->path);
    free(file->op);
    free(file->role);
    free(file->name);
    free(file->kind_enum);
    free(file->symbol_suffix);
    *file = (gen_file){0};
}

static void gen_vec_push(gen_vec *vec, gen_file file)
{
    if (vec->count == vec->cap) {
        size_t next_cap = vec->cap == 0u ? 16u : vec->cap * 2u;

        if (next_cap < vec->cap) {
            die_msg("too many operator files");
        }
        vec->items = xrealloc(vec->items, next_cap * sizeof(vec->items[0]));
        vec->cap = next_cap;
    }
    vec->items[vec->count] = file;
    vec->count++;
}

static void gen_vec_clear(gen_vec *vec)
{
    size_t i;

    for (i = 0u; i < vec->count; i++) {
        gen_file_clear(&vec->items[i]);
    }
    free(vec->items);
    *vec = (gen_vec){0};
}

static void str_vec_push_unique(str_vec *vec, const char *s)
{
    size_t i;

    for (i = 0u; i < vec->count; i++) {
        if (strcmp(vec->items[i], s) == 0) {
            return;
        }
    }
    if (vec->count == vec->cap) {
        size_t next_cap = vec->cap == 0u ? 16u : vec->cap * 2u;

        if (next_cap < vec->cap) {
            die_msg("too many operator kinds");
        }
        vec->items = xrealloc(vec->items, next_cap * sizeof(vec->items[0]));
        vec->cap = next_cap;
    }
    vec->items[vec->count] = xstrdup(s);
    vec->count++;
}

static void str_vec_clear(str_vec *vec)
{
    size_t i;

    for (i = 0u; i < vec->count; i++) {
        free(vec->items[i]);
    }
    free(vec->items);
    *vec = (str_vec){0};
}

static const gen_file *find_by_kind(const gen_vec *vec, const char *kind_enum)
{
    size_t i;

    for (i = 0u; i < vec->count; i++) {
        if (strcmp(vec->items[i].kind_enum, kind_enum) == 0) {
            return &vec->items[i];
        }
    }
    return NULL;
}

static void ensure_unique_kind(const gen_vec *vec, const gen_file *file, const char *category)
{
    if (find_by_kind(vec, file->kind_enum) != NULL) {
        (void)fprintf(stderr,
                      "gen_ops: duplicate %s entry for %s: %s\n",
                      category,
                      file->kind_enum,
                      file->path);
        exit(1);
    }
}

static void fill_common(gen_file *file, const char *path, char *op, char *role)
{
    file->path = xstrdup(path);
    file->op = op;
    file->role = role;
    file->name = kind_name_from_parts(op, role);
    file->kind_enum = kind_enum_from_name(file->name);
    file->symbol_suffix = xstrdup(file->name);
}

static gen_file parse_role_file(const char *path, const char *prefix, const char *ext)
{
    const char *base = path_basename(path);
    size_t prefix_len = strlen(prefix);
    size_t ext_len = strlen(ext);
    size_t base_len = strlen(base);
    size_t body_len;
    char *body = NULL;
    char *parent = NULL;
    char *op = NULL;
    char *role = NULL;
    gen_file out = {0};

    if (!starts_with(base, prefix) || !ends_with(base, ext) || base_len <= prefix_len + ext_len) {
        die_path("filename does not match expected operator capsule grammar", path);
    }

    body_len = base_len - prefix_len - ext_len;
    body = copy_range(base + prefix_len, body_len);
    parent = path_parent_basename(path);

    if (parent != NULL) {
        size_t parent_len = strlen(parent);

        if (!is_snake_name(parent)) {
            die_path("operator directory is not lowercase snake_case", path);
        }
        if (body_len <= parent_len ||
            strncmp(body, parent, parent_len) != 0 ||
            body[parent_len] != '_' ||
            body[parent_len + 1u] == '\0') {
            die_path("filename operator prefix does not match operator directory", path);
        }
        op = xstrdup(parent);
        role = xstrdup(body + parent_len + 1u);
    } else {
        char *sep = strrchr(body, '_');

        if (sep == NULL || sep == body || sep[1] == '\0') {
            die_path("filename missing role suffix", path);
        }
        *sep = '\0';
        op = xstrdup(body);
        role = xstrdup(sep + 1);
    }

    if (!is_snake_name(op)) {
        die_path("operator name is not lowercase snake_case", path);
    }
    if (!is_snake_name(role)) {
        die_path("operator role is not lowercase snake_case", path);
    }

    fill_common(&out, path, op, role);
    free(body);
    free(parent);
    return out;
}

static gen_file parse_grad_file(const char *path)
{
    const char *prefix = "grad_";
    const char *ext = ".c";
    const char *base = path_basename(path);
    size_t prefix_len = strlen(prefix);
    size_t ext_len = strlen(ext);
    size_t base_len = strlen(base);
    size_t body_len;
    char *op = NULL;
    char *parent = NULL;
    gen_file out = {0};

    if (!starts_with(base, prefix) || !ends_with(base, ext) || base_len <= prefix_len + ext_len) {
        die_path("filename does not match expected grad capsule grammar", path);
    }

    body_len = base_len - prefix_len - ext_len;
    op = copy_range(base + prefix_len, body_len);
    parent = path_parent_basename(path);

    if (parent != NULL) {
        if (!is_snake_name(parent)) {
            die_path("operator directory is not lowercase snake_case", path);
        }
        if (strcmp(parent, op) != 0) {
            die_path("grad filename operator does not match operator directory", path);
        }
    }
    if (!is_snake_name(op)) {
        die_path("operator name is not lowercase snake_case", path);
    }

    fill_common(&out, path, op, NULL);
    free(parent);
    return out;
}

static gen_category option_category(const char *arg)
{
    if (strcmp(arg, "--core") == 0) {
        return GEN_CORE;
    }
    if (strcmp(arg, "--grad") == 0) {
        return GEN_GRAD;
    }
    if (strcmp(arg, "--cpu") == 0) {
        return GEN_CPU;
    }
    if (strcmp(arg, "--metal") == 0) {
        return GEN_METAL;
    }
    if (strcmp(arg, "--metal-shaders") == 0) {
        return GEN_METAL_SHADER;
    }
    return GEN_NONE;
}

static bool is_option(const char *arg)
{
    return starts_with(arg, "--");
}

static void add_file(gen_inputs *inputs, gen_category category, const char *path)
{
    gen_file file;

    switch (category) {
    case GEN_CORE:
        file = parse_role_file(path, "core_", ".c");
        ensure_unique_kind(&inputs->core, &file, "core");
        gen_vec_push(&inputs->core, file);
        break;
    case GEN_GRAD:
        file = parse_grad_file(path);
        ensure_unique_kind(&inputs->grad, &file, "grad");
        gen_vec_push(&inputs->grad, file);
        break;
    case GEN_CPU:
        file = parse_role_file(path, "cpu_", ".c");
        ensure_unique_kind(&inputs->cpu, &file, "cpu");
        gen_vec_push(&inputs->cpu, file);
        break;
    case GEN_METAL:
        file = parse_role_file(path, "metal_", ".m");
        ensure_unique_kind(&inputs->metal, &file, "metal");
        gen_vec_push(&inputs->metal, file);
        break;
    case GEN_METAL_SHADER:
        file = parse_role_file(path, "metal_", ".metal");
        ensure_unique_kind(&inputs->metal_shader, &file, "metal shader");
        gen_vec_push(&inputs->metal_shader, file);
        break;
    case GEN_NONE:
        die_path("file argument appears before category option", path);
        break;
    }
}

static void parse_args(int argc, char **argv, gen_inputs *inputs)
{
    gen_category current = GEN_NONE;
    int i;

    if (argc == 1) {
        die_msg("usage: gen_ops --out DIR [--core files...] [--grad files...] [--cpu files...] [--metal files...] [--metal-shaders files...]");
    }

    for (i = 1; i < argc; i++) {
        gen_category category;

        if (strcmp(argv[i], "--out") == 0) {
            i++;
            if (i >= argc) {
                die_msg("--out requires a directory");
            }
            free(inputs->out_dir);
            inputs->out_dir = xstrdup(argv[i]);
            current = GEN_NONE;
            continue;
        }

        category = option_category(argv[i]);
        if (category != GEN_NONE) {
            current = category;
            continue;
        }

        if (is_option(argv[i])) {
            die_path("unknown option", argv[i]);
        }
        add_file(inputs, current, argv[i]);
    }

    if (inputs->out_dir == NULL) {
        die_msg("missing --out DIR");
    }
}

static char *out_path(const char *out_dir, const char *name)
{
    size_t len = strlen(out_dir);

    if (len > 0u && out_dir[len - 1u] == '/') {
        return concat2(out_dir, name);
    }
    return concat3(out_dir, "/", name);
}

static FILE *open_out(const char *out_dir, const char *name, char **path_out)
{
    char *path = out_path(out_dir, name);
    FILE *file = fopen(path, "w");

    if (file == NULL) {
        die_errno("open", path);
    }
    *path_out = path;
    return file;
}

static void close_out(FILE *file, const char *path)
{
    if (ferror(file) != 0) {
        die_errno("write", path);
    }
    if (fclose(file) != 0) {
        die_errno("close", path);
    }
}

static void emit_generated_banner(FILE *file)
{
    (void)fprintf(file, "/* Generated by tools/gen_ops.c; do not edit. */\n\n");
}

static void emit_op_kind_h(const gen_inputs *inputs)
{
    char *path = NULL;
    FILE *file = open_out(inputs->out_dir, "op_kind.h", &path);
    size_t i;

    emit_generated_banner(file);
    (void)fprintf(file, "#ifndef GRADIENTS_GENERATED_OP_KIND_H\n");
    (void)fprintf(file, "#define GRADIENTS_GENERATED_OP_KIND_H\n\n");
    (void)fprintf(file, "typedef enum _gd_op_kind {\n");
    (void)fprintf(file, "    _GD_OP_INVALID = 0,\n");
    for (i = 0u; i < inputs->core.count; i++) {
        (void)fprintf(file, "    %s,\n", inputs->core.items[i].kind_enum);
    }
    (void)fprintf(file, "    _GD_OP_COUNT\n");
    (void)fprintf(file, "} _gd_op_kind;\n\n");
    (void)fprintf(file, "#endif /* GRADIENTS_GENERATED_OP_KIND_H */\n");

    close_out(file, path);
    free(path);
}

static void emit_op_registry_inc(const gen_inputs *inputs)
{
    char *path = NULL;
    FILE *file = open_out(inputs->out_dir, "op_registry.inc", &path);
    size_t i;

    emit_generated_banner(file);
    for (i = 0u; i < inputs->core.count; i++) {
        const gen_file *entry = &inputs->core.items[i];

        (void)fprintf(file,
                      "extern const _gd_op_def _gd_opdef_%s;\n",
                      entry->symbol_suffix);
    }
    if (inputs->core.count > 0u) {
        (void)fprintf(file, "\n");
    }
    if (inputs->core.count == 0u) {
        (void)fprintf(file, "static const _gd_op_def *const g_op_defs[_GD_OP_COUNT] = {0};\n\n");
    } else {
        (void)fprintf(file, "static const _gd_op_def *const g_op_defs[_GD_OP_COUNT] = {\n");
        for (i = 0u; i < inputs->core.count; i++) {
            const gen_file *entry = &inputs->core.items[i];

            (void)fprintf(file,
                          "    [%s] = &_gd_opdef_%s,\n",
                          entry->kind_enum,
                          entry->symbol_suffix);
        }
        (void)fprintf(file, "};\n\n");
    }
    (void)fprintf(file, "static const char *const g_op_kind_names[_GD_OP_COUNT] = {\n");
    (void)fprintf(file, "    [_GD_OP_INVALID] = \"invalid\",\n");
    for (i = 0u; i < inputs->core.count; i++) {
        const gen_file *entry = &inputs->core.items[i];

        (void)fprintf(file,
                      "    [%s] = \"%s\",\n",
                      entry->kind_enum,
                      entry->name);
    }
    (void)fprintf(file, "};\n");

    close_out(file, path);
    free(path);
}

static void emit_bwd_registry_inc(const gen_inputs *inputs)
{
    char *path = NULL;
    FILE *file = open_out(inputs->out_dir, "bwd_registry.inc", &path);
    size_t i;

    emit_generated_banner(file);
    for (i = 0u; i < inputs->grad.count; i++) {
        const gen_file *entry = &inputs->grad.items[i];

        (void)fprintf(file,
                      "extern const _gd_bwd_rule _gd_bwd_rule_%s;\n",
                      entry->symbol_suffix);
    }
    if (inputs->grad.count > 0u) {
        (void)fprintf(file, "\n");
    }
    if (inputs->grad.count == 0u) {
        (void)fprintf(file, "static const _gd_bwd_rule *const g_bwd_rules[_GD_OP_COUNT] = {0};\n");
    } else {
        (void)fprintf(file, "static const _gd_bwd_rule *const g_bwd_rules[_GD_OP_COUNT] = {\n");
        for (i = 0u; i < inputs->grad.count; i++) {
            const gen_file *entry = &inputs->grad.items[i];

            (void)fprintf(file,
                          "    [%s] = &_gd_bwd_rule_%s,\n",
                          entry->kind_enum,
                          entry->symbol_suffix);
        }
        (void)fprintf(file, "};\n");
    }

    close_out(file, path);
    free(path);
}

static void emit_backend_registry_inc(const gen_inputs *inputs,
                                      const gen_vec *vec,
                                      const char *filename,
                                      const char *type_name,
                                      const char *symbol_prefix,
                                      const char *array_name,
                                      const char *count_name)
{
    char *path = NULL;
    FILE *file = open_out(inputs->out_dir, filename, &path);
    size_t i;

    emit_generated_banner(file);
    for (i = 0u; i < vec->count; i++) {
        const gen_file *entry = &vec->items[i];

        (void)fprintf(file,
                      "extern const %s %s_%s;\n",
                      type_name,
                      symbol_prefix,
                      entry->symbol_suffix);
    }
    if (vec->count > 0u) {
        (void)fprintf(file, "\n");
    }
    if (vec->count == 0u) {
        (void)fprintf(file, "static const %s *const %s[1] = {0};\n", type_name, array_name);
    } else {
        (void)fprintf(file,
                      "static const %s *const %s[%zu] = {\n",
                      type_name,
                      array_name,
                      vec->count);
        for (i = 0u; i < vec->count; i++) {
            const gen_file *entry = &vec->items[i];

            (void)fprintf(file, "    &%s_%s,\n", symbol_prefix, entry->symbol_suffix);
        }
        (void)fprintf(file, "};\n");
    }
    (void)fprintf(file, "static const unsigned %s = %zuu;\n", count_name, vec->count);

    close_out(file, path);
    free(path);
}

static void emit_metal_shaders_mk(const gen_inputs *inputs)
{
    char *path = NULL;
    FILE *file = open_out(inputs->out_dir, "metal_shaders.mk", &path);
    size_t i;

    (void)fprintf(file, "# Generated by tools/gen_ops.c; do not edit.\n\n");
    (void)fprintf(file, "METAL_OP_SHADER_FILES :=");
    if (inputs->metal_shader.count == 0u) {
        (void)fprintf(file, "\n");
    } else {
        (void)fprintf(file, " \\\n");
        for (i = 0u; i < inputs->metal_shader.count; i++) {
            const char *suffix = i + 1u == inputs->metal_shader.count ? "" : " \\";

            (void)fprintf(file, "  %s%s\n", inputs->metal_shader.items[i].path, suffix);
        }
    }

    close_out(file, path);
    free(path);
}

static const char *matrix_path(const gen_vec *vec, const char *kind_enum)
{
    const gen_file *entry = find_by_kind(vec, kind_enum);

    return entry == NULL ? "" : entry->path;
}

static const char *matrix_mark(const gen_vec *vec, const char *kind_enum)
{
    return find_by_kind(vec, kind_enum) == NULL ? "" : "yes";
}

static void matrix_add_all(str_vec *kinds, const gen_vec *vec)
{
    size_t i;

    for (i = 0u; i < vec->count; i++) {
        str_vec_push_unique(kinds, vec->items[i].kind_enum);
    }
}

static void emit_op_matrix_md(const gen_inputs *inputs)
{
    char *path = NULL;
    FILE *file = open_out(inputs->out_dir, "op_matrix.md", &path);
    str_vec kinds = {0};
    size_t i;

    matrix_add_all(&kinds, &inputs->core);
    matrix_add_all(&kinds, &inputs->grad);
    matrix_add_all(&kinds, &inputs->cpu);
    matrix_add_all(&kinds, &inputs->metal);
    matrix_add_all(&kinds, &inputs->metal_shader);

    (void)fprintf(file, "# Operator Matrix\n\n");
    (void)fprintf(file, "Generated by `tools/gen_ops.c`. Source of truth is `src/ops/*/*`.\n\n");
    if (kinds.count == 0u) {
        (void)fprintf(file, "No operator capsule files discovered.\n");
    } else {
        (void)fprintf(file, "| op kind | core | grad | cpu | metal host | metal shader |\n");
        (void)fprintf(file, "| --- | --- | --- | --- | --- | --- |\n");
        for (i = 0u; i < kinds.count; i++) {
            const char *kind = kinds.items[i];

            (void)fprintf(file,
                          "| `%s` | %s | %s | %s | %s | %s |\n",
                          kind,
                          matrix_path(&inputs->core, kind),
                          matrix_mark(&inputs->grad, kind),
                          matrix_mark(&inputs->cpu, kind),
                          matrix_mark(&inputs->metal, kind),
                          matrix_mark(&inputs->metal_shader, kind));
        }
    }

    str_vec_clear(&kinds);
    close_out(file, path);
    free(path);
}

static void emit_all(const gen_inputs *inputs)
{
    emit_op_kind_h(inputs);
    emit_op_registry_inc(inputs);
    emit_bwd_registry_inc(inputs);
    emit_backend_registry_inc(inputs,
                              &inputs->cpu,
                              "cpu_registry.inc",
                              "_gd_cpu_op",
                              "_gd_cpu_op",
                              "g_cpu_ops",
                              "g_cpu_op_count");
    emit_backend_registry_inc(inputs,
                              &inputs->metal,
                              "metal_registry.inc",
                              "_gd_metal_op",
                              "_gd_metal_op",
                              "g_metal_ops",
                              "g_metal_op_count");
    emit_metal_shaders_mk(inputs);
    emit_op_matrix_md(inputs);
}

static void gen_inputs_clear(gen_inputs *inputs)
{
    free(inputs->out_dir);
    gen_vec_clear(&inputs->core);
    gen_vec_clear(&inputs->grad);
    gen_vec_clear(&inputs->cpu);
    gen_vec_clear(&inputs->metal);
    gen_vec_clear(&inputs->metal_shader);
    *inputs = (gen_inputs){0};
}

int main(int argc, char **argv)
{
    gen_inputs inputs = {0};

    parse_args(argc, argv, &inputs);
    emit_all(&inputs);
    gen_inputs_clear(&inputs);
    return 0;
}
