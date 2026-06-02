#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define GD_NEW_OP_MAX_NAME 128
#define GD_NEW_OP_MAX_PATH 4096
#define GD_NEW_OP_MAX_ARITY 32

typedef struct string_builder {
    char *data;
    size_t len;
    size_t cap;
} string_builder;

typedef struct op_config {
    char name[GD_NEW_OP_MAX_NAME];
    char upper[GD_NEW_OP_MAX_NAME];
    int inputs;
    int outputs;
    bool is_public;
    bool diff;
    bool custom_bwd;
    bool cpu_only;
    bool force;
    bool dry_run;
    bool run_generated;
} op_config;

static void usage(FILE *f)
{
    fprintf(f,
            "usage: gradients-new-op OP_NAME [options]\n"
            "\n"
            "Scaffold compile-valid operator stubs. Stubs return GD_ERR_UNSUPPORTED\n"
            "until math/meta/backward/backend implementations are filled.\n"
            "\n"
            "defaults:\n"
            "  public=yes diff=yes custom_bwd=yes backends=cpu_ref+accelerated inputs=1 outputs=1\n"
            "\n"
            "options:\n"
            "  --private         skip public API and public-symbol test patches\n"
            "  --no-diff         forward-only op; skip autograd and bwd stubs\n"
            "  --no-custom-bwd   add grad rule stub only; skip explicit *_bwd op stubs\n"
            "  --inputs N        forward input count (default: 1)\n"
            "  --outputs N       forward output count (default: 1)\n"
            "  --cpu-only        skip accelerated backend stubs (Metal today)\n"
            "  --no-generated    do not run `make generated` after scaffolding\n"
            "  --force           overwrite existing scaffold files\n"
            "  --dry-run         print plan without writing files or running make\n"
            "  --help            show this help\n");
}

static void sb_free(string_builder *sb)
{
    if (sb != NULL) {
        free(sb->data);
        sb->data = NULL;
        sb->len = 0U;
        sb->cap = 0U;
    }
}

static int sb_reserve(string_builder *sb, size_t need)
{
    char *next = NULL;
    size_t cap = 0U;

    if (sb == NULL) {
        return 1;
    }
    if (need <= sb->cap) {
        return 0;
    }
    cap = sb->cap == 0U ? 256U : sb->cap;
    while (cap < need) {
        if (cap > (size_t)-1 / 2U) {
            return 1;
        }
        cap *= 2U;
    }
    next = (char *)realloc(sb->data, cap);
    if (next == NULL) {
        return 1;
    }
    sb->data = next;
    sb->cap = cap;
    return 0;
}

static int sb_append_n(string_builder *sb, const char *s, size_t n)
{
    if (sb == NULL || (n > 0U && s == NULL)) {
        return 1;
    }
    if (n > (size_t)-1 - sb->len - 1U) {
        return 1;
    }
    if (sb_reserve(sb, sb->len + n + 1U) != 0) {
        return 1;
    }
    if (n > 0U) {
        memcpy(sb->data + sb->len, s, n);
    }
    sb->len += n;
    sb->data[sb->len] = '\0';
    return 0;
}

static int sb_append(string_builder *sb, const char *s)
{
    return sb_append_n(sb, s, strlen(s));
}

static int sb_appendf(string_builder *sb, const char *fmt, ...)
{
    va_list ap;
    va_list ap2;
    int n = 0;
    size_t old_len = 0U;

    if (sb == NULL || fmt == NULL) {
        return 1;
    }
    va_start(ap, fmt);
    va_copy(ap2, ap);
    n = vsnprintf(NULL, 0U, fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(ap2);
        return 1;
    }
    old_len = sb->len;
    if (sb_reserve(sb, old_len + (size_t)n + 1U) != 0) {
        va_end(ap2);
        return 1;
    }
    (void)vsnprintf(sb->data + old_len, sb->cap - old_len, fmt, ap2);
    va_end(ap2);
    sb->len = old_len + (size_t)n;
    return 0;
}

static bool file_exists(const char *path)
{
    struct stat st;
    return path != NULL && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static bool dir_exists(const char *path)
{
    struct stat st;
    return path != NULL && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int mkdir_p(const char *path)
{
    char tmp[GD_NEW_OP_MAX_PATH];
    size_t len = 0U;
    char *p = NULL;

    if (path == NULL || path[0] == '\0') {
        return 1;
    }
    len = strlen(path);
    if (len >= sizeof(tmp)) {
        return 1;
    }
    memcpy(tmp, path, len + 1U);
    if (len > 1U && tmp[len - 1U] == '/') {
        tmp[len - 1U] = '\0';
    }
    for (p = tmp + 1; *p != '\0'; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
                return 1;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
        return 1;
    }
    return 0;
}

static bool is_lower_snake(const char *s)
{
    size_t i = 0U;
    size_t len = 0U;
    bool prev_underscore = false;

    if (s == NULL) {
        return false;
    }
    len = strlen(s);
    if (len == 0U || len >= GD_NEW_OP_MAX_NAME || s[0] == '_' || s[len - 1U] == '_') {
        return false;
    }
    for (i = 0U; i < len; ++i) {
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

static void make_upper(const char *name, char *out, size_t cap)
{
    size_t i = 0U;
    size_t n = strlen(name);

    if (cap == 0U) {
        return;
    }
    if (n >= cap) {
        n = cap - 1U;
    }
    for (i = 0U; i < n; ++i) {
        char c = name[i];
        out[i] = (c >= 'a' && c <= 'z') ? (char)(c - ('a' - 'A')) : c;
    }
    out[n] = '\0';
}

static int parse_int_arg(const char *s, int *out)
{
    char *end = NULL;
    long v = 0L;

    if (s == NULL || out == NULL) {
        return 1;
    }
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 1L || v > GD_NEW_OP_MAX_ARITY) {
        return 1;
    }
    *out = (int)v;
    return 0;
}

static int parse_args(int argc, char **argv, op_config *cfg)
{
    int i = 0;
    bool have_name = false;

    if (cfg == NULL) {
        return 2;
    }
    memset(cfg, 0, sizeof(*cfg));
    cfg->inputs = 1;
    cfg->outputs = 1;
    cfg->is_public = true;
    cfg->diff = true;
    cfg->custom_bwd = true;
    cfg->run_generated = true;

    if (argc <= 1) {
        usage(stderr);
        return 2;
    }
    for (i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--help") == 0) {
            usage(stdout);
            return 0;
        } else if (strcmp(arg, "--private") == 0) {
            cfg->is_public = false;
        } else if (strcmp(arg, "--no-diff") == 0) {
            cfg->diff = false;
            cfg->custom_bwd = false;
        } else if (strcmp(arg, "--no-custom-bwd") == 0) {
            cfg->custom_bwd = false;
        } else if (strcmp(arg, "--inputs") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &cfg->inputs) != 0) {
                fprintf(stderr, "invalid --inputs (expected 1..%d)\n", GD_NEW_OP_MAX_ARITY);
                return 2;
            }
        } else if (strcmp(arg, "--outputs") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &cfg->outputs) != 0) {
                fprintf(stderr, "invalid --outputs (expected 1..%d)\n", GD_NEW_OP_MAX_ARITY);
                return 2;
            }
        } else if (strcmp(arg, "--cpu-only") == 0) {
            cfg->cpu_only = true;
        } else if (strcmp(arg, "--no-generated") == 0) {
            cfg->run_generated = false;
        } else if (strcmp(arg, "--force") == 0) {
            cfg->force = true;
        } else if (strcmp(arg, "--dry-run") == 0) {
            cfg->dry_run = true;
        } else if (arg[0] == '-') {
            fprintf(stderr, "unknown option: %s\n", arg);
            return 2;
        } else if (!have_name) {
            if (!is_lower_snake(arg)) {
                fprintf(stderr, "invalid OP_NAME '%s' (use lowercase snake_case)\n", arg);
                return 2;
            }
            (void)snprintf(cfg->name, sizeof(cfg->name), "%s", arg);
            have_name = true;
        } else {
            fprintf(stderr, "unexpected argument: %s\n", arg);
            return 2;
        }
    }
    if (!have_name) {
        fprintf(stderr, "missing OP_NAME\n");
        return 2;
    }
    if (!cfg->diff) {
        cfg->custom_bwd = false;
    }
    make_upper(cfg->name, cfg->upper, sizeof(cfg->upper));
    return 1;
}

static int read_file(const char *path, char **text_out, size_t *len_out)
{
    FILE *f = NULL;
    long end = 0L;
    char *text = NULL;
    size_t nread = 0U;

    if (path == NULL || text_out == NULL || len_out == NULL) {
        return 1;
    }
    *text_out = NULL;
    *len_out = 0U;
    f = fopen(path, "rb");
    if (f == NULL) {
        return 1;
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        (void)fclose(f);
        return 1;
    }
    end = ftell(f);
    if (end < 0L) {
        (void)fclose(f);
        return 1;
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        (void)fclose(f);
        return 1;
    }
    text = (char *)malloc((size_t)end + 1U);
    if (text == NULL) {
        (void)fclose(f);
        return 1;
    }
    nread = fread(text, 1U, (size_t)end, f);
    if (nread != (size_t)end) {
        free(text);
        (void)fclose(f);
        return 1;
    }
    if (fclose(f) != 0) {
        free(text);
        return 1;
    }
    text[(size_t)end] = '\0';
    *text_out = text;
    *len_out = (size_t)end;
    return 0;
}

static int write_text_file(const char *path, const char *text, bool force, bool dry_run)
{
    FILE *f = NULL;
    const char *slash = NULL;
    char dir[GD_NEW_OP_MAX_PATH];
    size_t dir_len = 0U;
    size_t n = 0U;

    if (path == NULL || text == NULL) {
        return 1;
    }
    if (dry_run) {
        printf("create: %s%s\n", path, file_exists(path) ? " (exists)" : "");
        return 0;
    }
    if (file_exists(path) && !force) {
        fprintf(stderr, "refusing to overwrite existing file: %s (pass --force)\n", path);
        return 1;
    }
    slash = strrchr(path, '/');
    if (slash != NULL) {
        dir_len = (size_t)(slash - path);
        if (dir_len >= sizeof(dir)) {
            return 1;
        }
        memcpy(dir, path, dir_len);
        dir[dir_len] = '\0';
        if (!dir_exists(dir) && mkdir_p(dir) != 0) {
            fprintf(stderr, "failed to create directory: %s\n", dir);
            return 1;
        }
    }
    f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return 1;
    }
    n = strlen(text);
    if (fwrite(text, 1U, n, f) != n) {
        (void)fclose(f);
        fprintf(stderr, "failed to write %s\n", path);
        return 1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "failed to close %s: %s\n", path, strerror(errno));
        return 1;
    }
    printf("create: %s\n", path);
    return 0;
}

static int overwrite_text_file(const char *path, const char *text)
{
    FILE *f = NULL;
    size_t n = 0U;

    if (path == NULL || text == NULL) {
        return 1;
    }
    f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return 1;
    }
    n = strlen(text);
    if (fwrite(text, 1U, n, f) != n) {
        (void)fclose(f);
        fprintf(stderr, "failed to write %s\n", path);
        return 1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "failed to close %s: %s\n", path, strerror(errno));
        return 1;
    }
    return 0;
}

static int patch_insert_before(const char *path,
                               const char *marker,
                               const char *insert,
                               const char *already_token,
                               bool dry_run)
{
    char *text = NULL;
    size_t len = 0U;
    char *pos = NULL;
    string_builder out = {0};
    int rc = 1;

    if (path == NULL || marker == NULL || insert == NULL || already_token == NULL) {
        return 1;
    }
    if (read_file(path, &text, &len) != 0) {
        fprintf(stderr, "failed to read patch target: %s\n", path);
        return 1;
    }
    if (strstr(text, already_token) != NULL) {
        printf("patch: %s (already contains %s)\n", path, already_token);
        free(text);
        return 0;
    }
    pos = strstr(text, marker);
    if (pos == NULL) {
        fprintf(stderr, "patch marker not found in %s: %s\n", path, marker);
        free(text);
        return 1;
    }
    if (dry_run) {
        printf("patch: %s\n", path);
        free(text);
        return 0;
    }
    if (sb_append_n(&out, text, (size_t)(pos - text)) != 0 ||
        sb_append(&out, insert) != 0 ||
        sb_append(&out, pos) != 0) {
        fprintf(stderr, "out of memory while patching %s\n", path);
        goto done;
    }
    if (overwrite_text_file(path, out.data) != 0) {
        goto done;
    }
    printf("patch: %s\n", path);
    rc = 0;

done:
    sb_free(&out);
    free(text);
    return rc;
}

static int append_input_params(string_builder *sb, const op_config *cfg, bool names)
{
    int i = 0;

    for (i = 0; i < cfg->inputs; ++i) {
        if (sb_append(sb, ", gd_tensor *") != 0) {
            return 1;
        }
        if (names) {
            if (cfg->inputs == 1) {
                if (sb_append(sb, "x") != 0) {
                    return 1;
                }
            } else if (sb_appendf(sb, "x%d", i) != 0) {
                return 1;
            }
        }
    }
    return 0;
}

static int append_output_params(string_builder *sb, const op_config *cfg, bool names)
{
    int i = 0;

    for (i = 0; i < cfg->outputs; ++i) {
        if (sb_append(sb, ", gd_tensor **") != 0) {
            return 1;
        }
        if (names) {
            if (cfg->outputs == 1) {
                if (sb_append(sb, "out") != 0) {
                    return 1;
                }
            } else if (sb_appendf(sb, "out%d", i) != 0) {
                return 1;
            }
        }
    }
    return 0;
}

static int append_public_proto(string_builder *sb, const op_config *cfg, bool names)
{
    if (sb_appendf(sb, "gd_status gd_%s(gd_context *ctx", cfg->name) != 0 ||
        append_input_params(sb, cfg, names) != 0 ||
        append_output_params(sb, cfg, names) != 0 ||
        sb_append(sb, ")") != 0) {
        return 1;
    }
    return 0;
}

static int append_gd_ref_args(string_builder *sb, const op_config *cfg)
{
    int i = 0;

    if (sb_append(sb, "(gd_context *") != 0) {
        return 1;
    }
    for (i = 0; i < cfg->inputs; ++i) {
        if (sb_append(sb, ", gd_tensor *") != 0) {
            return 1;
        }
    }
    for (i = 0; i < cfg->outputs; ++i) {
        if (sb_append(sb, ", gd_tensor **") != 0) {
            return 1;
        }
    }
    return sb_append(sb, ")");
}

static int build_ops_h_patch(const op_config *cfg, string_builder *sb)
{
    if (sb_appendf(sb, "/* Autogenerated scaffold for `%s`. TODO: document op contract after implementation. */\n", cfg->name) != 0 ||
        append_public_proto(sb, cfg, true) != 0 ||
        sb_append(sb, ";\n\n") != 0) {
        return 1;
    }
    return 0;
}

static int build_public_symbols_patch(const op_config *cfg, string_builder *sb)
{
    if (sb_appendf(sb, "GD_REF(gd_%s, gd_status, ", cfg->name) != 0 ||
        append_gd_ref_args(sb, cfg) != 0 ||
        sb_append(sb, ");\n") != 0) {
        return 1;
    }
    return 0;
}

static int build_registry_test_patch(const op_config *cfg, string_builder *sb)
{
    if (sb_appendf(sb, "    {_GD_OP_%s, \"%s\"},\n", cfg->upper, cfg->name) != 0) {
        return 1;
    }
    if (cfg->custom_bwd &&
        sb_appendf(sb, "    {_GD_OP_%s_BWD, \"%s_bwd\"},\n", cfg->upper, cfg->name) != 0) {
        return 1;
    }
    return 0;
}

static int build_metal_backend_patch(const op_config *cfg, string_builder *sb)
{
    if (sb_appendf(sb, "    {_GD_OP_%s, \"gd_%s\"},\n", cfg->upper, cfg->name) != 0) {
        return 1;
    }
    if (cfg->custom_bwd &&
        sb_appendf(sb, "    {_GD_OP_%s_BWD, \"gd_%s_bwd\"},\n", cfg->upper, cfg->name) != 0) {
        return 1;
    }
    return 0;
}

static int build_core_fwd(const op_config *cfg, string_builder *sb)
{
    const char *flags = NULL;

    if (cfg->is_public && cfg->diff) {
        flags = "GD_OPF_PUBLIC | GD_OPF_DIFF";
    } else if (cfg->is_public) {
        flags = "GD_OPF_PUBLIC";
    } else if (cfg->diff) {
        flags = "GD_OPF_INTERNAL | GD_OPF_DIFF";
    } else {
        flags = "GD_OPF_INTERNAL";
    }

    if (sb_append(sb,
                  "#include \"../op_impl.h\"\n"
                  "#include \"../meta_common.h\"\n") != 0) {
        return 1;
    }
    if (cfg->is_public && sb_append(sb, "#include \"gradients/ops.h\"\n") != 0) {
        return 1;
    }
    if (sb_appendf(sb,
                   "\n"
                   "#include \"../../core/internal.h\"\n"
                   "\n"
                   "#define GD_%s_N_INPUTS %d\n"
                   "#define GD_%s_N_OUTPUTS %d\n"
                   "\n"
                   "/* Autogenerated scaffold. TODO: implement `%s` meta validation and shape inference. */\n"
                   "static gd_status %s_meta(const gd_tensor_desc *const *inputs,\n"
                   "                                int n_inputs,\n"
                   "                                _gd_op_attrs *attrs,\n"
                   "                                gd_tensor_desc *outputs,\n"
                   "                                int *n_outputs)\n"
                   "{\n"
                   "    gd_status status = GD_OK;\n"
                   "    int i = 0;\n"
                   "\n"
                   "    (void)attrs;\n"
                   "    if (inputs == NULL || outputs == NULL || n_outputs == NULL) {\n"
                   "        return _gd_error(GD_ERR_INVALID_ARGUMENT, \"%s meta arguments are NULL\");\n"
                   "    }\n"
                   "    if (n_inputs != GD_%s_N_INPUTS) {\n"
                   "        return _gd_error(GD_ERR_INVALID_ARGUMENT, \"%s input count mismatch\");\n"
                   "    }\n"
                   "    if (inputs[0] == NULL) {\n"
                   "        return _gd_error(GD_ERR_INVALID_ARGUMENT, \"%s input desc is NULL\");\n"
                   "    }\n"
                   "    status = _gd_meta_set_output_count(GD_%s_N_OUTPUTS, n_outputs);\n"
                   "    if (status != GD_OK) {\n"
                   "        return status;\n"
                   "    }\n"
                   "    for (i = 0; i < GD_%s_N_OUTPUTS; ++i) {\n"
                   "        outputs[i] = *inputs[0];\n"
                   "    }\n"
                   "    return GD_OK;\n"
                   "}\n"
                   "\n"
                   "const _gd_op_def _gd_opdef_%s = {\n"
                   "    .kind = _GD_OP_%s,\n"
                   "    .name = \"%s\",\n"
                   "    .min_inputs = GD_%s_N_INPUTS,\n"
                   "    .max_inputs = GD_%s_N_INPUTS,\n"
                   "    .n_outputs = GD_%s_N_OUTPUTS,\n"
                   "    .flags = %s,\n"
                   "    .meta = %s_meta,\n"
                   "};\n",
                   cfg->upper, cfg->inputs,
                   cfg->upper, cfg->outputs,
                   cfg->name,
                   cfg->name,
                   cfg->name,
                   cfg->upper,
                   cfg->name,
                   cfg->name,
                   cfg->upper,
                   cfg->upper,
                   cfg->name,
                   cfg->upper,
                   cfg->name,
                   cfg->upper,
                   cfg->upper,
                   cfg->upper,
                   flags,
                   cfg->name) != 0) {
        return 1;
    }
    if (cfg->is_public) {
        int i = 0;
        if (sb_append(sb, "\n") != 0 || append_public_proto(sb, cfg, true) != 0 ||
            sb_append(sb, "\n{\n") != 0 ||
            sb_appendf(sb, "    gd_tensor *inputs[GD_%s_N_INPUTS] = {", cfg->upper) != 0) {
            return 1;
        }
        for (i = 0; i < cfg->inputs; ++i) {
            if (i > 0 && sb_append(sb, ", ") != 0) {
                return 1;
            }
            if (cfg->inputs == 1) {
                if (sb_append(sb, "x") != 0) {
                    return 1;
                }
            } else if (sb_appendf(sb, "x%d", i) != 0) {
                return 1;
            }
        }
        if (sb_appendf(sb, "};\n    gd_tensor *outputs[GD_%s_N_OUTPUTS] = {0};\n    gd_status status = GD_OK;\n\n",
                       cfg->upper) != 0) {
            return 1;
        }
        if (sb_append(sb, "    if (ctx == NULL") != 0) {
            return 1;
        }
        for (i = 0; i < cfg->inputs; ++i) {
            if (cfg->inputs == 1) {
                if (sb_append(sb, " || x == NULL") != 0) {
                    return 1;
                }
            } else if (sb_appendf(sb, " || x%d == NULL", i) != 0) {
                return 1;
            }
        }
        for (i = 0; i < cfg->outputs; ++i) {
            if (cfg->outputs == 1) {
                if (sb_append(sb, " || out == NULL") != 0) {
                    return 1;
                }
            } else if (sb_appendf(sb, " || out%d == NULL", i) != 0) {
                return 1;
            }
        }
        if (sb_appendf(sb, ") {\n        return _gd_error(GD_ERR_INVALID_ARGUMENT, \"gd_%s argument is NULL\");\n    }\n",
                       cfg->name) != 0) {
            return 1;
        }
        for (i = 0; i < cfg->outputs; ++i) {
            if (cfg->outputs == 1) {
                if (sb_append(sb, "    *out = NULL;\n") != 0) {
                    return 1;
                }
            } else if (sb_appendf(sb, "    *out%d = NULL;\n", i) != 0) {
                return 1;
            }
        }
        if (sb_appendf(sb,
                       "    status = _gd_emit_checked(ctx, _GD_OP_%s, inputs, GD_%s_N_INPUTS,\n"
                       "                              NULL, outputs, GD_%s_N_OUTPUTS);\n"
                       "    if (status != GD_OK) {\n"
                       "        return status;\n"
                       "    }\n",
                       cfg->upper, cfg->upper, cfg->upper) != 0) {
            return 1;
        }
        for (i = 0; i < cfg->outputs; ++i) {
            if (cfg->outputs == 1) {
                if (sb_append(sb, "    *out = outputs[0];\n") != 0) {
                    return 1;
                }
            } else if (sb_appendf(sb, "    *out%d = outputs[%d];\n", i, i) != 0) {
                return 1;
            }
        }
        if (sb_append(sb, "    return GD_OK;\n}\n") != 0) {
            return 1;
        }
    }
    return 0;
}

static int build_core_bwd(const op_config *cfg, string_builder *sb)
{
    if (sb_appendf(sb,
                   "#include \"../op_impl.h\"\n"
                   "#include \"../meta_common.h\"\n"
                   "\n"
                   "#include \"../../core/internal.h\"\n"
                   "\n"
                   "#define GD_%s_BWD_N_INPUTS %d\n"
                   "#define GD_%s_BWD_N_OUTPUTS %d\n"
                   "#define GD_%s_FWD_N_OUTPUTS %d\n"
                   "\n"
                   "/* Autogenerated scaffold. TODO: implement `%s_bwd` meta validation and grad descs. */\n"
                   "static gd_status %s_bwd_meta(const gd_tensor_desc *const *inputs,\n"
                   "                                    int n_inputs,\n"
                   "                                    _gd_op_attrs *attrs,\n"
                   "                                    gd_tensor_desc *outputs,\n"
                   "                                    int *n_outputs)\n"
                   "{\n"
                   "    gd_status status = GD_OK;\n"
                   "    int i = 0;\n"
                   "\n"
                   "    (void)attrs;\n"
                   "    if (inputs == NULL || outputs == NULL || n_outputs == NULL) {\n"
                   "        return _gd_error(GD_ERR_INVALID_ARGUMENT, \"%s_bwd meta arguments are NULL\");\n"
                   "    }\n"
                   "    if (n_inputs != GD_%s_BWD_N_INPUTS) {\n"
                   "        return _gd_error(GD_ERR_INVALID_ARGUMENT, \"%s_bwd input count mismatch\");\n"
                   "    }\n"
                   "    status = _gd_meta_set_output_count(GD_%s_BWD_N_OUTPUTS, n_outputs);\n"
                   "    if (status != GD_OK) {\n"
                   "        return status;\n"
                   "    }\n"
                   "    for (i = 0; i < GD_%s_BWD_N_OUTPUTS; ++i) {\n"
                   "        outputs[i] = *inputs[GD_%s_FWD_N_OUTPUTS + i];\n"
                   "    }\n"
                   "    return GD_OK;\n"
                   "}\n"
                   "\n"
                   "const _gd_op_def _gd_opdef_%s_bwd = {\n"
                   "    .kind = _GD_OP_%s_BWD,\n"
                   "    .name = \"%s_bwd\",\n"
                   "    .min_inputs = GD_%s_BWD_N_INPUTS,\n"
                   "    .max_inputs = GD_%s_BWD_N_INPUTS,\n"
                   "    .n_outputs = GD_%s_BWD_N_OUTPUTS,\n"
                   "    .flags = GD_OPF_INTERNAL,\n"
                   "    .meta = %s_bwd_meta,\n"
                   "};\n",
                   cfg->upper, cfg->inputs + cfg->outputs,
                   cfg->upper, cfg->inputs,
                   cfg->upper, cfg->outputs,
                   cfg->name,
                   cfg->name,
                   cfg->name,
                   cfg->upper,
                   cfg->name,
                   cfg->upper,
                   cfg->upper,
                   cfg->upper,
                   cfg->name,
                   cfg->upper,
                   cfg->name,
                   cfg->upper,
                   cfg->upper,
                   cfg->upper,
                   cfg->name) != 0) {
        return 1;
    }
    return 0;
}

static int build_cpu_file(const op_config *cfg, bool bwd, string_builder *sb)
{
    const char *suffix = bwd ? "_bwd" : "";
    const char *kind_suffix = bwd ? "_BWD" : "";
    const char *part = bwd ? "CPU backward" : "CPU forward";

    if (sb_appendf(sb,
                   "#include \"../../backends/cpu_ref/cpu_op.h\"\n"
                   "\n"
                   "/* Autogenerated scaffold. TODO: implement `%s%s` CPU kernel in this op capsule.\n"
                   " * Do not add new per-op kernel bodies to src/backends/cpu_ref/cpu_kernels.c. */\n"
                   "static gd_status %s%s_run(_gd_cpu_exec *exec, const _gd_node *node)\n"
                   "{\n"
                   "    (void)exec;\n"
                   "    (void)node;\n"
                   "    return _gd_error(GD_ERR_UNSUPPORTED,\n"
                   "                     \"Autogenerated stub: op '%s%s' %s not implemented; TODO fill src/ops/%s/cpu_%s%s.c\");\n"
                   "}\n"
                   "\n"
                   "const _gd_cpu_op _gd_cpu_op_%s%s = {\n"
                   "    .kind = _GD_OP_%s%s,\n"
                   "    .name = \"%s%s\",\n"
                   "    .support = _gd_cpu_support_default,\n"
                   "    .run = %s%s_run,\n"
                   "};\n",
                   cfg->name, suffix,
                   cfg->name, suffix,
                   cfg->name, suffix, part, cfg->name, cfg->name, suffix,
                   cfg->name, suffix,
                   cfg->upper, kind_suffix,
                   cfg->name, suffix,
                   cfg->name, suffix) != 0) {
        return 1;
    }
    return 0;
}

static int build_grad_file(const op_config *cfg, string_builder *sb)
{
    if (sb_appendf(sb,
                   "#include \"../grad_impl.h\"\n"
                   "\n"
                   "/* Autogenerated scaffold. TODO: implement backward rule for `%s`. */\n"
                   "static gd_status %s_backward(_gd_bwd_ctx *b, const _gd_node *node)\n"
                   "{\n"
                   "    (void)b;\n"
                   "    (void)node;\n"
                   "    return _gd_error(GD_ERR_UNSUPPORTED,\n"
                   "                     \"Autogenerated stub: op '%s' backward rule not implemented; TODO fill src/ops/%s/grad_%s.c\");\n"
                   "}\n"
                   "\n"
                   "const _gd_bwd_rule _gd_bwd_rule_%s = {\n"
                   "    .op = _GD_OP_%s,\n"
                   "    .fn = %s_backward,\n"
                   "    .unsupported_reason = NULL,\n"
                   "};\n",
                   cfg->name,
                   cfg->name,
                   cfg->name, cfg->name, cfg->name,
                   cfg->name,
                   cfg->upper,
                   cfg->name) != 0) {
        return 1;
    }
    return 0;
}

static int build_metal_host(const op_config *cfg, bool bwd, string_builder *sb)
{
    const char *suffix = bwd ? "_bwd" : "";
    const char *kind_suffix = bwd ? "_BWD" : "";
    const char *part = bwd ? "Metal backward" : "Metal forward";

    if (sb_appendf(sb,
                   "#import \"../../backends/metal/metal_op.h\"\n"
                   "\n"
                   "/* Autogenerated scaffold. TODO: implement `%s%s` Metal encoder. */\n"
                   "static gd_status %s%s_encode(_gd_metal_encode_ctx *ctx)\n"
                   "{\n"
                   "    (void)ctx;\n"
                   "    return _gd_error(GD_ERR_UNSUPPORTED,\n"
                   "                     \"Autogenerated stub: op '%s%s' %s not implemented; TODO fill src/ops/%s/metal_%s%s.m\");\n"
                   "}\n"
                   "\n"
                   "const _gd_metal_op _gd_metal_op_%s%s = {\n"
                   "    .kind = _GD_OP_%s%s,\n"
                   "    .name = \"%s%s\",\n"
                   "    .support = _gd_metal_support_default,\n"
                   "    .encode = %s%s_encode,\n"
                   "};\n",
                   cfg->name, suffix,
                   cfg->name, suffix,
                   cfg->name, suffix, part, cfg->name, cfg->name, suffix,
                   cfg->name, suffix,
                   cfg->upper, kind_suffix,
                   cfg->name, suffix,
                   cfg->name, suffix) != 0) {
        return 1;
    }
    return 0;
}

static int build_metal_shader(const op_config *cfg, bool bwd, string_builder *sb)
{
    const char *suffix = bwd ? "_bwd" : "";

    if (sb_appendf(sb,
                   "#include \"metal_common.metal\"\n"
                   "\n"
                   "/* Autogenerated scaffold. TODO: implement `gd_%s%s` Metal shader.\n"
                   " * Host encoder currently returns GD_ERR_UNSUPPORTED, so this kernel is not called. */\n"
                   "kernel void gd_%s%s(uint gid [[thread_position_in_grid]])\n"
                   "{\n"
                   "    (void)gid;\n"
                   "}\n",
                   cfg->name, suffix,
                   cfg->name, suffix) != 0) {
        return 1;
    }
    return 0;
}

static int write_built_file(const char *path,
                            int (*builder)(const op_config *, string_builder *),
                            const op_config *cfg)
{
    string_builder sb = {0};
    int rc = 1;

    if (builder(cfg, &sb) != 0) {
        fprintf(stderr, "failed to build template for %s\n", path);
        goto done;
    }
    if (write_text_file(path, sb.data, cfg->force, cfg->dry_run) != 0) {
        goto done;
    }
    rc = 0;

done:
    sb_free(&sb);
    return rc;
}

static int build_cpu_fwd_wrapper(const op_config *cfg, string_builder *sb)
{
    return build_cpu_file(cfg, false, sb);
}

static int build_cpu_bwd_wrapper(const op_config *cfg, string_builder *sb)
{
    return build_cpu_file(cfg, true, sb);
}

static int build_metal_fwd_host_wrapper(const op_config *cfg, string_builder *sb)
{
    return build_metal_host(cfg, false, sb);
}

static int build_metal_bwd_host_wrapper(const op_config *cfg, string_builder *sb)
{
    return build_metal_host(cfg, true, sb);
}

static int build_metal_fwd_shader_wrapper(const op_config *cfg, string_builder *sb)
{
    return build_metal_shader(cfg, false, sb);
}

static int build_metal_bwd_shader_wrapper(const op_config *cfg, string_builder *sb)
{
    return build_metal_shader(cfg, true, sb);
}

static int scaffold_files(const op_config *cfg)
{
    char dir[GD_NEW_OP_MAX_PATH];
    char path[GD_NEW_OP_MAX_PATH];

    (void)snprintf(dir, sizeof(dir), "src/ops/%s", cfg->name);
    if (!cfg->dry_run && mkdir_p(dir) != 0) {
        fprintf(stderr, "failed to create op directory: %s\n", dir);
        return 1;
    }

    (void)snprintf(path, sizeof(path), "%s/core_%s_fwd.c", dir, cfg->name);
    if (write_built_file(path, build_core_fwd, cfg) != 0) {
        return 1;
    }
    (void)snprintf(path, sizeof(path), "%s/cpu_%s_fwd.c", dir, cfg->name);
    if (write_built_file(path, build_cpu_fwd_wrapper, cfg) != 0) {
        return 1;
    }
    if (cfg->diff) {
        (void)snprintf(path, sizeof(path), "%s/grad_%s.c", dir, cfg->name);
        if (write_built_file(path, build_grad_file, cfg) != 0) {
            return 1;
        }
    }
    if (cfg->custom_bwd) {
        (void)snprintf(path, sizeof(path), "%s/core_%s_bwd.c", dir, cfg->name);
        if (write_built_file(path, build_core_bwd, cfg) != 0) {
            return 1;
        }
        (void)snprintf(path, sizeof(path), "%s/cpu_%s_bwd.c", dir, cfg->name);
        if (write_built_file(path, build_cpu_bwd_wrapper, cfg) != 0) {
            return 1;
        }
    }
    if (!cfg->cpu_only) {
        (void)snprintf(path, sizeof(path), "%s/metal_%s_fwd.m", dir, cfg->name);
        if (write_built_file(path, build_metal_fwd_host_wrapper, cfg) != 0) {
            return 1;
        }
        (void)snprintf(path, sizeof(path), "%s/metal_%s_fwd.metal", dir, cfg->name);
        if (write_built_file(path, build_metal_fwd_shader_wrapper, cfg) != 0) {
            return 1;
        }
        if (cfg->custom_bwd) {
            (void)snprintf(path, sizeof(path), "%s/metal_%s_bwd.m", dir, cfg->name);
            if (write_built_file(path, build_metal_bwd_host_wrapper, cfg) != 0) {
                return 1;
            }
            (void)snprintf(path, sizeof(path), "%s/metal_%s_bwd.metal", dir, cfg->name);
            if (write_built_file(path, build_metal_bwd_shader_wrapper, cfg) != 0) {
                return 1;
            }
        }
    }
    return 0;
}

static int apply_patch_from_builder(const char *path,
                                    const char *marker,
                                    const char *already_token,
                                    int (*builder)(const op_config *, string_builder *),
                                    const op_config *cfg)
{
    string_builder sb = {0};
    int rc = 1;

    if (builder(cfg, &sb) != 0) {
        fprintf(stderr, "failed to build patch for %s\n", path);
        goto done;
    }
    if (patch_insert_before(path, marker, sb.data, already_token, cfg->dry_run) != 0) {
        goto done;
    }
    rc = 0;

done:
    sb_free(&sb);
    return rc;
}

static int patch_repo(const op_config *cfg)
{
    char token[GD_NEW_OP_MAX_NAME + 32];

    if (cfg->is_public) {
        (void)snprintf(token, sizeof(token), "gd_%s(", cfg->name);
        if (apply_patch_from_builder("include/gradients/ops.h",
                                     "gd_status gd_backward(gd_context *ctx, gd_tensor *loss);",
                                     token,
                                     build_ops_h_patch,
                                     cfg) != 0) {
            return 1;
        }
        (void)snprintf(token, sizeof(token), "GD_REF(gd_%s", cfg->name);
        if (apply_patch_from_builder("tests/test_public_symbols.c",
                                     "/* module.h */",
                                     token,
                                     build_public_symbols_patch,
                                     cfg) != 0) {
            return 1;
        }
    }

    (void)snprintf(token, sizeof(token), "_GD_OP_%s", cfg->upper);
    if (apply_patch_from_builder("tests/test_op_registry.c",
                                 "};\n\nstatic int test_registry_contents",
                                 token,
                                 build_registry_test_patch,
                                 cfg) != 0) {
        return 1;
    }

    if (!cfg->cpu_only) {
        if (apply_patch_from_builder("src/backends/metal/backend.m",
                                     "};\n\n/* Kernels not mapped 1:1 to an op",
                                     token,
                                     build_metal_backend_patch,
                                     cfg) != 0) {
            return 1;
        }
    }
    return 0;
}

static void print_summary(const op_config *cfg)
{
    printf("new op: %s\n", cfg->name);
    printf("  public: %s\n", cfg->is_public ? "yes" : "no");
    printf("  diff: %s\n", cfg->diff ? "yes" : "no");
    printf("  custom_bwd: %s\n", cfg->custom_bwd ? "yes" : "no");
    printf("  inputs: %d\n", cfg->inputs);
    printf("  outputs: %d\n", cfg->outputs);
    printf("  backends: cpu_ref%s\n", cfg->cpu_only ? "" : ", metal");
    printf("  generated: %s\n", cfg->run_generated && !cfg->dry_run ? "run make generated" : "skip");
}

static int run_make_generated(void)
{
    int rc = 0;

    printf("run: make generated\n");
    rc = system("make generated");
    if (rc != 0) {
        fprintf(stderr, "make generated failed\n");
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    op_config cfg;
    int parsed = 0;

    setvbuf(stdout, NULL, _IONBF, 0);
    parsed = parse_args(argc, argv, &cfg);

    if (parsed == 0) {
        return 0;
    }
    if (parsed != 1) {
        return parsed;
    }

    print_summary(&cfg);
    if (scaffold_files(&cfg) != 0) {
        return 1;
    }
    if (patch_repo(&cfg) != 0) {
        return 1;
    }
    if (cfg.run_generated && !cfg.dry_run) {
        if (run_make_generated() != 0) {
            return 1;
        }
    }
    printf("next: fill TODO stubs, then run `make build` and focused tests\n");
    return 0;
}
