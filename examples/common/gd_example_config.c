#include "gd_example_config.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void gd_example_config_error_clear(gd_example_config_error *error)
{
    if (error == NULL) {
        return;
    }
    error->line = 0U;
    error->message[0] = '\0';
}

static void gd_example_config_set_error(gd_example_config_error *error,
                                        unsigned line,
                                        const char *fmt,
                                        ...)
{
    va_list ap;
    if (error == NULL) {
        return;
    }
    error->line = line;
    va_start(ap, fmt);
    (void)vsnprintf(error->message, sizeof(error->message), fmt, ap);
    va_end(ap);
}

const char *gd_example_config_error_message(const gd_example_config_error *error)
{
    if (error == NULL || error->message[0] == '\0') {
        return "unknown configuration error";
    }
    return error->message;
}

void gd_example_config_doc_init(gd_example_config_doc *doc)
{
    if (doc == NULL) {
        return;
    }
    doc->entries = NULL;
    doc->count = 0U;
}

void gd_example_config_doc_free(gd_example_config_doc *doc)
{
    size_t i;
    if (doc == NULL) {
        return;
    }
    for (i = 0U; i < doc->count; ++i) {
        free(doc->entries[i].key);
        free(doc->entries[i].value);
    }
    free(doc->entries);
    gd_example_config_doc_init(doc);
}

static char *gd_example_config_strdup_range(const char *text, size_t len)
{
    char *out;
    if (text == NULL || len > SIZE_MAX - 1U) {
        return NULL;
    }
    out = (char *)malloc(len + 1U);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, text, len);
    out[len] = '\0';
    return out;
}

static char *gd_example_config_strdup(const char *text)
{
    if (text == NULL) {
        return NULL;
    }
    return gd_example_config_strdup_range(text, strlen(text));
}

static char *gd_example_config_ltrim(char *text)
{
    if (text == NULL) {
        return NULL;
    }
    while (*text != '\0' && isspace((unsigned char)*text)) {
        ++text;
    }
    return text;
}

static void gd_example_config_rtrim_in_place(char *text)
{
    size_t len;
    if (text == NULL) {
        return;
    }
    len = strlen(text);
    while (len > 0U && isspace((unsigned char)text[len - 1U])) {
        text[len - 1U] = '\0';
        --len;
    }
}

static char *gd_example_config_trim(char *text)
{
    char *out = gd_example_config_ltrim(text);
    gd_example_config_rtrim_in_place(out);
    return out;
}

static int gd_example_config_is_key_char(char ch)
{
    return isalnum((unsigned char)ch) || ch == '_' || ch == '-' || ch == '.';
}

static int gd_example_config_validate_key(const char *key)
{
    size_t i;
    if (key == NULL || key[0] == '\0') {
        return 0;
    }
    for (i = 0U; key[i] != '\0'; ++i) {
        if (!gd_example_config_is_key_char(key[i])) {
            return 0;
        }
    }
    return 1;
}

static int gd_example_config_line_is_marker(const char *text, const char *marker)
{
    const size_t marker_len = strlen(marker);
    if (strncmp(text, marker, marker_len) != 0) {
        return 0;
    }
    text += marker_len;
    while (*text != '\0') {
        if (!isspace((unsigned char)*text)) {
            return 0;
        }
        ++text;
    }
    return 1;
}

static void gd_example_config_strip_comment(char *line)
{
    int in_single = 0;
    int in_double = 0;
    int escaped = 0;
    size_t i;
    if (line == NULL) {
        return;
    }
    for (i = 0U; line[i] != '\0'; ++i) {
        const char ch = line[i];
        if (escaped) {
            escaped = 0;
            continue;
        }
        if (in_double && ch == '\\') {
            escaped = 1;
            continue;
        }
        if (!in_double && ch == '\'') {
            in_single = !in_single;
            continue;
        }
        if (!in_single && ch == '"') {
            in_double = !in_double;
            continue;
        }
        if (!in_single && !in_double && ch == '#') {
            line[i] = '\0';
            return;
        }
    }
}

static char *gd_example_config_find_colon(char *line)
{
    int in_single = 0;
    int in_double = 0;
    int escaped = 0;
    size_t i;
    if (line == NULL) {
        return NULL;
    }
    for (i = 0U; line[i] != '\0'; ++i) {
        const char ch = line[i];
        if (escaped) {
            escaped = 0;
            continue;
        }
        if (in_double && ch == '\\') {
            escaped = 1;
            continue;
        }
        if (!in_double && ch == '\'') {
            in_single = !in_single;
            continue;
        }
        if (!in_single && ch == '"') {
            in_double = !in_double;
            continue;
        }
        if (!in_single && !in_double && ch == ':') {
            return &line[i];
        }
    }
    return NULL;
}

static int gd_example_config_trailing_is_empty(const char *text)
{
    while (text != NULL && *text != '\0') {
        if (!isspace((unsigned char)*text)) {
            return 0;
        }
        ++text;
    }
    return 1;
}

static int gd_example_config_decode_single_quoted(const char *text,
                                                  unsigned line,
                                                  char **out,
                                                  gd_example_config_error *error)
{
    const size_t len = strlen(text);
    char *value = (char *)malloc(len + 1U);
    size_t read_index = 1U;
    size_t write_index = 0U;
    int closed = 0;
    if (value == NULL) {
        gd_example_config_set_error(error, line, "out of memory while parsing single-quoted scalar");
        return 0;
    }
    while (text[read_index] != '\0') {
        if (text[read_index] == '\'') {
            if (text[read_index + 1U] == '\'') {
                value[write_index++] = '\'';
                read_index += 2U;
                continue;
            }
            closed = 1;
            ++read_index;
            break;
        }
        value[write_index++] = text[read_index++];
    }
    if (!closed) {
        free(value);
        gd_example_config_set_error(error, line, "unterminated single-quoted scalar");
        return 0;
    }
    if (!gd_example_config_trailing_is_empty(text + read_index)) {
        free(value);
        gd_example_config_set_error(error, line, "unexpected text after single-quoted scalar");
        return 0;
    }
    value[write_index] = '\0';
    *out = value;
    return 1;
}

static int gd_example_config_decode_double_quoted(const char *text,
                                                  unsigned line,
                                                  char **out,
                                                  gd_example_config_error *error)
{
    const size_t len = strlen(text);
    char *value = (char *)malloc(len + 1U);
    size_t read_index = 1U;
    size_t write_index = 0U;
    int closed = 0;
    if (value == NULL) {
        gd_example_config_set_error(error, line, "out of memory while parsing double-quoted scalar");
        return 0;
    }
    while (text[read_index] != '\0') {
        char ch = text[read_index++];
        if (ch == '"') {
            closed = 1;
            break;
        }
        if (ch == '\\') {
            ch = text[read_index++];
            switch (ch) {
            case '\0':
                free(value);
                gd_example_config_set_error(error, line, "unterminated escape in double-quoted scalar");
                return 0;
            case 'n': ch = '\n'; break;
            case 'r': ch = '\r'; break;
            case 't': ch = '\t'; break;
            case '"': ch = '"'; break;
            case '\\': ch = '\\'; break;
            default:
                free(value);
                gd_example_config_set_error(error, line, "unsupported escape \\%c in double-quoted scalar", ch);
                return 0;
            }
        }
        value[write_index++] = ch;
    }
    if (!closed) {
        free(value);
        gd_example_config_set_error(error, line, "unterminated double-quoted scalar");
        return 0;
    }
    if (!gd_example_config_trailing_is_empty(text + read_index)) {
        free(value);
        gd_example_config_set_error(error, line, "unexpected text after double-quoted scalar");
        return 0;
    }
    value[write_index] = '\0';
    *out = value;
    return 1;
}

static int gd_example_config_parse_scalar(const char *text,
                                          unsigned line,
                                          char **out,
                                          gd_example_config_error *error)
{
    char *copy;
    char *trimmed;
    if (text == NULL || out == NULL) {
        gd_example_config_set_error(error, line, "invalid scalar parse arguments");
        return 0;
    }
    *out = NULL;
    copy = gd_example_config_strdup(text);
    if (copy == NULL) {
        gd_example_config_set_error(error, line, "out of memory while parsing scalar");
        return 0;
    }
    trimmed = gd_example_config_trim(copy);
    if (trimmed[0] == '\0') {
        free(copy);
        gd_example_config_set_error(error, line, "missing value");
        return 0;
    }
    if (trimmed[0] == '\'') {
        const int ok = gd_example_config_decode_single_quoted(trimmed, line, out, error);
        free(copy);
        return ok;
    }
    if (trimmed[0] == '"') {
        const int ok = gd_example_config_decode_double_quoted(trimmed, line, out, error);
        free(copy);
        return ok;
    }
    *out = gd_example_config_strdup(trimmed);
    free(copy);
    if (*out == NULL) {
        gd_example_config_set_error(error, line, "out of memory while storing scalar");
        return 0;
    }
    return 1;
}

const gd_example_config_entry *gd_example_config_find(const gd_example_config_doc *doc,
                                                      const char *key)
{
    size_t i;
    if (doc == NULL || key == NULL) {
        return NULL;
    }
    for (i = 0U; i < doc->count; ++i) {
        if (strcmp(doc->entries[i].key, key) == 0) {
            return &doc->entries[i];
        }
    }
    return NULL;
}

const char *gd_example_config_get(const gd_example_config_doc *doc, const char *key)
{
    const gd_example_config_entry *entry = gd_example_config_find(doc, key);
    return entry != NULL ? entry->value : NULL;
}

static int gd_example_config_add_entry(gd_example_config_doc *doc,
                                       char *key,
                                       char *value,
                                       unsigned line,
                                       gd_example_config_error *error)
{
    gd_example_config_entry *items;
    if (gd_example_config_find(doc, key) != NULL) {
        gd_example_config_set_error(error, line, "duplicate key '%s'", key);
        return 0;
    }
    if (doc->count == SIZE_MAX / sizeof(doc->entries[0])) {
        gd_example_config_set_error(error, line, "too many configuration entries");
        return 0;
    }
    items = (gd_example_config_entry *)realloc(doc->entries,
                                               (doc->count + 1U) * sizeof(doc->entries[0]));
    if (items == NULL) {
        gd_example_config_set_error(error, line, "out of memory while storing configuration entry");
        return 0;
    }
    doc->entries = items;
    doc->entries[doc->count].key = key;
    doc->entries[doc->count].value = value;
    doc->entries[doc->count].line = line;
    doc->count += 1U;
    return 1;
}

static int gd_example_config_parse_line(gd_example_config_doc *doc,
                                        char *line,
                                        unsigned line_no,
                                        gd_example_config_error *error)
{
    char *trimmed;
    char *colon;
    char *key;
    char *value_text;
    char *key_copy = NULL;
    char *value_copy = NULL;
    gd_example_config_strip_comment(line);
    trimmed = gd_example_config_trim(line);
    if (trimmed[0] == '\0' || gd_example_config_line_is_marker(trimmed, "---") ||
        gd_example_config_line_is_marker(trimmed, "...")) {
        return 1;
    }
    colon = gd_example_config_find_colon(trimmed);
    if (colon == NULL) {
        gd_example_config_set_error(error, line_no, "expected a flat YAML mapping entry 'key: value'");
        return 0;
    }
    *colon = '\0';
    key = gd_example_config_trim(trimmed);
    value_text = colon + 1;
    if (!gd_example_config_validate_key(key)) {
        gd_example_config_set_error(error, line_no, "invalid key '%s'", key);
        return 0;
    }
    key_copy = gd_example_config_strdup(key);
    if (key_copy == NULL) {
        gd_example_config_set_error(error, line_no, "out of memory while storing key '%s'", key);
        return 0;
    }
    if (!gd_example_config_parse_scalar(value_text, line_no, &value_copy, error)) {
        free(key_copy);
        return 0;
    }
    if (!gd_example_config_add_entry(doc, key_copy, value_copy, line_no, error)) {
        free(key_copy);
        free(value_copy);
        return 0;
    }
    return 1;
}

static int gd_example_config_read_file(const char *path,
                                       char **buffer_out,
                                       gd_example_config_error *error)
{
    FILE *file;
    long file_size;
    size_t nread;
    char *buffer;
    if (path == NULL || path[0] == '\0' || buffer_out == NULL) {
        gd_example_config_set_error(error, 0U, "invalid configuration path");
        return 0;
    }
    *buffer_out = NULL;
    file = fopen(path, "rb");
    if (file == NULL) {
        gd_example_config_set_error(error, 0U, "failed to open '%s'", path);
        return 0;
    }
    if (fseek(file, 0L, SEEK_END) != 0) {
        (void)fclose(file);
        gd_example_config_set_error(error, 0U, "failed to seek '%s'", path);
        return 0;
    }
    file_size = ftell(file);
    if (file_size < 0L) {
        (void)fclose(file);
        gd_example_config_set_error(error, 0U, "failed to get size of '%s'", path);
        return 0;
    }
    if ((unsigned long)file_size > (unsigned long)(SIZE_MAX - 1U)) {
        (void)fclose(file);
        gd_example_config_set_error(error, 0U, "configuration file '%s' is too large", path);
        return 0;
    }
    if (fseek(file, 0L, SEEK_SET) != 0) {
        (void)fclose(file);
        gd_example_config_set_error(error, 0U, "failed to rewind '%s'", path);
        return 0;
    }
    buffer = (char *)malloc((size_t)file_size + 1U);
    if (buffer == NULL) {
        (void)fclose(file);
        gd_example_config_set_error(error, 0U, "out of memory while reading '%s'", path);
        return 0;
    }
    nread = fread(buffer, 1U, (size_t)file_size, file);
    if (nread != (size_t)file_size || ferror(file)) {
        free(buffer);
        (void)fclose(file);
        gd_example_config_set_error(error, 0U, "failed to read '%s'", path);
        return 0;
    }
    if (fclose(file) != 0) {
        free(buffer);
        gd_example_config_set_error(error, 0U, "failed to close '%s'", path);
        return 0;
    }
    buffer[nread] = '\0';
    *buffer_out = buffer;
    return 1;
}

int gd_example_config_load_yaml_file(const char *path,
                                     gd_example_config_doc *out,
                                     gd_example_config_error *error)
{
    char *buffer = NULL;
    const char *line_start;
    unsigned line_no = 1U;
    gd_example_config_error_clear(error);
    if (out == NULL) {
        gd_example_config_set_error(error, 0U, "missing configuration output");
        return 0;
    }
    gd_example_config_doc_init(out);
    if (!gd_example_config_read_file(path, &buffer, error)) {
        return 0;
    }
    line_start = buffer;
    while (*line_start != '\0') {
        const char *line_end = strchr(line_start, '\n');
        size_t line_len = line_end != NULL ? (size_t)(line_end - line_start) : strlen(line_start);
        char *line_copy;
        if (line_len > 0U && line_start[line_len - 1U] == '\r') {
            --line_len;
        }
        line_copy = gd_example_config_strdup_range(line_start, line_len);
        if (line_copy == NULL) {
            gd_example_config_set_error(error, line_no, "out of memory while parsing line");
            free(buffer);
            gd_example_config_doc_free(out);
            return 0;
        }
        if (!gd_example_config_parse_line(out, line_copy, line_no, error)) {
            free(line_copy);
            free(buffer);
            gd_example_config_doc_free(out);
            return 0;
        }
        free(line_copy);
        if (line_end == NULL) {
            break;
        }
        line_start = line_end + 1;
        ++line_no;
    }
    free(buffer);
    return 1;
}

int gd_example_config_validate_keys(const gd_example_config_doc *doc,
                                    const char *const *known_keys,
                                    size_t known_key_count,
                                    gd_example_config_error *error)
{
    size_t i;
    size_t j;
    if (doc == NULL || known_keys == NULL) {
        gd_example_config_set_error(error, 0U, "invalid known-key validation arguments");
        return 0;
    }
    for (i = 0U; i < doc->count; ++i) {
        int known = 0;
        for (j = 0U; j < known_key_count; ++j) {
            if (known_keys[j] != NULL && strcmp(doc->entries[i].key, known_keys[j]) == 0) {
                known = 1;
                break;
            }
        }
        if (!known) {
            gd_example_config_set_error(error,
                                        doc->entries[i].line,
                                        "unknown key '%s'",
                                        doc->entries[i].key);
            return 0;
        }
    }
    return 1;
}

static const gd_example_config_entry *gd_example_config_require_entry(const gd_example_config_doc *doc,
                                                                      const char *key,
                                                                      gd_example_config_error *error)
{
    const gd_example_config_entry *entry = gd_example_config_find(doc, key);
    if (entry == NULL) {
        gd_example_config_set_error(error, 0U, "missing required key '%s'", key);
    }
    return entry;
}

int gd_example_config_require_string(const gd_example_config_doc *doc,
                                     const char *key,
                                     const char **out,
                                     gd_example_config_error *error)
{
    const gd_example_config_entry *entry;
    if (out == NULL) {
        gd_example_config_set_error(error, 0U, "invalid output for key '%s'", key);
        return 0;
    }
    *out = NULL;
    entry = gd_example_config_require_entry(doc, key, error);
    if (entry == NULL) {
        return 0;
    }
    if (entry->value[0] == '\0') {
        gd_example_config_set_error(error, entry->line, "key '%s' must not be empty", key);
        return 0;
    }
    *out = entry->value;
    return 1;
}

int gd_example_config_require_i64(const gd_example_config_doc *doc,
                                  const char *key,
                                  int64_t min_value,
                                  int64_t max_value,
                                  int64_t *out,
                                  gd_example_config_error *error)
{
    const gd_example_config_entry *entry;
    char *end = NULL;
    long long parsed;
    if (out == NULL || min_value > max_value) {
        gd_example_config_set_error(error, 0U, "invalid integer bounds for key '%s'", key);
        return 0;
    }
    *out = 0;
    entry = gd_example_config_require_entry(doc, key, error);
    if (entry == NULL) {
        return 0;
    }
    errno = 0;
    parsed = strtoll(entry->value, &end, 0);
    if (errno != 0 || end == entry->value || !gd_example_config_trailing_is_empty(end) ||
        parsed < (long long)min_value || parsed > (long long)max_value) {
        gd_example_config_set_error(error,
                                    entry->line,
                                    "key '%s' must be an integer in [%lld, %lld]; got '%s'",
                                    key,
                                    (long long)min_value,
                                    (long long)max_value,
                                    entry->value);
        return 0;
    }
    *out = (int64_t)parsed;
    return 1;
}

int gd_example_config_require_u64(const gd_example_config_doc *doc,
                                  const char *key,
                                  uint64_t max_value,
                                  uint64_t *out,
                                  gd_example_config_error *error)
{
    const gd_example_config_entry *entry;
    char *end = NULL;
    unsigned long long parsed;
    if (out == NULL) {
        gd_example_config_set_error(error, 0U, "invalid unsigned integer output for key '%s'", key);
        return 0;
    }
    *out = 0U;
    entry = gd_example_config_require_entry(doc, key, error);
    if (entry == NULL) {
        return 0;
    }
    if (entry->value[0] == '-') {
        gd_example_config_set_error(error,
                                    entry->line,
                                    "key '%s' must be an unsigned integer; got '%s'",
                                    key,
                                    entry->value);
        return 0;
    }
    errno = 0;
    parsed = strtoull(entry->value, &end, 0);
    if (errno != 0 || end == entry->value || !gd_example_config_trailing_is_empty(end) ||
        (uint64_t)parsed > max_value) {
        gd_example_config_set_error(error,
                                    entry->line,
                                    "key '%s' must be an unsigned integer <= %llu; got '%s'",
                                    key,
                                    (unsigned long long)max_value,
                                    entry->value);
        return 0;
    }
    *out = (uint64_t)parsed;
    return 1;
}

int gd_example_config_require_int(const gd_example_config_doc *doc,
                                  const char *key,
                                  int min_value,
                                  int max_value,
                                  int *out,
                                  gd_example_config_error *error)
{
    int64_t parsed = 0;
    if (out == NULL) {
        gd_example_config_set_error(error, 0U, "invalid integer output for key '%s'", key);
        return 0;
    }
    if (!gd_example_config_require_i64(doc,
                                       key,
                                       (int64_t)min_value,
                                       (int64_t)max_value,
                                       &parsed,
                                       error)) {
        return 0;
    }
    *out = (int)parsed;
    return 1;
}

int gd_example_config_require_f32(const gd_example_config_doc *doc,
                                  const char *key,
                                  float min_value,
                                  float max_value,
                                  float *out,
                                  gd_example_config_error *error)
{
    const gd_example_config_entry *entry;
    char *end = NULL;
    float parsed;
    if (out == NULL || min_value > max_value) {
        gd_example_config_set_error(error, 0U, "invalid float bounds for key '%s'", key);
        return 0;
    }
    *out = 0.0f;
    entry = gd_example_config_require_entry(doc, key, error);
    if (entry == NULL) {
        return 0;
    }
    errno = 0;
    parsed = strtof(entry->value, &end);
    if (errno != 0 || end == entry->value || !gd_example_config_trailing_is_empty(end) ||
        !isfinite(parsed) || parsed < min_value || parsed > max_value) {
        gd_example_config_set_error(error,
                                    entry->line,
                                    "key '%s' must be a finite float in [%.9g, %.9g]; got '%s'",
                                    key,
                                    (double)min_value,
                                    (double)max_value,
                                    entry->value);
        return 0;
    }
    *out = parsed;
    return 1;
}

static int gd_example_config_ascii_equal_ci(const char *a, const char *b)
{
    size_t i;
    if (a == NULL || b == NULL) {
        return 0;
    }
    for (i = 0U; a[i] != '\0' || b[i] != '\0'; ++i) {
        const unsigned char ca = (unsigned char)a[i];
        const unsigned char cb = (unsigned char)b[i];
        if (tolower(ca) != tolower(cb)) {
            return 0;
        }
    }
    return 1;
}

int gd_example_config_require_bool(const gd_example_config_doc *doc,
                                   const char *key,
                                   bool *out,
                                   gd_example_config_error *error)
{
    const gd_example_config_entry *entry;
    if (out == NULL) {
        gd_example_config_set_error(error, 0U, "invalid boolean output for key '%s'", key);
        return 0;
    }
    *out = false;
    entry = gd_example_config_require_entry(doc, key, error);
    if (entry == NULL) {
        return 0;
    }
    if (gd_example_config_ascii_equal_ci(entry->value, "true") ||
        gd_example_config_ascii_equal_ci(entry->value, "yes") ||
        gd_example_config_ascii_equal_ci(entry->value, "on") || strcmp(entry->value, "1") == 0) {
        *out = true;
        return 1;
    }
    if (gd_example_config_ascii_equal_ci(entry->value, "false") ||
        gd_example_config_ascii_equal_ci(entry->value, "no") ||
        gd_example_config_ascii_equal_ci(entry->value, "off") || strcmp(entry->value, "0") == 0) {
        *out = false;
        return 1;
    }
    gd_example_config_set_error(error,
                                entry->line,
                                "key '%s' must be a boolean; got '%s'",
                                key,
                                entry->value);
    return 0;
}
