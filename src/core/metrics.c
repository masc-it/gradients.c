#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <gradients/metrics.h>

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

typedef struct gd_metrics_json_buffer {
    char *data;
    size_t len;
    size_t cap;
} gd_metrics_json_buffer;

struct gd_metrics_logger {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_t thread;
    bool mutex_initialized;
    bool cond_initialized;
    bool thread_started;
    bool stop;
    char **queue;
    size_t capacity;
    size_t head;
    size_t tail;
    size_t count;
    FILE *file;
    char *root_dir;
    char *project;
    char *run_id;
    char *path;
    size_t max_record_bytes;
    unsigned int flush_every_records;
    atomic_uint_fast64_t dropped;
};

static char *gd_metrics_strdup(const char *text)
{
    size_t len;
    char *copy;
    if (text == NULL) {
        return NULL;
    }
    len = strlen(text);
    copy = (char *)malloc(len + 1U);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, text, len + 1U);
    return copy;
}

static bool gd_metrics_valid_segment(const char *text)
{
    size_t i;
    if (text == NULL || text[0] == '\0' || strcmp(text, ".") == 0 || strcmp(text, "..") == 0) {
        return false;
    }
    for (i = 0U; text[i] != '\0'; ++i) {
        const unsigned char ch = (unsigned char)text[i];
        if (!(isalnum(ch) != 0 || ch == '_' || ch == '-' || ch == '.')) {
            return false;
        }
    }
    return true;
}

static gd_status gd_metrics_join2(const char *left, const char *right, char **out)
{
    const size_t left_len = left != NULL ? strlen(left) : 0U;
    const size_t right_len = right != NULL ? strlen(right) : 0U;
    const bool need_sep = left_len > 0U && left[left_len - 1U] != '/';
    char *joined;
    size_t len;
    if (left == NULL || right == NULL || out == NULL || left_len == 0U || right_len == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (left_len > SIZE_MAX - right_len - (need_sep ? 2U : 1U)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    len = left_len + right_len + (need_sep ? 1U : 0U);
    joined = (char *)malloc(len + 1U);
    if (joined == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    memcpy(joined, left, left_len);
    if (need_sep) {
        joined[left_len] = '/';
        memcpy(joined + left_len + 1U, right, right_len);
    } else {
        memcpy(joined + left_len, right, right_len);
    }
    joined[len] = '\0';
    *out = joined;
    return GD_OK;
}

static gd_status gd_metrics_make_filename(const char *run_id, char **out)
{
    const char *suffix = ".jsonl";
    const size_t run_len = run_id != NULL ? strlen(run_id) : 0U;
    const size_t suffix_len = strlen(suffix);
    char *filename;
    if (run_len == 0U || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (run_len > SIZE_MAX - suffix_len - 1U) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    filename = (char *)malloc(run_len + suffix_len + 1U);
    if (filename == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    memcpy(filename, run_id, run_len);
    memcpy(filename + run_len, suffix, suffix_len + 1U);
    *out = filename;
    return GD_OK;
}

static gd_status gd_metrics_mkdirs(const char *path)
{
    char *copy;
    size_t i;
    if (path == NULL || path[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    copy = gd_metrics_strdup(path);
    if (copy == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    for (i = 1U; copy[i] != '\0'; ++i) {
        if (copy[i] == '/') {
            copy[i] = '\0';
            if (copy[0] != '\0' && mkdir(copy, 0775) != 0 && errno != EEXIST) {
                free(copy);
                return GD_ERR_IO;
            }
            copy[i] = '/';
        }
    }
    if (mkdir(copy, 0775) != 0 && errno != EEXIST) {
        free(copy);
        return GD_ERR_IO;
    }
    free(copy);
    return GD_OK;
}

static gd_status gd_metrics_generate_run_id(char **out)
{
    time_t raw_time;
    struct tm tm_value;
    char buffer[128];
    int n;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    raw_time = time(NULL);
    if (raw_time == (time_t)-1) {
        return GD_ERR_IO;
    }
    if (localtime_r(&raw_time, &tm_value) == NULL) {
        return GD_ERR_IO;
    }
    n = snprintf(buffer,
                 sizeof(buffer),
                 "%04d%02d%02d-%02d%02d%02d-%llu",
                 tm_value.tm_year + 1900,
                 tm_value.tm_mon + 1,
                 tm_value.tm_mday,
                 tm_value.tm_hour,
                 tm_value.tm_min,
                 tm_value.tm_sec,
                 (unsigned long long)getpid());
    if (n < 0 || (size_t)n >= sizeof(buffer)) {
        return GD_ERR_INTERNAL;
    }
    *out = gd_metrics_strdup(buffer);
    return *out != NULL ? GD_OK : GD_ERR_OUT_OF_MEMORY;
}

static double gd_metrics_wall_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        return (double)time(NULL);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void gd_metrics_json_buffer_free(gd_metrics_json_buffer *buffer)
{
    if (buffer == NULL) {
        return;
    }
    free(buffer->data);
    buffer->data = NULL;
    buffer->len = 0U;
    buffer->cap = 0U;
}

static gd_status gd_metrics_json_reserve(gd_metrics_json_buffer *buffer, size_t add_len)
{
    size_t required;
    size_t new_cap;
    char *new_data;
    if (buffer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (add_len > SIZE_MAX - buffer->len - 1U) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    required = buffer->len + add_len + 1U;
    if (required <= buffer->cap) {
        return GD_OK;
    }
    new_cap = buffer->cap != 0U ? buffer->cap : 256U;
    while (new_cap < required) {
        if (new_cap > SIZE_MAX / 2U) {
            new_cap = required;
            break;
        }
        new_cap *= 2U;
    }
    new_data = (char *)realloc(buffer->data, new_cap);
    if (new_data == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    buffer->data = new_data;
    buffer->cap = new_cap;
    return GD_OK;
}

static gd_status gd_metrics_json_append_bytes(gd_metrics_json_buffer *buffer,
                                              const char *text,
                                              size_t len)
{
    gd_status st;
    if (text == NULL && len != 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metrics_json_reserve(buffer, len);
    if (st != GD_OK) {
        return st;
    }
    if (len != 0U) {
        memcpy(buffer->data + buffer->len, text, len);
    }
    buffer->len += len;
    buffer->data[buffer->len] = '\0';
    return GD_OK;
}

static gd_status gd_metrics_json_append_text(gd_metrics_json_buffer *buffer, const char *text)
{
    if (text == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_metrics_json_append_bytes(buffer, text, strlen(text));
}

static gd_status gd_metrics_json_append_char(gd_metrics_json_buffer *buffer, char ch)
{
    return gd_metrics_json_append_bytes(buffer, &ch, 1U);
}

static gd_status gd_metrics_json_append_escaped(gd_metrics_json_buffer *buffer, const char *text)
{
    static const char hex[] = "0123456789abcdef";
    gd_status st;
    size_t i;
    if (buffer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metrics_json_append_char(buffer, '"');
    if (st != GD_OK) {
        return st;
    }
    if (text != NULL) {
        for (i = 0U; text[i] != '\0'; ++i) {
            const unsigned char ch = (unsigned char)text[i];
            char escape[6];
            switch (ch) {
            case '"': st = gd_metrics_json_append_text(buffer, "\\\""); break;
            case '\\': st = gd_metrics_json_append_text(buffer, "\\\\"); break;
            case '\b': st = gd_metrics_json_append_text(buffer, "\\b"); break;
            case '\f': st = gd_metrics_json_append_text(buffer, "\\f"); break;
            case '\n': st = gd_metrics_json_append_text(buffer, "\\n"); break;
            case '\r': st = gd_metrics_json_append_text(buffer, "\\r"); break;
            case '\t': st = gd_metrics_json_append_text(buffer, "\\t"); break;
            default:
                if (ch < 0x20U) {
                    escape[0] = '\\';
                    escape[1] = 'u';
                    escape[2] = '0';
                    escape[3] = '0';
                    escape[4] = hex[(unsigned int)ch >> 4U];
                    escape[5] = hex[(unsigned int)ch & 0x0fU];
                    st = gd_metrics_json_append_bytes(buffer, escape, sizeof(escape));
                } else {
                    st = gd_metrics_json_append_char(buffer, (char)ch);
                }
                break;
            }
            if (st != GD_OK) {
                return st;
            }
        }
    }
    return gd_metrics_json_append_char(buffer, '"');
}

static gd_status gd_metrics_json_append_key(gd_metrics_json_buffer *buffer,
                                            const char *key,
                                            bool *first)
{
    gd_status st;
    if (key == NULL || key[0] == '\0' || first == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!*first) {
        st = gd_metrics_json_append_char(buffer, ',');
        if (st != GD_OK) {
            return st;
        }
    }
    *first = false;
    st = gd_metrics_json_append_escaped(buffer, key);
    if (st != GD_OK) {
        return st;
    }
    return gd_metrics_json_append_char(buffer, ':');
}

static gd_status gd_metrics_json_append_u64(gd_metrics_json_buffer *buffer, uint64_t value)
{
    char temp[32];
    int n = snprintf(temp, sizeof(temp), "%llu", (unsigned long long)value);
    if (n < 0 || (size_t)n >= sizeof(temp)) {
        return GD_ERR_INTERNAL;
    }
    return gd_metrics_json_append_bytes(buffer, temp, (size_t)n);
}

static gd_status gd_metrics_json_append_i64(gd_metrics_json_buffer *buffer, int64_t value)
{
    char temp[32];
    int n = snprintf(temp, sizeof(temp), "%lld", (long long)value);
    if (n < 0 || (size_t)n >= sizeof(temp)) {
        return GD_ERR_INTERNAL;
    }
    return gd_metrics_json_append_bytes(buffer, temp, (size_t)n);
}

static gd_status gd_metrics_json_append_f64(gd_metrics_json_buffer *buffer, double value)
{
    char temp[64];
    int n;
    if (!isfinite(value)) {
        return gd_metrics_json_append_text(buffer, "null");
    }
    n = snprintf(temp, sizeof(temp), "%.17g", value);
    if (n < 0 || (size_t)n >= sizeof(temp)) {
        return GD_ERR_INTERNAL;
    }
    return gd_metrics_json_append_bytes(buffer, temp, (size_t)n);
}

static gd_status gd_metrics_json_append_value(gd_metrics_json_buffer *buffer,
                                              const gd_metrics_value *value)
{
    if (buffer == NULL || value == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    switch (value->kind) {
    case GD_METRICS_VALUE_NULL: return gd_metrics_json_append_text(buffer, "null");
    case GD_METRICS_VALUE_BOOL: return gd_metrics_json_append_text(buffer, value->as.b ? "true" : "false");
    case GD_METRICS_VALUE_I64: return gd_metrics_json_append_i64(buffer, value->as.i64);
    case GD_METRICS_VALUE_U64: return gd_metrics_json_append_u64(buffer, value->as.u64);
    case GD_METRICS_VALUE_F64: return gd_metrics_json_append_f64(buffer, value->as.f64);
    case GD_METRICS_VALUE_STRING:
        if (value->as.string == NULL) {
            return gd_metrics_json_append_text(buffer, "null");
        }
        return gd_metrics_json_append_escaped(buffer, value->as.string);
    default: return GD_ERR_INVALID_ARGUMENT;
    }
}

static void gd_metrics_record_drop(gd_metrics_logger *logger)
{
    if (logger != NULL) {
        (void)atomic_fetch_add_explicit(&logger->dropped, (uint_fast64_t)1U, memory_order_relaxed);
    }
}

static gd_status gd_metrics_enqueue_owned(gd_metrics_logger *logger, char *record)
{
    int lock_status;
    if (logger == NULL || record == NULL) {
        free(record);
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (strlen(record) > logger->max_record_bytes) {
        free(record);
        return GD_ERR_INVALID_ARGUMENT;
    }
    lock_status = pthread_mutex_trylock(&logger->mutex);
    if (lock_status != 0) {
        gd_metrics_record_drop(logger);
        free(record);
        return GD_ERR_BUSY;
    }
    if (logger->stop || logger->count == logger->capacity) {
        pthread_mutex_unlock(&logger->mutex);
        gd_metrics_record_drop(logger);
        free(record);
        return GD_ERR_BUSY;
    }
    logger->queue[logger->tail] = record;
    logger->tail = (logger->tail + 1U) % logger->capacity;
    logger->count += 1U;
    pthread_cond_signal(&logger->cond);
    pthread_mutex_unlock(&logger->mutex);
    return GD_OK;
}

static char *gd_metrics_pop_locked(gd_metrics_logger *logger)
{
    char *record;
    if (logger == NULL || logger->count == 0U) {
        return NULL;
    }
    record = logger->queue[logger->head];
    logger->queue[logger->head] = NULL;
    logger->head = (logger->head + 1U) % logger->capacity;
    logger->count -= 1U;
    return record;
}

static void *gd_metrics_writer_main(void *user_data)
{
    gd_metrics_logger *logger = (gd_metrics_logger *)user_data;
    unsigned int records_since_flush = 0U;
    pthread_mutex_lock(&logger->mutex);
    for (;;) {
        char *record;
        while (!logger->stop && logger->count == 0U) {
            pthread_cond_wait(&logger->cond, &logger->mutex);
        }
        if (logger->stop && logger->count == 0U) {
            break;
        }
        record = gd_metrics_pop_locked(logger);
        pthread_mutex_unlock(&logger->mutex);
        if (record != NULL) {
            if (fputs(record, logger->file) >= 0 && fputc('\n', logger->file) != EOF) {
                records_since_flush += 1U;
                if (logger->flush_every_records == 0U || records_since_flush >= logger->flush_every_records) {
                    (void)fflush(logger->file);
                    records_since_flush = 0U;
                }
            }
            free(record);
        }
        pthread_mutex_lock(&logger->mutex);
    }
    pthread_mutex_unlock(&logger->mutex);
    (void)fflush(logger->file);
    return NULL;
}

static void gd_metrics_logger_cleanup(gd_metrics_logger *logger)
{
    size_t i;
    if (logger == NULL) {
        return;
    }
    if (logger->queue != NULL) {
        for (i = 0U; i < logger->capacity; ++i) {
            free(logger->queue[i]);
        }
    }
    free(logger->queue);
    if (logger->file != NULL) {
        (void)fclose(logger->file);
    }
    if (logger->cond_initialized) {
        pthread_cond_destroy(&logger->cond);
    }
    if (logger->mutex_initialized) {
        pthread_mutex_destroy(&logger->mutex);
    }
    free(logger->root_dir);
    free(logger->project);
    free(logger->run_id);
    free(logger->path);
    free(logger);
}

gd_metrics_config gd_metrics_config_default(const char *project)
{
    gd_metrics_config config;
    config.root_dir = "data/metrics";
    config.project = project != NULL ? project : "default";
    config.run_id = NULL;
    config.queue_capacity = 1024U;
    config.max_record_bytes = 16384U;
    config.flush_every_records = 1U;
    return config;
}

gd_status gd_metrics_logger_start(const gd_metrics_config *config, gd_metrics_logger **out)
{
    const char *root_dir;
    const char *project;
    gd_metrics_logger *logger;
    char *project_dir = NULL;
    char *filename = NULL;
    gd_status st;
    int pthread_status;
    if (config == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    root_dir = config->root_dir != NULL && config->root_dir[0] != '\0' ? config->root_dir : "data/metrics";
    project = config->project != NULL && config->project[0] != '\0' ? config->project : "default";
    if (!gd_metrics_valid_segment(project) ||
        (config->run_id != NULL && !gd_metrics_valid_segment(config->run_id))) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    logger = (gd_metrics_logger *)calloc(1U, sizeof(*logger));
    if (logger == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    atomic_init(&logger->dropped, (uint_fast64_t)0U);
    logger->capacity = config->queue_capacity != 0U ? config->queue_capacity : 1024U;
    logger->max_record_bytes = config->max_record_bytes != 0U ? config->max_record_bytes : 16384U;
    logger->flush_every_records = config->flush_every_records != 0U ? config->flush_every_records : 1U;
    logger->root_dir = gd_metrics_strdup(root_dir);
    logger->project = gd_metrics_strdup(project);
    if (config->run_id != NULL) {
        logger->run_id = gd_metrics_strdup(config->run_id);
        st = logger->run_id != NULL ? GD_OK : GD_ERR_OUT_OF_MEMORY;
    } else {
        st = gd_metrics_generate_run_id(&logger->run_id);
    }
    if (logger->root_dir == NULL || logger->project == NULL || st != GD_OK) {
        gd_metrics_logger_cleanup(logger);
        return logger->root_dir == NULL || logger->project == NULL ? GD_ERR_OUT_OF_MEMORY : st;
    }
    logger->queue = (char **)calloc(logger->capacity, sizeof(logger->queue[0]));
    if (logger->queue == NULL) {
        gd_metrics_logger_cleanup(logger);
        return GD_ERR_OUT_OF_MEMORY;
    }
    st = gd_metrics_join2(logger->root_dir, logger->project, &project_dir);
    if (st == GD_OK) {
        st = gd_metrics_mkdirs(project_dir);
    }
    if (st == GD_OK) {
        st = gd_metrics_make_filename(logger->run_id, &filename);
    }
    if (st == GD_OK) {
        st = gd_metrics_join2(project_dir, filename, &logger->path);
    }
    free(project_dir);
    free(filename);
    if (st != GD_OK) {
        gd_metrics_logger_cleanup(logger);
        return st;
    }
    logger->file = fopen(logger->path, "ab");
    if (logger->file == NULL) {
        gd_metrics_logger_cleanup(logger);
        return GD_ERR_IO;
    }
    pthread_status = pthread_mutex_init(&logger->mutex, NULL);
    if (pthread_status != 0) {
        gd_metrics_logger_cleanup(logger);
        return GD_ERR_INTERNAL;
    }
    logger->mutex_initialized = true;
    pthread_status = pthread_cond_init(&logger->cond, NULL);
    if (pthread_status != 0) {
        gd_metrics_logger_cleanup(logger);
        return GD_ERR_INTERNAL;
    }
    logger->cond_initialized = true;
    pthread_status = pthread_create(&logger->thread, NULL, gd_metrics_writer_main, logger);
    if (pthread_status != 0) {
        gd_metrics_logger_cleanup(logger);
        return GD_ERR_INTERNAL;
    }
    logger->thread_started = true;
    *out = logger;
    return GD_OK;
}

gd_status gd_metrics_logger_log_json(gd_metrics_logger *logger, const char *json_object)
{
    size_t len;
    size_t i;
    char *copy;
    if (logger == NULL || json_object == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    len = strlen(json_object);
    if (len == 0U || len > logger->max_record_bytes) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < len; ++i) {
        if (json_object[i] == '\n' || json_object[i] == '\r') {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    copy = (char *)malloc(len + 1U);
    if (copy == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    memcpy(copy, json_object, len + 1U);
    return gd_metrics_enqueue_owned(logger, copy);
}

gd_status gd_metrics_logger_log_event(gd_metrics_logger *logger,
                                      const char *event,
                                      const gd_metrics_field *fields,
                                      size_t n_fields)
{
    gd_metrics_json_buffer buffer;
    bool first = true;
    size_t i;
    gd_status st;
    gd_metrics_value value;
    if (logger == NULL || event == NULL || event[0] == '\0' || (fields == NULL && n_fields != 0U)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(&buffer, 0, sizeof(buffer));
    st = gd_metrics_json_append_char(&buffer, '{');
    if (st == GD_OK) {
        st = gd_metrics_json_append_key(&buffer, "v", &first);
    }
    if (st == GD_OK) {
        st = gd_metrics_json_append_u64(&buffer, 1U);
    }
    if (st == GD_OK) {
        st = gd_metrics_json_append_key(&buffer, "time", &first);
    }
    if (st == GD_OK) {
        st = gd_metrics_json_append_f64(&buffer, gd_metrics_wall_seconds());
    }
    if (st == GD_OK) {
        st = gd_metrics_json_append_key(&buffer, "project", &first);
    }
    if (st == GD_OK) {
        st = gd_metrics_json_append_escaped(&buffer, logger->project);
    }
    if (st == GD_OK) {
        st = gd_metrics_json_append_key(&buffer, "run_id", &first);
    }
    if (st == GD_OK) {
        st = gd_metrics_json_append_escaped(&buffer, logger->run_id);
    }
    if (st == GD_OK) {
        st = gd_metrics_json_append_key(&buffer, "event", &first);
    }
    if (st == GD_OK) {
        st = gd_metrics_json_append_escaped(&buffer, event);
    }
    for (i = 0U; st == GD_OK && i < n_fields; ++i) {
        if (fields[i].key == NULL || fields[i].key[0] == '\0') {
            st = GD_ERR_INVALID_ARGUMENT;
            break;
        }
        st = gd_metrics_json_append_key(&buffer, fields[i].key, &first);
        if (st == GD_OK) {
            value = fields[i].value;
            st = gd_metrics_json_append_value(&buffer, &value);
        }
    }
    if (st == GD_OK) {
        st = gd_metrics_json_append_char(&buffer, '}');
    }
    if (st != GD_OK || buffer.len > logger->max_record_bytes) {
        gd_metrics_json_buffer_free(&buffer);
        return st != GD_OK ? st : GD_ERR_INVALID_ARGUMENT;
    }
    return gd_metrics_enqueue_owned(logger, buffer.data);
}

void gd_metrics_logger_stop(gd_metrics_logger *logger)
{
    if (logger == NULL) {
        return;
    }
    if (logger->thread_started) {
        pthread_mutex_lock(&logger->mutex);
        logger->stop = true;
        pthread_cond_signal(&logger->cond);
        pthread_mutex_unlock(&logger->mutex);
        pthread_join(logger->thread, NULL);
        logger->thread_started = false;
    }
    gd_metrics_logger_cleanup(logger);
}

const char *gd_metrics_logger_path(const gd_metrics_logger *logger)
{
    return logger != NULL ? logger->path : NULL;
}

const char *gd_metrics_logger_project(const gd_metrics_logger *logger)
{
    return logger != NULL ? logger->project : NULL;
}

const char *gd_metrics_logger_run_id(const gd_metrics_logger *logger)
{
    return logger != NULL ? logger->run_id : NULL;
}

uint64_t gd_metrics_logger_dropped(const gd_metrics_logger *logger)
{
    if (logger == NULL) {
        return 0U;
    }
    return (uint64_t)atomic_load_explicit(&logger->dropped, memory_order_relaxed);
}
