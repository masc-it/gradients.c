#ifndef GRADIENTS_METRICS_H
#define GRADIENTS_METRICS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <gradients/status.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_metrics_logger gd_metrics_logger;

typedef struct gd_metrics_config {
    const char *root_dir;
    const char *project;
    const char *run_id;
    size_t queue_capacity;
    size_t max_record_bytes;
    unsigned int flush_every_records;
} gd_metrics_config;

typedef enum gd_metrics_value_kind {
    GD_METRICS_VALUE_NULL = 0,
    GD_METRICS_VALUE_BOOL,
    GD_METRICS_VALUE_I64,
    GD_METRICS_VALUE_U64,
    GD_METRICS_VALUE_F64,
    GD_METRICS_VALUE_STRING,
} gd_metrics_value_kind;

typedef struct gd_metrics_value {
    gd_metrics_value_kind kind;
    union {
        bool b;
        int64_t i64;
        uint64_t u64;
        double f64;
        const char *string;
    } as;
} gd_metrics_value;

typedef struct gd_metrics_field {
    const char *key;
    gd_metrics_value value;
} gd_metrics_field;

static inline gd_metrics_field gd_metrics_null(const char *key)
{
    gd_metrics_field field;
    field.key = key;
    field.value.kind = GD_METRICS_VALUE_NULL;
    field.value.as.u64 = 0U;
    return field;
}

static inline gd_metrics_field gd_metrics_bool(const char *key, bool value)
{
    gd_metrics_field field;
    field.key = key;
    field.value.kind = GD_METRICS_VALUE_BOOL;
    field.value.as.b = value;
    return field;
}

static inline gd_metrics_field gd_metrics_i64(const char *key, int64_t value)
{
    gd_metrics_field field;
    field.key = key;
    field.value.kind = GD_METRICS_VALUE_I64;
    field.value.as.i64 = value;
    return field;
}

static inline gd_metrics_field gd_metrics_u64(const char *key, uint64_t value)
{
    gd_metrics_field field;
    field.key = key;
    field.value.kind = GD_METRICS_VALUE_U64;
    field.value.as.u64 = value;
    return field;
}

static inline gd_metrics_field gd_metrics_f64(const char *key, double value)
{
    gd_metrics_field field;
    field.key = key;
    field.value.kind = GD_METRICS_VALUE_F64;
    field.value.as.f64 = value;
    return field;
}

static inline gd_metrics_field gd_metrics_string(const char *key, const char *value)
{
    gd_metrics_field field;
    field.key = key;
    field.value.kind = value != NULL ? GD_METRICS_VALUE_STRING : GD_METRICS_VALUE_NULL;
    field.value.as.string = value;
    return field;
}

gd_metrics_config gd_metrics_config_default(const char *project);
gd_status gd_metrics_logger_start(const gd_metrics_config *config, gd_metrics_logger **out);
gd_status gd_metrics_logger_log_json(gd_metrics_logger *logger, const char *json_object);
gd_status gd_metrics_logger_log_event(gd_metrics_logger *logger,
                                      const char *event,
                                      const gd_metrics_field *fields,
                                      size_t n_fields);
void gd_metrics_logger_stop(gd_metrics_logger *logger);
const char *gd_metrics_logger_path(const gd_metrics_logger *logger);
const char *gd_metrics_logger_project(const gd_metrics_logger *logger);
const char *gd_metrics_logger_run_id(const gd_metrics_logger *logger);
uint64_t gd_metrics_logger_dropped(const gd_metrics_logger *logger);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_METRICS_H */
