#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <gradients/gradients.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK_OK(expr)                                                        \
    do {                                                                      \
        gd_status _st = (expr);                                                \
        if (_st != GD_OK) {                                                    \
            fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,   \
                    gd_status_string(_st));                                    \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define CHECK_TRUE(expr)                                                      \
    do {                                                                      \
        if (!(expr)) {                                                        \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,  \
                    #expr);                                                   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static char *read_file(const char *path)
{
    FILE *file;
    long end_pos;
    size_t size;
    char *data;
    file = fopen(path, "rb");
    CHECK_TRUE(file != NULL);
    CHECK_TRUE(fseek(file, 0L, SEEK_END) == 0);
    end_pos = ftell(file);
    CHECK_TRUE(end_pos >= 0);
    CHECK_TRUE(fseek(file, 0L, SEEK_SET) == 0);
    size = (size_t)end_pos;
    data = (char *)malloc(size + 1U);
    CHECK_TRUE(data != NULL);
    if (size != 0U) {
        CHECK_TRUE(fread(data, 1U, size, file) == size);
    }
    data[size] = '\0';
    CHECK_TRUE(fclose(file) == 0);
    return data;
}

int main(void)
{
    gd_metrics_config config = gd_metrics_config_default("unit_test");
    gd_metrics_logger *logger = NULL;
    char run_id[64];
    char path[512];
    const char *logger_path;
    char *data;
    const gd_metrics_field fields[] = {
        gd_metrics_u64("step", 7U),
        gd_metrics_f64("loss", 0.25),
        gd_metrics_bool("ok", true),
        gd_metrics_string("note", "hello \"jsonl\""),
    };
    (void)snprintf(run_id, sizeof(run_id), "run-%llu", (unsigned long long)getpid());
    config.root_dir = "build/test_metrics";
    config.run_id = run_id;
    config.queue_capacity = 8U;
    config.flush_every_records = 1U;
    CHECK_OK(gd_metrics_logger_start(&config, &logger));
    logger_path = gd_metrics_logger_path(logger);
    CHECK_TRUE(logger_path != NULL);
    CHECK_TRUE(strlen(logger_path) < sizeof(path));
    (void)strcpy(path, logger_path);
    CHECK_TRUE(strcmp(gd_metrics_logger_project(logger), "unit_test") == 0);
    CHECK_TRUE(strcmp(gd_metrics_logger_run_id(logger), run_id) == 0);
    CHECK_OK(gd_metrics_logger_log_event(logger, "train", fields, GD_ARRAY_LEN(fields)));
    CHECK_OK(gd_metrics_logger_log_json(logger, "{\"v\":1,\"event\":\"raw\",\"value\":3}"));
    gd_metrics_logger_stop(logger);

    data = read_file(path);
    CHECK_TRUE(strstr(data, "\"project\":\"unit_test\"") != NULL);
    CHECK_TRUE(strstr(data, "\"run_id\":") != NULL);
    CHECK_TRUE(strstr(data, "\"event\":\"train\"") != NULL);
    CHECK_TRUE(strstr(data, "\"loss\":0.25") != NULL);
    CHECK_TRUE(strstr(data, "hello \\\"jsonl\\\"") != NULL);
    CHECK_TRUE(strstr(data, "\"event\":\"raw\"") != NULL);
    free(data);
    return 0;
}
