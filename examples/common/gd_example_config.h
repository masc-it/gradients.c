#ifndef GD_EXAMPLE_CONFIG_H
#define GD_EXAMPLE_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Minimal reusable reader for example runtime configs.  It intentionally
 * supports only flat YAML mappings with scalar values (`key: value`) so C
 * examples can avoid large argv surfaces without adding a third-party parser.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_example_config_error {
    unsigned line;
    char message[256];
} gd_example_config_error;

typedef struct gd_example_config_entry {
    char *key;
    char *value;
    unsigned line;
} gd_example_config_entry;

typedef struct gd_example_config_doc {
    gd_example_config_entry *entries;
    size_t count;
} gd_example_config_doc;

void gd_example_config_doc_init(gd_example_config_doc *doc);
void gd_example_config_doc_free(gd_example_config_doc *doc);

int gd_example_config_load_yaml_file(const char *path,
                                     gd_example_config_doc *out,
                                     gd_example_config_error *error);

const gd_example_config_entry *gd_example_config_find(const gd_example_config_doc *doc,
                                                      const char *key);
const char *gd_example_config_get(const gd_example_config_doc *doc, const char *key);

int gd_example_config_validate_keys(const gd_example_config_doc *doc,
                                    const char *const *known_keys,
                                    size_t known_key_count,
                                    gd_example_config_error *error);

int gd_example_config_require_string(const gd_example_config_doc *doc,
                                     const char *key,
                                     const char **out,
                                     gd_example_config_error *error);
int gd_example_config_require_string_allow_empty(const gd_example_config_doc *doc,
                                                 const char *key,
                                                 const char **out,
                                                 gd_example_config_error *error);
int gd_example_config_require_i64(const gd_example_config_doc *doc,
                                  const char *key,
                                  int64_t min_value,
                                  int64_t max_value,
                                  int64_t *out,
                                  gd_example_config_error *error);
int gd_example_config_require_u64(const gd_example_config_doc *doc,
                                  const char *key,
                                  uint64_t max_value,
                                  uint64_t *out,
                                  gd_example_config_error *error);
int gd_example_config_require_int(const gd_example_config_doc *doc,
                                  const char *key,
                                  int min_value,
                                  int max_value,
                                  int *out,
                                  gd_example_config_error *error);
int gd_example_config_require_f32(const gd_example_config_doc *doc,
                                  const char *key,
                                  float min_value,
                                  float max_value,
                                  float *out,
                                  gd_example_config_error *error);
int gd_example_config_require_bool(const gd_example_config_doc *doc,
                                   const char *key,
                                   bool *out,
                                   gd_example_config_error *error);

const char *gd_example_config_error_message(const gd_example_config_error *error);

#ifdef __cplusplus
}
#endif

#endif /* GD_EXAMPLE_CONFIG_H */
