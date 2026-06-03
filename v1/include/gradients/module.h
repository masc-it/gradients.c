#ifndef GRADIENTS_MODULE_H
#define GRADIENTS_MODULE_H

#include <stdbool.h>

#include "gradients/context.h"
#include "gradients/status.h"
#include "gradients/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_module gd_module;

gd_status gd_module_create(gd_context *ctx, const char *type_name, gd_module **out);
void gd_module_destroy(gd_module *module);

gd_status gd_module_param(gd_module *module, const char *name, gd_tensor *param);
gd_status gd_module_child(gd_module *module, const char *name, gd_module *child);
gd_status gd_module_parameters(gd_module *module,
                               gd_tensor ***params_out,
                               int *n_out);
gd_status gd_module_zero_grad(gd_context *ctx, gd_module *module);

gd_status gd_module_save(gd_module *module, const char *path);
gd_status gd_module_load(gd_module *module, const char *path, bool strict);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_MODULE_H */
