#ifndef GRADIENTS_DEVICE_H
#define GRADIENTS_DEVICE_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum gd_device_type {
    GD_DEVICE_CPU = 0,
    GD_DEVICE_METAL,
    GD_DEVICE_CUDA,
    GD_DEVICE_VULKAN
} gd_device_type;

typedef struct gd_device {
    gd_device_type type;
    int index;
} gd_device;

bool gd_device_equal(gd_device a, gd_device b);
const char *gd_device_type_name(gd_device_type type);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_DEVICE_H */
