#include "gradients/device.h"

#include "internal.h"

bool gd_device_equal(gd_device a, gd_device b)
{
    return a.type == b.type && a.index == b.index;
}

const char *gd_device_type_name(gd_device_type type)
{
    switch (type) {
    case GD_DEVICE_CPU:
        return "CPU";
    case GD_DEVICE_METAL:
        return "METAL";
    case GD_DEVICE_CUDA:
        return "CUDA";
    case GD_DEVICE_VULKAN:
        return "VULKAN";
    }
    return "UNKNOWN";
}
