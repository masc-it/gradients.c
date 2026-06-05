#ifndef GRADIENTS_STATUS_H
#define GRADIENTS_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum gd_status {
    GD_OK = 0,
    GD_ERR_INVALID_ARGUMENT = -1,
    GD_ERR_OUT_OF_MEMORY = -2,
    GD_ERR_FROZEN = -3,
    GD_ERR_BUSY = -4,
    GD_ERR_BAD_STATE = -5,
    GD_ERR_UNSUPPORTED = -6,
    GD_ERR_INTERNAL = -7,
    GD_ERR_NOT_IMPLEMENTED = -8,
    GD_ERR_IO = -9,
} gd_status;

const char *gd_status_string(gd_status status);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_STATUS_H */
