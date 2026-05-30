#ifndef GRADIENTS_STATUS_H
#define GRADIENTS_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum gd_status {
    GD_OK = 0,
    GD_ERR_INVALID_ARGUMENT,
    GD_ERR_OUT_OF_MEMORY,
    GD_ERR_UNSUPPORTED,
    GD_ERR_BACKEND,
    GD_ERR_DTYPE,
    GD_ERR_SHAPE,
    GD_ERR_DEVICE,
    GD_ERR_INVALID_STATE,
    GD_ERR_IO,
    GD_ERR_INTERNAL
} gd_status;

const char *gd_status_name(gd_status status);
const char *gd_status_message(gd_status status);
const char *gd_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_STATUS_H */
