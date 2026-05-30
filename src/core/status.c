#include "gradients/status.h"

#include <stdio.h>

#include "internal.h"

static _Thread_local char gd_last_error_buf[256];

const char *gd_status_name(gd_status status)
{
    switch (status) {
    case GD_OK:
        return "GD_OK";
    case GD_ERR_INVALID_ARGUMENT:
        return "GD_ERR_INVALID_ARGUMENT";
    case GD_ERR_OUT_OF_MEMORY:
        return "GD_ERR_OUT_OF_MEMORY";
    case GD_ERR_UNSUPPORTED:
        return "GD_ERR_UNSUPPORTED";
    case GD_ERR_BACKEND:
        return "GD_ERR_BACKEND";
    case GD_ERR_DTYPE:
        return "GD_ERR_DTYPE";
    case GD_ERR_SHAPE:
        return "GD_ERR_SHAPE";
    case GD_ERR_DEVICE:
        return "GD_ERR_DEVICE";
    case GD_ERR_INVALID_STATE:
        return "GD_ERR_INVALID_STATE";
    case GD_ERR_IO:
        return "GD_ERR_IO";
    case GD_ERR_INTERNAL:
        return "GD_ERR_INTERNAL";
    }
    return "GD_ERR_UNKNOWN";
}

const char *gd_status_message(gd_status status)
{
    switch (status) {
    case GD_OK:
        return "ok";
    case GD_ERR_INVALID_ARGUMENT:
        return "invalid argument";
    case GD_ERR_OUT_OF_MEMORY:
        return "out of memory";
    case GD_ERR_UNSUPPORTED:
        return "unsupported";
    case GD_ERR_BACKEND:
        return "backend error";
    case GD_ERR_DTYPE:
        return "dtype error";
    case GD_ERR_SHAPE:
        return "shape error";
    case GD_ERR_DEVICE:
        return "device error";
    case GD_ERR_INVALID_STATE:
        return "invalid state";
    case GD_ERR_IO:
        return "io error";
    case GD_ERR_INTERNAL:
        return "internal error";
    }
    return "unknown error";
}

const char *gd_last_error(void)
{
    return gd_last_error_buf;
}

void _gd_set_last_error(gd_status status, const char *message)
{
    if (status == GD_OK) {
        gd_last_error_buf[0] = '\0';
        return;
    }

    if (message == NULL) {
        message = gd_status_message(status);
    }

    (void)snprintf(gd_last_error_buf,
                   sizeof(gd_last_error_buf),
                   "%s: %s",
                   gd_status_name(status),
                   message);
}

gd_status _gd_error(gd_status status, const char *message)
{
    _gd_set_last_error(status, message);
    return status;
}
