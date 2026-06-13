#include <gradients/status.h>

const char *gd_status_string(gd_status status)
{
    switch (status) {
    case GD_OK: return "ok";
    case GD_ERR_INVALID_ARGUMENT: return "invalid argument";
    case GD_ERR_OUT_OF_MEMORY: return "out of memory";
    case GD_ERR_FROZEN: return "frozen";
    case GD_ERR_BUSY: return "busy";
    case GD_ERR_BAD_STATE: return "bad state";
    case GD_ERR_UNSUPPORTED: return "unsupported";
    case GD_ERR_INTERNAL: return "internal error";
    case GD_ERR_NOT_IMPLEMENTED: return "not implemented";
    case GD_ERR_IO: return "I/O error";
    default: return "unknown error";
    }
}
