#include "../_shared/binary/cuda_binary_common.cuh"

extern "C" gd_status gd_backend_sub(gd_backend *backend,
                                      const gd_backend_tensor_view *x,
                                      const gd_backend_tensor_view *y,
                                      const gd_backend_tensor_view *out)
{
    return gd_cuda_binary_dispatch(backend, x, y, out, GD_CUDA_BINARY_OP_SUB);
}
