#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

gd_status _gd_cpu_k_step_inc(float *step)
{
    step[0] += 1.0F;
    return GD_OK;
}
