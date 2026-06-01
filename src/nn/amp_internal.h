#ifndef GRADIENTS_AMP_INTERNAL_H
#define GRADIENTS_AMP_INTERNAL_H

#include "gradients/optim.h"

gd_status _gd_amp_scaler_validate(gd_amp_scaler *scaler);
gd_tensor *_gd_amp_scaler_scale_tensor(gd_amp_scaler *scaler);
gd_tensor *_gd_amp_scaler_found_inf_tensor(gd_amp_scaler *scaler);

#endif /* GRADIENTS_AMP_INTERNAL_H */
