#ifndef GRADIENTS_AMP_INTERNAL_H
#define GRADIENTS_AMP_INTERNAL_H

#include "gradients/optim.h"

gd_status _gd_amp_scaler_validate(gd_amp_scaler *scaler);
gd_tensor *_gd_amp_scaler_scale_tensor(gd_amp_scaler *scaler);
gd_tensor *_gd_amp_scaler_found_inf_tensor(gd_amp_scaler *scaler);
gd_status _gd_amp_clip_grad_norm(gd_context *ctx,
                                 gd_tensor **params,
                                 int n_params,
                                 gd_tensor *scale,
                                 gd_tensor *found_inf,
                                 float max_norm,
                                 gd_tensor **norm_out);

#endif /* GRADIENTS_AMP_INTERNAL_H */
