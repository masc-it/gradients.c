# AdamW and AMP

`gradients.c` provides a Metal-backed AdamW optimizer with AMP support.

## Components

- `gd_adamw_create` creates optimizer state.
  - FP32 `m`/`v` moments live in the state arena.
  - F16 trainable params get FP32 master weights in the state arena.
- `gd_amp_scaler` tracks dynamic loss scale.
- `gd_backward_scaled` seeds backward with `grad_output * scale`.
- `gd_optimizer_step_amp` unscales accumulated grads, checks finite values, skips on overflow, updates AdamW state/params on finite steps, then updates the scaler.

## Typical flow

```c
gd_amp_scaler *scaler;
gd_amp_scaler_create(NULL, &scaler);

/* inside GD_SCOPE_TRAIN */
gd_backward_scaled(ctx, &loss, NULL, gd_amp_scaler_scale(scaler));
gd_optimizer_step_amp(ctx, optimizer, scaler);
gd_optimizer_zero_grad(ctx, optimizer);
```

If overflow is detected, `gd_optimizer_step_amp` leaves params/moments/master weights unchanged and backs off the scale. Query with:

```c
gd_amp_scaler_last_found_inf(scaler);
gd_amp_scaler_scale(scaler);
```
