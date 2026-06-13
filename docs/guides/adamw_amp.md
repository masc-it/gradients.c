# AdamW and AMP

`gradients.c` provides a Metal-backed AdamW optimizer with AMP support.

## Components

- `gd_adamw_create` creates optimizer state.
  - FP32 `m`/`v` moments live in the state arena.
  - F16 trainable params get FP32 master weights in the state arena.
- `gd_amp_scaler` owns device-side dynamic loss-scale state.
- `gd_backward_amp` seeds backward from the scaler's device scale, avoiding a CPU scale read per step.
- `gd_optimizer_step_amp` unscales accumulated grads, checks finite values, conditionally updates AdamW state/params, advances optimizer step state, and updates the scaler in one device-side transaction.

## Typical flow

```c
gd_amp_scaler *scaler;
gd_amp_scaler_create(ctx, NULL, &scaler);

/* inside GD_SCOPE_TRAIN */
gd_backward_amp(ctx, &loss, NULL, scaler);
gd_optimizer_step_amp(ctx, optimizer, scaler);
gd_optimizer_zero_grad(ctx, optimizer);
```

If overflow is detected, `gd_optimizer_step_amp` leaves params/moments/master weights and optimizer step counters unchanged, then backs off the scale on device. Host-visible scaler and optimizer counters are lazy after AMP steps; sync only when needed for logging/checkpointing:

```c
gd_amp_scaler_sync(ctx, scaler);
gd_optimizer_sync_state(ctx, optimizer);
gd_amp_scaler_last_found_inf(scaler);
gd_amp_scaler_scale(scaler);
gd_optimizer_step_count(optimizer);
```
