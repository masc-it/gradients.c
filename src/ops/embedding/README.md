# embedding

Materialized token embedding lookup.

```c
gd_embedding(ctx, table, ids, &out);
gd_embedding_backward(ctx, table, ids, grad_out, &grad_table);
```

Contract:

- `table`: contiguous `f16`/`f32` tensor shaped `[vocab, dim]`.
- `ids`: contiguous `i32` tensor with rank `>= 1`.
- `out`: contiguous tensor shaped `ids.shape + [dim]`, with `table` dtype.
- Backward returns a dense contiguous gradient table with `table` shape/dtype.

Metal implementation:

- Forward uses dtype-specialized kernels and a 16-byte vectorized row-copy path when
  the embedding row width is 16-byte aligned.
- Backward zeroes a dense table, then atomically scatters `grad_out` rows by id.
- `f32` tables accumulate directly into the output gradient table.
- `f16` tables accumulate into an internal `f32 [vocab, dim]` scratch table and cast
  once to `f16`, avoiding unsupported half atomics while preserving correct duplicate-id
  accumulation before final storage quantization.

Invalid ids are guarded on GPU: forward writes NaNs for invalid rows and backward
ignores invalid ids instead of reading/writing out of bounds.
