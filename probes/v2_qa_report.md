# v2 QA stress report

Date: 2026-06-03
Scope: `docs/design_spec.md` foundation/runtime contract in current checkout.

## New probe files

- `probes/v2_api_contract_stress_probe.c` — links against `libgradients.a`; stresses public memory/tensor/transfer API contracts and aggregates failures.
- `probes/v2_semantics_oracle_probe.c` — standalone oracle for VLM suffix layout, suffix-only LMCE, duplicate embedding scatter-add, KV cache by-value `start_pos`, and one-backward multi-loss flow.
- `probes/v2_design_spec_audit.py` — static coverage audit against spec milestones.
- `probes/run_v2_qa_stress.sh` — builds/runs probes without Makefile changes.

## Commands run

```sh
make BUILD_DIR=build-qa-probes check
sh probes/run_v2_qa_stress.sh
```

## Results

- Existing `make check`: **PASS**.
- `v2_semantics_oracle_probe`: **PASS**.
- `v2_design_spec_audit.py`: **PASS=12 MISS=25** coverage rows.
- `v2_api_contract_stress_probe`: **FAIL** — `pass=74 fail=5 skip=0`.

## Findings

1. **Active-scope blocking read stale.** `gd_tensor_ones(... GD_ARENA_SCRATCH ...)` then `gd_tensor_read` before `gd_end` returns `[0,0,0,0]`; after `gd_end` returns `[1,1,1,1]`. Violates blocking-read semantics inside scoped eager execution.
2. **Active-scope CPU write ordering broken.** `gd_tensor_ones`, then `gd_tensor_write([2,3,4,5])`, then `gd_end` leaves tensor as `[1,1,1,1]`. Queued fill overwrites later blocking write.
3. **Failed rand init burns arena bytes.** `gd_tensor_rand_uniform` with unsupported dtype (`I32`) or invalid range does not publish descriptor, but advances `params` offset. Validate dtype/range before allocation.
4. **Early scratch alloc error leaves stale output span.** `gd_alloc_scratch` outside scope returns `GD_ERR_BAD_STATE` but does not clear caller-provided `gd_span`.
5. **Spec coverage gaps expected but visible.** Static audit shows foundation memory/tensor/transfer present, but stream/kernel/matmul abstractions, op capsules, autograd, optimizer/AMP, checkpoint, modules, and VLM/GPT model markers are not in `include/` or `src/` yet. README still describes v1 CPU/graph behavior.

## Passed stress areas

- OOM does not advance `params` offset or publish span.
- Sealed `params` rejects late allocation.
- Scratch ring cycles slots and bumps generations before stale tensor validation.
- Middle-dim suffix slice is non-contiguous; explicit `contiguous` allocates distinct packed output.
- Core debug heap guard sees no tracked heap allocation in tensor hot path.
- F16/BF16 one bit patterns correct.
- F32 `rand_uniform` deterministic for same seed and bounded.
- Oracle semantics: suffix LMCE computes `204` logits vs `374` full logits, prefix grad remains zero, duplicate embedding grads scatter-add, KV path uses no cache-position tensor, multi-head loss uses one backward seed.

## Recommended fixes

1. Transfer helpers must handle active command buffers: before blocking read/write, commit/wait queued work touching target span or split/flush current backend scope and resume ordered execution.
2. `gd_tensor_rand_uniform` should validate dtype/range before `gd_tensor_empty` allocation.
3. Public alloc functions should zero output structs before every early return path.
4. Keep static audit non-strict while implementing spec stepwise; switch selected rows to strict as milestones land.
