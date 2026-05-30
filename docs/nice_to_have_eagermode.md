# Nice-to-have: Eager execution façade

Status: proposal (not scheduled)

## Context / motivation

External feedback flagged the biggest UX wart: every compute op requires an
active graph, and the only escape (`gd_graph_run_immediate`) is a callback
wrapper that builds/runs/tears down a temporary graph per call. Consequences:

- High friction to "just add two tensors" or explore interactively.
- Interactive debugging is hard: inspecting an intermediate means inserting a
  debug node and recompiling, instead of calling an op and printing.
- Immediate helpers don't compose (chain op → inspect → op).
- This contradicts design pillar 10 ("graph debugging is first-class") and §20
  ("graph-only execution must still be easy to debug") — there is no interactive
  path today.

## Diagnosis (important)

The problem is **not** the graph-only IR core. "Eager" and "graph" are not
opposites: PyTorch eager also builds a graph (the autograd tape) per op. Eager =
*implicit graph + immediate execution + materialized result*.

So the gap is a missing **eager façade over the existing IR**, not a wrong core.
We deliberately rejected a *second* eager kernel path (two semantics, two impls,
parity tax — the cost PyTorch pays for eager + `torch.compile`). A fusion-first
GPU training library should keep one engine.

Crucially, this is **not baked into the architecture**. `GD_ERR_INVALID_STATE`
is a single branch in `require_active_graph`. After the P0–P3 backend refactor,
adding eager is a façade, not a rewrite.

## Goal

Make this work, no explicit graph lifecycle, output immediately readable:

```c
gd_context_set_eager(ctx, true);
gd_tensor *c = NULL;
gd_add(ctx, a, b, &c);              /* records + runs one node */
gd_debug_print_tensor(ctx, c, 8);  /* inspect right away */
gd_tensor *d = NULL;
gd_relu(ctx, c, &d);               /* composes naturally */
```

Same IR, same kernels, same backend `execute` path. **One engine.** Explicit
graphs remain the performance/fusion path (training step, prefill/decode).

## Design sketch

### Mode

- `gd_status gd_context_set_eager(gd_context *ctx, bool enabled);`
- `bool gd_context_eager(const gd_context *ctx);`
- Default: off for now (opt-in); could default on for dev builds later.

### Op behavior when eager and no explicit graph is active

Each op, instead of returning `GD_ERR_INVALID_STATE`:

1. Lazily create/reuse a context-owned **implicit graph** in `building` state.
2. Append the op node as usual (`_gd_graph_emit*`).
3. Finalize + compile + execute through the **same backend vtable** (P3), or run
   just the new node via `execute_until(node_id)`.
4. Materialize the output into a standalone tensor
   (`_gd_tensor_materialize_from_graph`), detaching it from the implicit graph.
5. Reset the implicit graph for the next op (inputs already materialized).

Net: every eager output is a materialized, immediately-readable tensor; inputs to
the next op are imported as leaves. No explicit lifecycle, no callback.

### Autograd in eager mode

- Two sane options; pick one explicitly:
  - **v1 simple:** eager is inference/no-grad by default; `gd_backward` requires
    an explicit graph (the training path). Eager is for exploration/debug.
  - **later:** maintain a persistent implicit tape across eager ops so
    `gd_backward` works eagerly (PyTorch-style). More state to manage; defer.
- Recommend v1 simple first; it removes the UX cliff without tape bookkeeping.

### Reuse, not duplication

- No new kernels, no second executor. Eager = implicit-graph + per-op
  compile/execute over the existing backend.
- A small per-node executable cache (keyed by op + input descs) makes repeated
  eager ops cheap; build cost is acceptable since eager targets dev/debug, not
  training throughput.

## What it fixes

- "Just add two tensors" works in 2 lines.
- Interactive inspection: call op → `gd_debug_print_tensor` → call op.
- Examples/tests shed lifecycle boilerplate for the exploratory paths.
- Resolves the pillar-10 / §20 contradiction with a real interactive path.

## What it explicitly does NOT change

- Graph-only **core** stays. Explicit graphs remain the fused, allocation-free
  performance path; training step graphs are still the recommended pattern.
- No second kernel/execution path. One engine, one set of semantics.
- Static-shape policy unchanged (separate concern; bucketing/symbolic later).

## Costs / risks

- Per-op compile/dispatch overhead in eager mode (mitigated by cache; acceptable
  for dev/debug use).
- Two *entry styles* (eager façade vs explicit graph) — but **one** underlying
  engine, so no semantic drift if eager strictly lowers to the same IR + backend.
- Implicit-graph lifetime/ownership: outputs must be materialized+detached before
  the implicit graph resets, or they dangle. Reuse existing
  `_gd_graph_materialize_live_virtuals` discipline.

## Rough scope

- `gd_context_set_eager` / `gd_context_eager` + an implicit graph handle on the
  context.
- `require_active_graph` (in `src/ops/op_schema.c`) gains an eager branch that
  drives implicit build→compile→execute→materialize.
- Tests: eager add/mul/matmul numeric == explicit-graph results; eager
  inspect-between-ops; eager + (v1) no-grad guard.
- Estimate: a few hundred lines, no backend changes, reuses P3 dispatch.

## Recommendation

Worth doing, but **after** the Metal prerequisites (P4–P10) since those are
strict blockers for the next milestone and the eager façade is additive. Slot it
as its own phase once the backend seam is fully closed. It is the highest-value
*ergonomics* improvement and the cleanest way to honor the debugging pillar.
