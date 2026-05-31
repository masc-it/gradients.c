# Nice-to-have: Dynamic / symbolic shapes

Status: proposal (not scheduled)

## Context / motivation

External feedback flagged static compiled shapes (design spec §23.5) as the
runner-up UX cost: each distinct shape needs its own compiled graph, so an
autoregressive model needs a separate compiled graph for prefill (`[B,T]`) and
decode (`[B,1]`).

## Diagnosis (calibrated)

- **"Juggle two graphs" is mild.** You compile prefill once and decode once and
  dispatch host-side per phase — two handles, not per-token juggling. Splitting
  prefill vs decode is *normal*; production inference stacks often want separate
  prefill/decode kernels anyway (different compute profiles, different fusions).
- **The real cost is combinatorial recompiles**, not "two graphs": arbitrary
  prefill lengths and dynamic batch sizes → a distinct compiled graph per shape →
  compile storms. Decode at `[B,1]` is fixed and fine.
- **This is cleanly additive to fix**, more so than most v1 limitations: the IR
  already carries shape descriptors. Making dims symbolic and specializing at
  compile/run is a well-trodden compiler feature (XLA dynamic, TVM Relax, ONNX
  dynamic axes). Not an architectural reversal.

## Why static-first was the right v1 choice (keep it)

Static shapes are partly *enabling*, not just limiting:

- They make "allocation-free after compile" and the naive per-value allocator
  trivial. Dynamic shapes force a shape-polymorphic memory planner — exactly the
  planner work we deferred.
- They keep kernel/executable cache keys simple.

So static shapes should stay for v1. This document is about the interim pattern
and the eventual feature, not a call to change the core now.

## Interim pattern (no compiler work): bucketing + caching

Available today with the current static-shape engine:

- **Shape bucketing + padding.** Pad variable dims (e.g. seqlen) up to a small
  set of buckets (powers of two), mask the padded region. Turns "one graph per
  length" into a handful of compiled graphs.
- **Max-shape compile + masking** for prefill: compile at the max bucket, mask.
- **Executable cache keyed by shape.** Repeated shapes reuse a compiled graph
  instead of recompiling. (Pairs naturally with the per-node executable cache
  proposed for eager mode.)

Document these as the recommended pattern for autoregressive / variable-shape
models in v1.

## Eventual feature: symbolic shapes (post-Metal, with the planner)

- Allow selected dims to be **symbolic** in `gd_tensor_desc` (e.g. a dim bound to
  a named size variable resolved at run).
- Compile once into a **shape-polymorphic executable**; bind concrete sizes at
  run via a small shape-environment.
- Requires:
  - shape inference passes that propagate symbolic dims,
  - a memory planner that sizes/reuses buffers from the bound shapes at run
    (depends on the layout/reuse planner we deferred),
  - backend kernels that accept runtime dims (most do; some fused GPU kernels may
    specialize per shape-class and fall back to bucketing).
- Natural home: alongside the post-Metal compiler roadmap (layout selection +
  buffer reuse + fusion), since dynamic memory planning is the hard dependency.

## Recommendation

- Keep static shapes in v1; they simplify the allocator we have not built.
- Ship **bucketing + executable cache** as the interim ergonomic answer.
- Put **symbolic shapes** on the post-Metal compiler roadmap, gated behind the
  memory-reuse planner.
- Priority: below the eager façade (daily friction, every op) and below the Metal
  prerequisites (next-milestone blockers). This bites only variable-shape
  workloads and has cheap interim mitigations.
