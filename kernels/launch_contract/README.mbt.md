# AOT launch contracts

`kernels/launch_contract` admits inert, bounded launch metadata for every exact
full-prefill profile and every AOT-backed semantic model operation. It keeps
the three relevant identities separate:

- a catalog binding chooses an AOT kernel family inside a content-addressed
  module;
- a profile contract chooses a stable entry point in that family;
- artifact admission later maps the stable entry point to a bounded CUDA
  function symbol and owns the module bytes.

An admitted contract fixes exact batch, sequence, and token-row counts, launch
geometry, ordered semantic operand roles, byte counts, alignments, and the
catalog workspace contract. It has no scalar, arbitrary payload, path, compiler,
cache, or JIT channel. A max-row contract is never substituted for a smaller
profile.

This package validates semantic ABI completeness against `ModelPlan` and
`ResolvedKernelCatalog`. It intentionally does not import engine memory plans.
Consequently its byte counts and alignments are manifest claims, not proof of
physical allocation safety. The device executor must validate every admitted
operand against the exact resolved weight, external-input, activation, and
workspace region before constructing reusable device arguments. Execution is
forbidden until that second validation succeeds.

## Paged-KV semantic version 3

`admit_paged_kv` is a separate catalog-v3 admission path for only positioned
rotary and paged causal-attention operations. Its serialized operand
order is exact: operation runtime inputs in `ModelPlan` order, semantic value
inputs, activation outputs, optional per-layer Key and Value component bases,
then an optional catalog workspace.
Alternate orders are invalid even when they contain the same roles.

The v3 runtime metadata ABI is deliberately concrete: `StepCounts` is exactly
five little-endian Int32 values, and positions, packed row offsets, sequence
lengths, packed page offsets, and physical page indices use bounded Int32
arrays. This statement applies only to semantic version 3. Persistent KV uses
two exact per-layer component-base operands, Key then Value. The catalog binds
the canonical `KvCacheLayout` version and tokens per page, while physical page
count only sizes each admitted component span and never specializes indexing.

Paged admission verifies model, catalog, and physical-layout identities and
geometry, but returns inert metadata only. It does not load a CUDA module,
construct device arguments, launch a kernel, or claim executor support.
`admit_paged_kv` deliberately omits embedding, projections, normalization,
residual, MLP, and language-model-head launches; supplying any such contract
fails closed. `admit_paged_graph` instead requires one exact AOT contract for
every operation and marks the result `FullGraph`. The explicit scope prevents
partial rotary/attention evidence from being mistaken for a complete graph.
