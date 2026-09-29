# Elementwise physical programs

Pure immutable row-domain plans complement matrix/attention IR without forcing
normalization, rotary transforms, KV writes or selection into GEMM.

- `NormalizationProgram` preserves residual BF16 rounding and distinguishes
  owner-tree from subgroup-tree F32 reduction order. `ReduceOwnerTree` is a
  descending-stride fold with synchronization after every stride.
- `RotaryProgram` partitions split-half Q/K rotation from disjoint value-copy
  work. `KvWriteProgram` retains split cache geometry and ordered write effects.
- `SelectionProgram` reduces greatest finite value / lowest tied token / lowest
  invalid token summaries. `StochasticProgram` retains ordered F64 softmax,
  minimal top-p prefix, retained-mass sum and counter-based inverse-CDF order.
- `IngressProgram` explicitly orders scratch load, publication, Q/K-only
  normalization/rotation, activation/cache stores and scratch retirement.

CUDA lowering supplies owner/subgroup widths and renders device instructions.
These plans neither allocate device storage nor introduce token-path work.
Model identities, launch artifact authority and runtime page ownership stay in
their existing owners. This module covers the legacy dense decoder generators;
it does not claim migration of unrelated advanced-model source candidates.
