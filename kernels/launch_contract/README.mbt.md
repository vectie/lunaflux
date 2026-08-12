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
