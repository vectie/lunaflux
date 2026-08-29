# Luna CUDA finite-E4M3 projection AOT

This package emits deterministic, root-free CUDA source and recipe candidates
for catalog-v4 finite-E4M3 projections. Version 1 remains byte-stable: it
supports QKV, output, and language-model-head projections with one final
4-byte `Workspace` operand and fails closed for GatedMlp.

The additive v2 lowerer authenticates the model plan's exact numeric execution
policy and complete canonical paged-v4 raw operand order. Simple QKV, output,
and language-model-head operations own one final 4-byte `Workspace`; compound
GatedMlp owns one final 8-byte `Workspace`. Each weight is immediately followed
by its scalar F32 `WeightScaleInput`. The one raw workspace pointer addresses
cell zero for the external-input scale and, for GatedMlp, cell one for the
post-SiLU gate/up-product scale. No out-of-band pointer is emitted.

The gated reference CUDA performs the external reduction and finite-E4M3
round/reconstruction, strict F32 gate/up dots, target/compiler-bound `expf`
SiLU product, and a deterministic token-row then intermediate-column reduction
for the second scale. Version 2 uses one serial index-zero reference thread and
two passes: every live output and intermediate is first checked for finiteness,
then output bytes are written, and only then are finite-positive scale cells
published. Both workspace cells start as canonical quiet-NaN bits `0x7fc00000`.
The executor must reject either cell unless it is finite and strictly positive.
This avoids invalid grid-wide synchronization. The deliberately slow reference
execution is an honesty/correctness path, not a performance claim.

Both versions require CUDA 12 `cuda_fp8.h`, use the native
`__nv_fp8_e4m3` conversion boundary with SATFINITE/RNE semantics, canonicalize
negative zero to positive zero, and are closed to `sm_89` and `sm_90`.

Compilation remains offline. The companion `luna_kernel_bundle` binder accepts
only byte-identical dual CUBIN outputs covered by a deterministic compiler
receipt and an already-admitted catalog-v4 artifact module/symbol mapping. No
compiler process, filesystem root, runtime JIT, module loader, CUDA context,
executor route, readiness claim, or physical correctness evidence is owned
here.
