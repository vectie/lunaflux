# Fused parallel CUDA AOT candidates

This package emits two offline, deterministic, candidate-only CUDA families:

- QKV projection fused with per-head Q/K RMSNorm, positioned RoPE, and paged
  split-K/V writes;
- residual addition fused with the immediately following RMSNorm.

The Qwen full-ingress CUDA lowering consumes a backend-neutral functional
projection schedule. When that schedule hoists the rotary inverse-frequency
basis and fuses paired rotary evaluation, the lowerer materializes the basis as
an AOT CUDA constant table and emits one paired `sincosf` evaluation per rotary
pair. No runtime `powf` remains in this path, and model or compiler-middle-end
code does not name CUDA storage classes or intrinsics.

Residual/RMSNorm has two deliberately distinct artifacts. The qualification
artifact retains the dispatch-canary pointer and atomic publication used by
offline correctness campaigns. Its production companion is derived from that
exact candidate, binds the qualification candidate digest, and has a six-
pointer ABI with no canary allocation, atomic write, or observation contract.
Both remain inert until their own content-addressed CUBIN and startup authority
are admitted; the current approval record covers only qualification and cannot
be projected into production runtime authority.

Both candidates require block size 128, bounded paged profiles, canonical BF16
layout, an explicitly supported CUDA target, strict non-reassociating compiler
policy, and exact alignment. Their recipes bind the model and operation chain,
layout, source, compiler/toolchain, numerical policy, diagnostic policy, and every
standalone correctness-kernel source and recipe digest. The fallback kernels
remain distinct and available; these candidates do not replace their catalog
entries.

Offline CUBIN output can now be joined to each candidate through
`FusedParallelCompiledArtifactBinding`. The binder requires two byte-identical
bounded CUBIN snapshots, the builder's canonical compile receipt, and a
separately pinned digest of that receipt. It rehashes the candidate source,
recipe, receipt, and both CUBINs; matches the exact toolchain, driver, target,
symbol, launch geometry, and family-specific raw-pointer ABI; and emits one
canonical binding record. Production bindings additionally retain the exact
qualification-candidate digest and require `dispatch_canary_per_token=0`. The
result remains candidate-only and has no
`KernelModuleInput`, manifest admission, deployment approval, compiler,
promotion, device, or runtime projection.

The two numerical names are deliberately separate from the reference kernels
and from one another. The QKV/RoPE path admits its ordered-F32 transcendental
tolerance, while residual/RMSNorm admits a block-128 F32 tree-reduction
tolerance. Neither name implies a performance result.

`FusedParallelPromotionBinding` separately joins the exact candidate to the
existing Phase 5 `LunaSpecializationEvidence` subjects. It remains
`manifest_bindable=false`, records `performance_claim=none`, and provides no
compiler, device, runtime, or deployment authority. Physical differential,
sanitizer, race, microbenchmark-win, mixed-workload, compilation, and reviewer
evidence are still required before any release integration.
