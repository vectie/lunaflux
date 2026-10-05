# Fused parallel CUDA AOT candidates

This package emits two offline, deterministic, candidate-only CUDA families:

- QKV projection fused with per-head Q/K RMSNorm, positioned RoPE, and paged
  split-K/V writes;
- residual addition fused with the immediately following RMSNorm.

The Qwen full-ingress CUDA lowering consumes a backend-neutral functional
projection schedule. When that schedule hoists the rotary inverse-frequency
basis and fuses paired rotary evaluation, the lowerer materializes the basis as
an AOT read-only CUDA table and emits one paired `sincosf` evaluation per rotary
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

The legacy families require block size 128 and bounded paged profiles (up to
8,192 query tokens and 16,384 aggregate page-table entries), canonical BF16
layout, an explicitly supported CUDA target, strict non-reassociating compiler
policy, and exact alignment. Their recipes bind the model and operation chain,
layout, source, compiler/toolchain, numerical policy, diagnostic policy, and every
standalone correctness-kernel source and recipe digest. The fallback kernels
remain distinct and available; these candidates do not replace their catalog
entries.

The query bound is an offline compilation envelope, not the default scheduler
chunk or a performance decision. Concrete operand extents, physical programs,
launches and source guards are regenerated from the requested profile. Known
strict toolchain identities are CUDA 13.0.88 and 13.1.115; recipes retain the
actual version and digest. Other versions remain unsupported. Larger envelopes
still require their own physical correctness, sanitizer and whole-serving gates;
acceptance by the source exporter does not promote an 8192-token serving route.

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

Ingress qualification and production rendering share one retained physical
program: selected column window, operand ownership/layout, matrix fragment
lifetime, scalar fold and numerical epilogue. The source and recipe consume
that same program; neither independently reconstructs its column-fold plan.
An incompatible refinement raises the existing typed compiler-policy error.
Digest regressions track deterministic complete ingress sources for three
head dimensions and three token bounds, including production forms.

## Packed complete-head producer and row-scoped rotary reuse

`Policy.head_tiles` is an offline, backend-neutral ownership choice. The pure
`AttentionIngressPacking` plan maps producer columns to Q/K/V operand rows,
gives masked tails no store authority, and retains the explicit
prepare-row / consume-complete-heads / retire-row rotary lifetime. CUDA lowers
that plan to one input-tile traversal shared by the retained local head domain
and one lane-owned rotary preparation reused across its Q/K heads. Each head
still uses the original BF16 rounding points and ordered normalization tree;
the scalar single-token law is unchanged.

The shared numerical lowerer now consumes register-pair retention from the
pure physical plan for aligned split-half heads. It avoids the intermediate
normalized scratch writes/reads and two helper-local warp publications while
retaining the exact reduction tree and normalization/rotation BF16 rounds.
The cache-bound helper is a distinct lowering which consumes GPU-prepared F32
sine/cosine pairs; its extra operand and preparation launch must be explicitly
bound by the production candidate and executor, never inferred from an old ABI.

The AOT frontier includes one, two and four heads per CTA, live accumulator
windows of one, two, four and eight, and the existing row/transfer/stage choices.
Packed producer and epilogue storage participate in the same resource bound;
for example head-128/h4 is rejected by the current 48 KiB shared limit while
head-128/h2 is available. More heads without enough live columns cause multiple
producer windows, so packing alone is not a claim of fewer input loads. Defaults
remain h1 until an exact target/workload measurement selects another candidate.
Recipes export `heads_per_cta`; grid Y is the ceiling of packed heads divided by
that field, not a guessed multiplicity from launch geometry.

`ingress_packing_export_wbtest.mbt` exports head-64/h1,h2,h4 with Q6/K3 and
head-128/h1,h2 with Q1/K1. `benchmarks/gpu_pipeline/ingress_packing_probe.cu`
checks distinct segment/head/component weights, row tails, segment-crossing
groups, a masked final head, reversed page tables, exact KV/output agreement,
and untouched cache locations. Hardware execution/sanitizer and timing remain
separate required checks; these compiler tests do not claim physical speedup.
