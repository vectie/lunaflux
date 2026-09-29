# Three-kernel source review — Spark, 2026-09-29

Three independent agents investigated QKV ingress, prefill attention and decode
attention. This is a read-only source/SASS follow-up to the
[hardware-counter diagnosis](BENCHMARK_SPARK_COUNTER_DIAGNOSIS_2026-09-29.md),
not a new GPU benchmark or implementation. Relevant LunaFlux source files are
unchanged from measured `53deab59`; unrelated working-tree edits were preserved.
No remote workload or additional GPU allocation was started.

Reference checkouts were clean: vLLM `10f9b5d74fb4110adfc310278e029af4faea0565`,
SGLang `a2e88279c28c16945c7c7eacb27f1e066b670a41`. They corroborate implementation
choices but are not claimed to be the pinned NVIDIA 26.01 images' exact source.
The executed kernel names, launch shapes and counters come from those images.
SGLang's vendored InfLLM/FlashAttention source is explicitly an algorithmic
reference, not proof of the backend selected in its previous E2E run.

## Main findings

| Area | Concrete restriction or overhead | What the baseline does differently |
| --- | --- | --- |
| Full QKV ingress | Projection constrained to 16 tokens × one head per CTA; epilogue geometry constrains GEMM | Large projection tile selected independently of headwise normalization/RoPE |
| Prefill selection | Actual serving CUBIN is synchronous c318; async c322 exists but was not selected | Selected FlashAttention kernel uses async global-to-shared copies |
| Prefill inner loop | KV tile limited to 32/64; extra shared/register movement and scalar softmax instructions | Selected kernel has query64/KV128, blockwise register-oriented processing |
| Decode inner loop | Per-key FP32 recurrence, shuffle dot products and frequent publications | Selected vLLM FlashAttention uses blockwise QK/softmax/PV; not a universal SGLang policy |
| Decode integration | Page-hoist plan not consumed by grouped async renderer; compiler split route capped at batch4 | Alternative address/partition schedules are available in reference implementations |

The common problem is not an absence of functional IR. It is **restricted
physical alternatives, uncalibrated selection, and incomplete realization of
recorded transformations**. Adding another IR wrapper without changing these
properties would preserve the same expensive code.

## 1. Full QKV ingress

### Projection geometry is coupled to the epilogue

The restriction exists at multiple layers:

- [qwen_source.mbt](../kernels/luna_cuda_fused_parallel_aot/qwen_source.mbt),
  lines22–35: rejects schedules other than 16×16×16 microtiles; lines107–124
  map one CTA to 16 token rows and one complete packed head.
- [fold_body.mbt](../kernels/luna_cuda_projection_aot/fold_body.mbt), line110:
  requires input rows16 and one row fragment.
- [column_fold.mbt](../kernels/luna_projection_tile_compiler/column_fold.mbt),
  line124: fused ingress uses the head-component extent as its column domain.
- [ingress_pipeline.mbt](../kernels/luna_projection_tile_compiler/ingress_pipeline.mbt),
  line24: constructs operand storage for that head-local domain.

For 2048 tokens and 32 packed heads of width128, this gives a 128×32 grid:
**4096 CTAs in one kernel launch**, not 4096 separate launches. Each CTA
produces a 16×128 output region. The selected baseline's CUTLASS 256×128×32
tile has 16 times that area and its grid has256 CTAs.

Do not infer matrix-axis orientation from grid dimensions. CUTLASS swizzling
can map a logical16×16 grid to128×2. A transposed problem with128 token rows
and256 channels is consistent with the observed launch, but exact axis
orientation remains an inference. The
[official CUTLASS swizzle implementation](https://raw.githubusercontent.com/NVIDIA/cutlass/v3.9.2/include/cutlass/gemm/threadblock/threadblock_swizzle.h)
explains that distinction.

### Repeated operand requests survive high cache hit rates

[source_ingress_projection.mbt](../kernels/luna_cuda_fused_parallel_aot/source_ingress_projection.mbt),
lines28–39, reloads the token tile for every head CTA and the head's weight
tile for every16-token CTA. With input width1024 and one column window, the
source-implied global-to-shared requested payload for2048 tokens is:

- Input: 4096 ×16 ×1024 ×2 bytes =128 MiB.
- Weights: 4096 ×128 ×1024 ×2 bytes =1024 MiB.
- Total: **1152 MiB requested operand payload**.

This is **not measured DRAM traffic**. Cache hits can serve repeated requests,
but copy instructions, address work, shared-memory traffic and dependencies
remain. An analogous ideal larger-tile calculation is160–192 MiB, depending
on matrix orientation; it is an analytical illustration, not a baseline byte
measurement.

The ingress already has async copies, two operand stages and fragment
lookahead. Its search varies transfer width, stages and live column fragments,
but does not escape16-row/one-head ownership. Reducing live columns can also
create multiple column windows: the complete operand-ring lifetime sits inside
the window loop at `source_ingress_projection.mbt:43`, so lower register usage
can mean repeated staging. More stages or higher occupancy alone are not a fix.

### The epilogue also repeats invariant work

[source.mbt:43](../kernels/luna_cuda_attention_ingress_source/source.mbt)
computes position-specific `sincosf` for every token and Q/K head. For2048
tokens,24 Q/K heads and64 rotary pairs, this is3,145,728 scalar pair evaluations
per layer. Only inverse frequencies are cached. Both reference frameworks
build position-indexed cosine/sine caches: vLLM
`model_executor/layers/rotary_embedding/base.py:94`, SGLang
`srt/layers/rotary_embedding/base.py:171`.

`qwen_source.mbt:146–156` also repeats token-to-request lookup and page metadata
for each head, including Q heads that do not write KV. WMMA accumulators are
stored to `projected[16][128]`, synchronized, and read by the headwise epilogue.
That ownership conversion is legitimate but currently constrains the projection
rather than being independently planned. These are demonstrable repeated
operations; their individual elapsed-time contributions still need ablations.

### What to change in the compiler

Separate the instruction microtile, CTA projection tile and epilogue ownership
region. Represent accumulator redistribution explicitly, allowing multiple row
tiles and multiple complete heads per CTA. Extend the existing physical search,
rather than adding another selector. The whole-chain fusion selector already
exists in [ingress_regions.mbt:103](../kernels/luna_fusion_plan/ingress_regions.mbt)
and [selection.mbt:220](../compiler/fusion_regions/selection.mbt); this benchmark
explicitly forced the full alternative.

The fair test is full fusion versus a larger standalone projection plus the
existing partial epilogue, counting the entire chain. Separately test cached
RoPE values and precomputed token mappings with an explicit numerical contract.
Do not label the 98.11M/13.88M instruction ratio as redundant work: our capture
contains the full epilogue while the baseline counter is projection only.

## 2. Prefill attention

### Correction: the executed route is synchronous

Frozen export metadata maps the unsuffixed
`lunaflux_attention_prefill_tile_compiler_v1` and compilation digest
`0db2adebd505fdd2bf49194aec0442870ffa12151f61a26eb7bb20529bb7b801`
to **candidate318**. Candidate322 is a separately suffixed alternative with a
different compilation digest. The full serving bundle binds the unsuffixed
symbol for modules6/7, with49,168 dynamic shared-memory bytes.

Executed SASS confirms the mapping: the prefill section uses `LDG.E.128` then
`STS.128`, not `LDGSTS`. Thus the previous interpretation that the selected
async prefill failed to hide latency was incorrect. Async code existed but was
not the implementation measured here.

The selection chain explains why:

1. [attention_prepare.mbt:171](../cmd/lunaflux_qwen3_bf16_candidate_export/attention_prepare.mbt)
   starts with empty tuning records; line218 returns the static/resource frontier
   when no tuning file is supplied.
2. [compiler_cost.mbt:98](../kernels/luna_attention_strategy/compiler_cost.mbt)
   gives staged and async matrix copies the same memory multiplier, explicitly
   preserving the synchronous default without measured observations.
3. [compile.mbt:349](../kernels/luna_attention_tile_compiler/compile.mbt)
   breaks equal scores by lower stable ID, selecting318 over322.

This does not prove322 is faster. It proves the serving selection did not use
it and the static estimate cannot distinguish those alternatives here.

### The available physical domain is smaller

[query_owned_frontier.mbt:25](../kernels/luna_attention_strategy/query_owned_frontier.mbt)
only enumerates query and KV dimensions32/64. The selected baseline trait
`<128,64,128,4>` means **head dimension128, query tile64, KV tile128, four warps**.
Its KV128 alternative is not currently expressible by this ownership domain.
The larger KV block amortizes loop, copy/publication and softmax state handling.
It may also increase resource usage, so expanding the domain must retain
resource-constrained selection rather than forcing the largest tile.

The selected LunaFlux metadata path is O(1), not the legacy request-row scan.
The first captured invocation takes the dense-current-token route; executed
counts for its paged fallback are zero. It still repeats position checks in
vector address production. Consequently **page-table traversal cannot explain
the first capture's56% long-scoreboard fraction**. Synchronous operand loads
and dependent position-address production are supported suspects; exact
per-PC stall attribution was not collected for this prefill capture.

### Extra instructions are not primarily extra matrix arithmetic

Existing SASS executed-count analysis shows:

| Executed instruction family, selected first captures | LunaFlux | vLLM |
| --- | ---: | ---: |
| All warp instructions |48,234,368|19,556,736|
| HMMA |4,325,376|4,456,448|
| MUFU.EX2 |1,148,928|1,144,832|
| MOV |4,187,136|380,672|

The first two arithmetic families are close; LunaFlux does not perform2.5×
as much tensor/exponential work. The excess is in staging, redistribution and
scalar arithmetic surrounding softmax. The synchronous path routes global
loads through registers before shared stores, unlike baseline async copies.

The source uses `expf(delta)` with explicit visibility handling and strict
arithmetic choices; the FlashAttention-family source scales into log2 space
and uses exp2 plus different contraction/FTZ policies. SASS contains additional
scalar multiply/add, saturating/range-adjustment and predicate instructions in
LunaFlux. **This is a numerical-policy difference**, not permission to remove
rounding boundaries or substitute fast math invisibly.

The relevant realization points are
[source_query_numeric.mbt:9](../kernels/luna_cuda_attention_tile_source/source_query_numeric.mbt),
[source_online_fold.mbt:50](../kernels/luna_cuda_attention_tile_source/source_online_fold.mbt)
and [source_matrix_fragments.mbt:17](../kernels/luna_cuda_attention_tile_source/source_matrix_fragments.mbt).
The fragment helpers generate packed-operand rearrangements; measured SASS
contains repeated MOV sequences between LDSM and HMMA. The MOV difference is
confirmed, but its division between source representation, register allocation
and compiler settings needs an isolated experiment. The query-owned kernel
already avoids a shared score/probability round trip: removing that nonexistent
round trip is not an appropriate proposed fix.

For this initial dense case, two unrolled position-validation load PCs each
execute135,168 times, and the broadcast instruction executes270,336 times,
despite the paged fallback being unused. The remaining specialization opportunity
is to derive an affine dense interval from step metadata and lower it directly,
not to add more token-path checks. See
[source_read_view.mbt:51](../kernels/luna_cuda_attention_tile_source/source_read_view.mbt).

The frozen exporter still names the RTX5060Ti tuning target. The Spark port
generates sm121 code, but the absence of a measured GB10 tuning table matters:
correct target code generation is not equivalent to calibrated target selection.

The launch grids are not equal useful-work counts: LunaFlux's63×16 grid contains
capacity padding; only512 CTAs execute the main fold. Comparing total CTAs alone
would overstate useful work. Close HMMA/EX2 counts help validate the comparison,
but do not make the two first captures identical tensors or schedules.

### What to test first

Use identical inputs, metadata and numeric policy to compare318 against322;
verify the selected CUBIN, not merely exported alternatives. Then independently
test KV128 ownership, reduced shared/register redistribution, and amortized
position metadata. Finally compare strict exp/arithmetic against an explicit
accuracy-bounded exp2/contraction variant. Record numerical error, instruction
families, load dependencies and full attention-chain time for each change.

## 3. Decode attention

### GQA reuse already exists; the scalar recurrence is expensive

[source_grouped_split.mbt:18](../kernels/luna_cuda_attention_tile_source/source_grouped_split.mbt)
maps one CTA to a sequence/KV head, sharing the tile between two query heads.
Eight warps provide four splits per query head. Adding GQA reuse is therefore
not the missing feature.

For each key, the selected scalar path performs a shuffle-reduced dot product,
maximum update, two exponential scales, scale broadcasts and output-state
rescaling. At history4096 this means4096 online state updates per head,
distributed across four warps. The FlashAttention-family blockwise QK/softmax/PV
schedule updates state approximately32 times per active query row with KV128.

The difference is structural: [compiler_candidate_specs.mbt:91](../kernels/luna_attention_strategy/compiler_candidate_specs.mbt)
only adds matrix candidates for Prefill. Decode cannot select that family.
Moreover [KeyFold](../compiler/attention_physical_ir/key_fold.mbt) deliberately
preserves FP32 per-key probabilities, whereas the matrix `OnlineFold` has a
distinct BF16 probability law. A blockwise/matrix rewrite needs a named numeric
variant and oracle tests, not a claim of bitwise-equivalent reassociation.

### Small tiles repeat effects and address production

The selected async32 path has128 iterations for4096 keys.
[grouped_decode_effects.mbt:62](../kernels/luna_attention_tile_schedule/grouped_decode_effects.mbt)
requires K wait/publication, V wait/publication, and reader-release publication:
**384 per-iteration workgroup publications**, plus setup/final merge, and256
K/V commit groups. The cited baseline-family loop amortizes its effects over
128-key blocks. Current barriers protect real dependencies; removing them
requires changing and proving slot ownership/lifetimes.

The generic optimizer records page lookup invariance in
[optimize.mbt:140](../kernels/luna_attention_tile_optimizer/optimize.mbt), but
[source_decode_pipeline.mbt:19](../kernels/luna_cuda_attention_tile_source/source_decode_pipeline.mbt)
does not consume the hoist flag. It repeats page lookup/check/index formation
for each8-byte vector, then reuses that address between K and V. A recorded
optimization is not yet an emitted optimization on this route.

The compiler-readonly split path is also capped at batch4 in
[paged_decode_split_prepare.mbt:118](../engine/device_step/paged_decode_split_prepare.mbt),
enforced by [graph_bucket.mbt:113](../engine/device_step/graph_bucket.mbt).
C16 cannot select this existing cross-CTA alternative. More splitting is not
automatically faster: it adds workspace and merge costs, which must be measured.

### Why fewer instructions do not imply proportional speedup

The selected decode captures have121.35M versus17.79M instructions, but replay
times1305.86 versus1207.97µs—only about8% apart. Baseline long-scoreboard share
is approximately67.7%, versus32.9% for LunaFlux. The baseline itself remains
load-dependency limited. Inputs/page maps also differ: our capture is the
CPU-oracle-checked exact-CUBIN probe, baseline is serving data.

Do not conclude a6.8× potential speedup, or equate sampled BAR/ISETP/BSSY PCs
with the instruction causing the underlying memory latency. Zero excess shared
wavefronts in this probe likewise do not imply no other synchronization cost.

SGLang nuance matters: its local FlashInfer policy in
`srt/layers/attention/flashinfer_backend.py:2419` defaults BF16 tensor-core decode
at GQA≥4; this model has GQA2. It is false to claim every baseline always uses
matrix decode here. Its actual selected backend requires a capture.

## Implementation priorities and acceptance tests

1. **Fix selection/realization before inventing another abstraction:** compare
   available prefill async against sync, and connect grouped decode page-hoist
   planning to emitted address ownership. Tests must inspect the selected
   artifact and emitted behavior, not just a plan flag.
2. **Expand representable schedules:** projection CTA rows/head groups,
   independently owned epilogue, prefill KV128, and bounded decode partitioning.
3. **Separate numerical variants:** ordered FP32 key fold, blockwise FP32 fold,
   BF16 matrix PV, strict/accuracy-bounded exponential arithmetic. Never hide
   rounding/association changes inside an ostensibly identical lowering.
4. **Reuse invariant work:** immutable RoPE values and bounded per-step token/page
   descriptors. Keep these away from per-head/per-vector rediscovery.
5. **Select complete-chain winners:** existing fusion and resource selectors
   should consume comparable measured results. A lower instruction count or
   bank-conflict count is not the optimization objective.

Future tests should keep one engine on the GPU at a time, cap allocations and
workspace, retain the32 GiB host-availability cutoff, and use equal tensors/page
maps for microbenchmarks. This investigation itself allocated no GPU memory.
