# DeepSeek matched original layer-stream replay — 2026-10-11

## Matched output-A accumulation isolation — 2026-10-12

After attention/inverse become exact, output-A has 324 differing BF16 values.
The new intervention uses that **identical original inverse input**, unchanged
weights and the complete 144-row original output-A. Exported production code
reproduces the retained native output bitwise; separate BF16 dequantization
also reproduces it. This isolates dot summation order, not input or scale drift.

| Output-A reduction | Different BF16 / 1,179,648 | Relative L2 |
| --- | ---: | ---: |
| Production ordered F32 | 324 | 0.002497892% |
| 8 striped partials + fixed tree | 169 | 0.001352904% |
| 16 striped partials + fixed tree | 137 | 0.001543245% |
| 32 contiguous partials + fixed tree | 195 | 0.002182036% |
| F64 accumulator, final F32/BF16 rounding | 121 | 0.001466773% |
| **32 striped partials + fixed tree** | **0** | **0%, bitwise** |
| Original-library one-row grouped GEMM control | **0** | **0%, bitwise** |

Thus higher accumulator precision alone does not reproduce the original's
rounding; partition ownership and its join order matter. Root
`/tmp/lunaflux-deepseek-output-arithmetic-20261012-v1` retains all controls.
`v2` exports the actual cooperative CUDA lowering: **all 1,179,648 outputs
match bitwise**, both all-row and incremental one-row execution. Default CUDA
source remains byte-identical. All arrays are finite.

The existing immutable `DotReductionLaw` gains `StripedTreeFloat32(partitions)`;
there is no second numerical-policy type. Grouped output owns that law, the
decoder propagates it through window/compressed paths, and CUDA alone owns
subgroup/shuffle geometry. Scalar renderers explicitly reject this unsupported
law instead of substituting another sum. The offline `prepare-striped` route
selects tiled attention, complex rotary and this output reduction at both export
and run. Ordinary serving retains its original ordered policy.

The full-144-row actual lowering passes memcheck with zero errors/leaked bytes.
Racecheck is killed at its 2-GiB then 8-GiB **instrumentation** cap, with zero
swap; those failures are preserved. Fresh `v3` bounded **one-row/all eight groups
and 1,024 channels per group**, bulk and incremental executions complete
memcheck/leak, racecheck and synccheck with zero errors/hazards/leaks. Racecheck
uses an 8-GiB tool allowance; the other tools retain 2 GiB, all with no swap.
This does not claim racecheck coverage of the full 144-row workload. CUDA owners
release after each run. Reusing the existing numerical-law type produces
byte-identical default and candidate CUDA to the physical `v2` exports.

Affected package tests pass **111/111**, and the two added serial-unsupported
regressions pass with their package tests **17/17**. Both offline runner suites
pass. Local warning-denied native check succeeds with the unchanged toolchain
exclusions; the prior clean full-suite TLS blocker is not waived. Complete-layer
propagation and full-model correctness remain pending; local bitwise agreement
is not a media-quality or throughput result. No reference or tolerance changes.

## Matched rotary arithmetic isolation — 2026-10-12

An offline intervention now separates generated frequency/trigonometry from
complex-product rounding. Every route consumes the same captured native sparse
attention, original complex64 frequencies and unchanged installed-original
inverse output. The actual exported default renderer reproduces its retained
production output **bitwise**, so this is not a substitute baseline or a new
reference. Prefix channels remain bitwise unchanged and all outputs are finite.

| Matched inverse route | Different BF16 / 4,718,592 | Relative L2 | Maximum error |
| --- | ---: | ---: | ---: |
| Current generated frequencies, separate products | 22 | 0.000359987% | 0.00390625 |
| Literal original frequencies, separate products | 17 | 0.000359987% | 0.00390625 |
| Generated frequencies, measured complex FMA ordering | 11 | 0.000086529% | 0.0009765625 |
| Literal original frequencies, measured complex FMA ordering | **0** | **0%, bitwise** | **0** |

The measured product is `real=fma(left,cosine,-round(right*sine))`,
`imag=fma(right,cosine,round(left*sine))`, after conjugating sine for inverse
rotation. Fusing the opposite products is **not equivalent** and can increase
error. The generated-frequency contracted path reduces local inverse error
about **76.0%**, but retains eleven differing values. The exact literal-frequency
control proves two distinct contributions; it does not mean runtime frequency
generation is already corrected or that full-model parity is achieved.

`RotaryProductLaw` now owns the explicit device-neutral rounding choice in
precision IR. KV rotation, standalone tensor inverse/forward rotation and
compact-cache lowering consume that immutable law through one private CUDA
renderer. Separate products remain the default and the exported default
inverse source is byte-identical to its previous source. The actual
`ComplexFloat32Fma` renderer matches the measured generated-frequency FMA
intervention exactly. Ordinary model builders retain the default; generated
frequency correction, forward/YaRN physical coverage and full-model comparison
remain required.

The explicit product law now propagates through decoder slice/block,
attention envelope, main query/KV rotation, pooled attention-cache rotation
and learned-output inverse rotation. A regression initially found only six of
the ten expected product consumers: grouped output had a duplicated inline
inverse implementation which **ignored the IR law**. It now reuses
`rotary_tensor_source`, removing the duplicate lowering rather than changing
the test expectation. The test covers two window and two compressed layers,
all ten main-attention product consumers, unchanged default policy and
unchanged expert-bank bytes. Indexer/Hadamard numerical policy is separate
and is not silently changed by this main-attention option.

The offline `replay-layer-complex` / runner `prepare-complex` route selects
tiled attention plus the explicit complex product for a matched complete-layer
test. Existing replay modes and ordinary serving retain their original policies.
The committed `1e86e56e` ARM executable now completes the matched one-row
layer-0 replay on `.179`, retaining all 144 original predecessors/token IDs,
unchanged model weights and the same reference. It exits zero with empty
application stderr and releases CUDA.

| Propagated frontier | Tiled + separate products | Tiled + complex products |
| --- | ---: | ---: |
| Rotary query relative L2 | 0.000226269% | 0.000011243% |
| Sparse-attention exact BF16 / values | 4,718,580 / 4,718,592 | **4,718,592 / 4,718,592** |
| After inverse exact BF16 / values | 4,718,570 / 4,718,592 | **4,718,592 / 4,718,592** |
| Grouped output-A relative L2 | 0.004923267% | 0.002497892% |
| Output-B / attention branch relative L2 | 0.027743304% | 0.009394881% |
| Four-stream residual relative L2 | 0.021144838% | 0.010978451% |
| FFN input relative L2 | 0.041244792% | 0.024279547% |
| FFN output relative L2 | 0.227111820% | 0.201573955% |
| Complete layer relative L2 | 0.175669656% | **0.155807089%** |

The attention and inverse frontiers now agree **bitwise** for this replay,
despite 139 tiny query differences remaining. Remaining layer drift first
appears at output-A (324 differing values on identical attention operands),
then amplifies through output-B, residual normalization and FFN. Complete-layer
error falls another **11.3%**, about **58.4% below** the original 0.3749400%
baseline. This is a real propagated numerical improvement, not full-model
parity or a throughput claim. The earlier isolated inverse's eleven residual
differences were on different sparse-attention operands and are not erased
or relabeled by this complete-layer result.

Native root: `/tmp/lunaflux-deepseek-production-layer-0-complex-rows1-20261012-v1`.
Boundary reports: `/tmp/lunaflux-deepseek-layer-0-complex-frontiers-20261012-v1`.
A separate physical default-policy replay at
`/tmp/lunaflux-deepseek-production-layer-0-default-shared-rows1-20261012-v1`
remains bitwise equal to the retained complete-layer default output from
`production-layer-0-components-20261012-v1`. Sharing the output inverse renderer
therefore preserves observed ordinary-policy arithmetic. All arrays are finite.
Clean commit-pinned ARM decoder regressions pass 15/15; local query/KV, grouped
source, checkpoint command and replay runner tests pass 18/18, 3/3, 4/4 and 2/2.
The main-attention law is not applied to indexer/Hadamard arithmetic; an explicit
complex-product request for an indexer cache is rejected instead of ignored.
Ordinary serving remains unchanged. Compressed-path and full-model numerical
corrections, remaining output-A drift and MiniMax media quality are unfinished.

Root: `/tmp/lunaflux-deepseek-rotary-arithmetic-20261012-v3` on `.179` and
locally; runner: `scripts/probe-deepseek-rotary-arithmetic.mbtx`.
Ordinary execution, full memcheck/leak, racecheck and synccheck all finish with
zero errors/hazards/leaks and released CUDA ownership. Controls are bounded
to 2 GiB, except racecheck's 8-GiB tool allowance; all disable swap. Precision
tests pass 51/51, rotary source tests 6/6 and runner regression 1/1. Local
warning-denied native check succeeds; generated interfaces are reviewed.
The previous clean whole-module TLS compilation blocker is not waived.

## Complete-layer tiled-accumulation intervention — 2026-10-12

Commit `1d509053` adds an explicit backend-neutral `TiledFloat32` accumulation
law, consumed by the window/compressed CUDA lowerings and diagnostic decoder
route. Ordered accumulation remains the ordinary serving default. The clean
commit-pinned ARM build succeeds; the physical replay uses the same original
144 four-stream predecessors, one-row submissions, literal IDs and weights
as the earlier ordered layer-0 replay. It completes, exits zero, has empty
application stderr and releases its CUDA context.

| Propagated layer-0 boundary | Earlier ordered relative L2 | Tiled relative L2 |
| --- | --- | --- |
| Normalized branch input | 0% | 0% |
| Sparse attention | 0.006288009% | 0.000421985% |
| After inverse rotation | 0.007179415% | 0.000574527% |
| Grouped output-A | 0.031320318% | 0.004923267% |
| Attention output-B | 0.138442652% | 0.027743304% |
| Four-stream attention residual | 0.0932574% | 0.021144838% |
| FFN input | 0.1506909% | 0.041244792% |
| FFN output | 0.5006931% | 0.227111820% |
| Complete layer output | 0.3749400% | 0.175669656% |

The complete-layer error norm falls about **53.1%**, and the attention branch
about **80.0%**. This establishes propagation of the local attention correction,
not full correctness or a performance result. All arrays are finite, but only
2,128,390 of 2,359,296 complete-layer BF16 outputs are exact. Rotary queries
still have the same 151 differences as before; low-rank queries and simulated
KV are exact. Downstream FFN differences therefore include input drift.
Neither a relaxed tolerance nor an altered reference was used.

Native root: `/tmp/lunaflux-deepseek-production-layer-0-tiled-rows1-20261012-v1`.
Original root: `/tmp/lunaflux-deepseek-original-layer-stream-inverse-20261012-v1`.
Boundary reports: `/tmp/lunaflux-deepseek-layer-0-tiled-frontiers-20261012-v1`.
`scripts/compare-deepseek-layer-frontiers.mbtx` reproduces every boundary and
per-row error without treating propagated differences as local operator errors.
Full-model correctness remains unresolved. The clean commit-pinned ARM
affected-package tests pass **96/96**: precision IR 51, shared-window source 7,
compressed source 7, query/KV integration 17 and decoder integration 14.
The whole-module warning-denied native suite instead stops during compilation
at existing `runtime/remote_tls/channel.mbt:174`: `Tls::server` is marked
`alert_internal` by this toolchain/dependency version. This is not an attention
assertion failure and does not justify suppressing warning 14 or claiming the
full suite passed. The unrelated local TLS/toolchain migration is preserved.
Exact failed user unit: `lunaflux-checkpoint-native-suite-tiled-20261012-v1.service`
on `.178`; its memory peak is 7.5 GiB and swap peak is zero.

The same commit also completes a matched **layer-3 compressed** one-row control.
Its ordered relative L2 is **0.231644100%**, versus **0.232746659%** tiled:
this is a small **regression**, not an improvement. Both receive the same
original predecessor and retain the same compressed state. The earlier
six-row ordered result has the same aggregate error, but is not substituted
for this new one-row control. A pure-window local bitwise match therefore does
not justify blanket replacement of every attention path.

Roots: `/tmp/lunaflux-deepseek-production-layer-3-ordered-rows1-20261012-v2`
and `/tmp/lunaflux-deepseek-production-layer-3-tiled-rows1-20261012-v1`.
Both user units terminate successfully and release CUDA. Desktop disk exhaustion
prevented writing the tiled run's console transaction after remote completion;
separate `terminal-recovery` systemd properties/journal and the untouched
captured tensors are preserved. Empty application stderr is **not** claimed
for that missing console capture. An earlier ordered preparation failed on
desktop disk space before execution; its partial root is preserved as v1.

The bounded offline bridge calls unmodified original `Block.forward`, loading
one complete layer at a time. It retains that layer's real KV state, all four
residual streams and the literal 144 prompt IDs. After each layer, it stores
every row before retiring the original layer and weights. No original kernel
is replaced by LunaFlux and no reference dependency enters production.

Two complete 43-layer TP1 runs finished on Spark `.179`, exited zero without
OOM, and released CUDA owners:

- `/tmp/lunaflux-deepseek-original-layer-stream-full-20261011-v1`: 144 one-row
  submissions per layer.
- `/tmp/lunaflux-deepseek-original-layer-stream-prefill-20261011-v1`: one
  144-row prefill submission per layer.

The initial embedding agrees bitwise with the earlier original four-stream
capture. Full hidden tensors are 4,718,592 bytes per boundary. Captured means
refer to the final literal prompt position (143), not padded rows or generated
token frontiers. The two runs preserve both original submission modes; the
compressor only supports a full initial prefill or later one-row increments,
so the bridge does not pretend it supports six-row continuation chunks.

## What these controls show

| Matched position-143 state, layer 40 | Relative L2 |
| --- | --- |
| Native PP2, physical rows 6 versus original TP1 incremental | 21.1670% |
| Native PP2 versus original TP1 full prefill | 20.9906% |
| Original TP1 incremental versus original TP1 full prefill | 14.5487% |
| Original TP1 incremental versus original TP2 incremental | 17.2535% |
| Original TP1 full prefill versus original TP2 incremental | 17.2087% |

Native versus original TP1 incremental at layers 41/42 remains
21.0974% / 17.7848%. All compared tensors are finite. These differences are
**not additive** and do not demonstrate that the native gap is acceptable or
fixed. The older TP2 sampler reference also submits one logical row at a time:
the incremental TP1-versus-TP2 comparison controls submission shape. Comparing
TP1 full prefill to that TP2 reference changes both topology and submission
shape, and must not be described as an isolated topology comparison. Actual original
arithmetic also varies considerably across these configurations; a single
retained-state norm cannot identify a production bug by itself.

Neither replay runs the prediction heads or proves generation quality.
The two earlier agreeing greedy tokens remain a separate, narrow observation.
Docker/unit limits are 16 GiB with no swap; they are not a measured complete
CUDA-residency peak. Compiler ThreadSync warnings are retained separately.

## Later-layer matched hyper-connection control

The component runner now accepts a literal predecessor and its actual layer
index. Layer 40 receives `hidden-40.bf16` from the original incremental replay,
including four **different** streams; it does not substitute repeated embeddings.
The original `Block.hc_pre`/RMSNorm and production native renderer consume
identical original checkpoint weights and input bytes.

| Boundary | Exact BF16 / values | Relative L2 |
| --- | --- | --- |
| Collapsed residual | 589,235 / 589,824 | 0.001820752% |
| Normalized branch input | 589,238 / 589,824 | 0.003406400% |
| F32 post controls | not a BF16 comparison | 0.000170275% |
| F32 Sinkhorn combination | not a BF16 comparison | 0.0000675411% |

These small local differences do not explain or eliminate the accumulated
18–21% state gap. The original and native runs terminate; memcheck/leak,
racecheck and synccheck all report zero errors, with no leaked allocations.
Root: `/tmp/lunaflux-deepseek-original-hyper-layer40-20261011-v1`.

## Next executable comparison

`cmd/deepseek_checkpoint` now has an offline `replay-layer` diagnostic. It uses
the existing production `DeepSeekDecoderSlice` for one base
layer, feeds literal original predecessor streams in explicit row batches,
and captures each complete output only after retirement. It does not change
the serving graph, precision laws or acceptance tolerances. Real matched
complete-layer replays now complete on Spark `.179`, using original TP1
incremental predecessor tensors and production native six-row batches:

| Layer | All-row relative L2 | Position-143 relative L2 |
| --- | --- | --- |
| 3 (compression ratio 128) | 0.2316441% | 0.6626280% |
| 40 | 0.0037391% | 0.547154% |

Both runs exit zero and release CUDA owners. Layer 40's small global norm is
dominated by unusually large early-row features; it is not evidence of equally
small error on every row. Layer 3 includes individual rows exceeding 4%.
These controls separate local complete-layer error from the accumulated gap,
but do not isolate submission geometry or prove full-model parity.
Roots: `/tmp/lunaflux-deepseek-production-layer-3-rows6-20261011-v1` and
`/tmp/lunaflux-deepseek-production-layer-40-rows6-20261011-v1`.

First-three-layer replay now accepts literal original I32 token IDs and loads
the actual checkpoint token-to-expert tables through the production index-table
uploader. It does not substitute learned routing for token-hash routing. Its
committed ARM executable now completes the first layer using one-row submissions
and all 144 original IDs. Layer 0 aggregate relative L2 is **0.3749400%**;
row 47 is **1.6313947%**. This is the first matched complete-layer discrepancy,
not only a late-layer accumulated effect. Root:
`/tmp/lunaflux-deepseek-production-layer-0-rows1-20261011-v1`.

The next diagnostic records attention input/output, the complete four-stream
attention residual, and FFN input/output after native queue retirement. An
optional original CPython return observer records the same boundaries from
unmodified `Block.forward`, RMSNorm, Attention and MoE, retaining one-row
submissions and token-hash routing. Ordinary serving has no observer, readback,
extra launch or changed arithmetic. The literal one-row layer-0 boundary run
now completes, with all 144 actual token IDs:

| Boundary | Exact BF16 / values | Relative L2 |
| --- | --- | --- |
| Attention input | 589,824 / 589,824 | 0% |
| Attention output | 491,296 / 589,824 | 0.1384427% |
| Four-stream attention residual | 2,298,899 / 2,359,296 | 0.0932574% |
| FFN input | 530,322 / 589,824 | 0.1506909% |
| FFN output | 156,579 / 589,824 | 0.5006931% |

Original root: `/tmp/lunaflux-deepseek-original-layer-stream-boundaries-20261011-v1`.
Native root: `/tmp/lunaflux-deepseek-production-layer-0-boundaries-20261011-v1`.
The first normalized branch input is bitwise identical; divergence first
appears inside attention. The larger FFN discrepancy includes a different
input, so it cannot yet be attributed solely to routed experts. A matched FFN
input intervention is required. The full-model numerical difference is **not
fixed** by adding an observer.

## Literal original FFN row-47 intervention

The layer-0 row with the largest complete-layer relative error is replayed using
the original `ffn-input.bf16` row **47** and its actual original token **45948**.
It does not use the changed native FFN input or substitute BOS routing.
All six original hash-selected routed experts complete on `.179`; both native
matched-down-input and propagated gate/up/down modes agree **bitwise** at every
weighted activation and output (12,288 activation values and 24,576 output values
per mode). Original/native runs exit zero and release CUDA owners.

Root: `/tmp/lunaflux-deepseek-original-expert-row47-20261011-v1`.
This excludes those six routed-expert kernels as the local source of this row's
discrepancy. It does not test the shared expert, final MoE combination, other
rows, or full-model parity. Attention remains the first differing branch.
The retired observer now exposes rotary query/KV, sparse attention and grouped
output-A as well, so that production one-row execution can be compared without
confounding it with the earlier 144-row component geometry.

## Executed one-row attention components — 2026-10-12

Both complete layer-0 executions retain the same original 144 token IDs,
one-row submission geometry and persistent history. Native `e8ad0197` and the
unmodified original complete, exit zero and release their GPU owners.

| Frontier | Exact BF16 / values | Relative L2 |
| --- | --- | --- |
| Attention input | 589,824 / 589,824 | 0 |
| Normalized low-rank query | 147,456 / 147,456 | 0 |
| Rotary query | 4,718,441 / 4,718,592 | 0.000226269% |
| Quantized simulated KV | 73,728 / 73,728 | 0 |
| Sparse attention | 4,717,504 / 4,718,592 | 0.006288009% |
| Grouped output-A | 1,163,683 / 1,179,648 | 0.031320318% |
| Output-B / attention branch | 491,296 / 589,824 | 0.138442652% |

Root: `/tmp/lunaflux-deepseek-one-row-components-20261012-v1`.
This locates the first bitwise difference at query normalization/rotary and
larger amplification across the output projection chain. Downstream inputs
are not identical, so the table alone does not prove an output-B arithmetic
bug. The active window path already selects `OnlineBf16(block_tokens=64)`;
the unused F32-only selected-sparse source is not the executed cause. No
production numerical change is justified by inspecting that unused renderer.

The output projection chain was then replayed with **literal native row 47
attention operands**, preserving its actual token ID. Both native matched and
propagated modes have **8,190/8,192** exact output-A values (relative L2
0.0004143985%) and **4,096/4,096** exact output-B values. Thus the larger
production output-B difference in this row is propagated from attention,
not a demonstrated local output-B kernel bug. This does not establish other
rows or full-model parity. Original unsharded versus serialized two-shard
output-B differs by 0.28165656%; topology remains part of the comparison.
Root: `/tmp/lunaflux-deepseek-original-output-native-row47-20261012-v1`.

The previously untested **shared expert** now consumes the identical original
row-47 FFN input. Its actual production precision plan is exported unchanged;
the offline original calls the installed `Expert` and FP8 linears with the
literal checkpoint payloads/scales. All **2,048** activation values are bitwise
identical; both matched and propagated down modes have **4,095/4,096** exact
outputs, relative L2 **0.0003934733%**, maximum difference **0.0000610352**.
Both original and native complete under bounded no-swap execution and release
CUDA owners. Combined with the six exact routed experts, this rules out a large
local expert-chain error for this input, but does not waive the propagated
full-layer/full-model difference or establish other rows.

Root: `/tmp/lunaflux-deepseek-original-shared-row47-20261012-v1`.
The next intervention must include routing and final combination on literal
inputs, and continue tracing the earliest differing attention operands. No
production arithmetic is changed merely to force a bitwise diagnostic match.

The complete original layer-0 FFN was then evaluated on **all 144 literal native
FFN inputs**, with one-row submissions and the actual token IDs. Original
routing, all routed/shared experts and their final combination are unchanged.
**587,809/589,824** BF16 outputs are exact; relative L2 is **0.00750338%**,
maximum absolute difference **0.00390625**, with no nonfinite outputs. This is
substantially smaller than the **0.5006931%** propagated FFN branch difference
above. It isolates upstream input drift as the dominant contributor in this
layer; it is not a full-model correctness pass or a justification for changing
the reference tolerance. Original execution exits zero and releases CUDA.

Root: `/tmp/lunaflux-deepseek-original-layer-stream-native-ffn-20261012-v1`.

### Identical native attention and output operands

The unchanged original sparse-attention implementation now consumes all 144
literal production rotary queries, simulated KV values and checkpoint sink
biases. Neither side repeats rotary or quantization. Current native attention
has **4,717,516/4,718,592** exact BF16 outputs and **0.006273834%** relative L2.
An offline intrinsic-exp experiment measures **0.006254344%** but has three
fewer exact outputs: this negligible change does not justify a production fix
and is **not selected in serving**. Both runs release their GPU allocations.
Root: `/tmp/lunaflux-deepseek-window-operands-native-20261012-v2`.

The output chain is separately replayed on all 144 literal native inverse-RoPE
attention operands. The original retains **one-row GEMM submissions**, not a
replacement 144-row prefill, and uses the unchanged grouped einsum/FP8 linear.

| Intervention | Exact BF16 / values | Relative L2 |
| --- | --- | --- |
| Output-A on identical attention input | 1,179,321 / 1,179,648 | 0.002753354% |
| Output-B on identical original output-A | 589,815 / 589,824 | 0.000026891% |
| Output-B consuming native output-A | 588,515 / 589,824 | 0.011284933% |

Root: `/tmp/lunaflux-deepseek-original-output-native-all144-20261012-v1`.
These isolate local output-B arithmetic from amplification of earlier input
differences; they do not establish full-model parity. The optional serialized
two-shard control remains explicitly batched and is not a TP2/NCCL reference.
The earlier 18–21% full-model difference remains unresolved. No altered
reference or relaxed tolerance is substituted for a faithful comparison.

```sh
moon run scripts/run-deepseek-original-layer-stream.mbtx prepare NEW_ROOT 43 [--prefill]
moon run scripts/run-deepseek-original-layer-stream.mbtx run ROOT 43
moon run scripts/run-deepseek-original-layer-stream.mbtx collect ROOT 43
moon run scripts/replay-deepseek-original-hyper.mbtx prepare NEW_ROOT IDS_TEXT --attention-layer 40 ORIGINAL_HIDDEN_40_BF16
```

Four hyper-runner regressions and the streamed-runner regression pass.
Warning-denied native checking and the full suite pass **5,015/5,015**, with
the existing toolchain migration-warning exclusions and no added exclusions.
Offline reference bridges compile with `-Wall -Wextra -Werror` on ARM.

## Original inverse-RoPE frontier — 2026-10-12

The previous production observation skipped the inverse rotation between
sparse attention and grouped output-A. The offline original observer now
captures the complete caller-owned attention row after the original in-place
inverse return, not merely its rotated suffix. Original math, token IDs and
one-row submission geometry remain unchanged. A fresh layer-0 replay at
`/tmp/lunaflux-deepseek-original-layer-stream-inverse-20261012-v1` exits zero
and releases CUDA; all 2,359,296 final four-stream values remain **bitwise
identical** to the earlier original observation.

| Propagated frontier | Exact BF16 / values | Relative L2 |
| --- | --- | --- |
| Sparse attention | 4,717,504 / 4,718,592 | 0.006288009% |
| After inverse RoPE | 4,717,521 / 4,718,592 | 0.007179415% |
| Grouped output-A | 1,163,683 / 1,179,648 | 0.031320318% |
| Output-B / attention branch | 491,296 / 589,824 | 0.138442652% |

These frontiers still carry differing upstream operands; they are **not**
local matched-input inverse-rotation errors. The added observation excludes a
large missing aggregate jump at this boundary in layer 0, but does not prove
the inverse implementation bitwise or resolve the accumulated full-model gap.

### Matched native sparse-attention inverse intervention

The installed original `apply_rotary_emb(..., inverse=True)` now consumes the
literal **native** sparse-attention output for all 144 rows, with one-row
submissions and the original pure-window complex64 frequency table retained
from the original replay. It does not repeat attention, change query/KV inputs,
or substitute a new frequency-generation algorithm. Native inverse output and
the original inverse therefore begin from identical attention operands.

Root: `/tmp/lunaflux-deepseek-original-inverse-native-all144-20261012-v1`.
Both computations retain the complete 64×512 row, including the 448 channels
outside the rotary suffix. **4,718,570/4,718,592** BF16 values agree; relative
L2 is **0.0003599874%**, maximum difference **0.00390625**, all values finite.
Original execution exits zero, releases CUDA and uses a 4-GiB/no-swap limit.
The ARM bridge compiles with `-Wall -Wextra -Werror`; its runner regression passes.

This isolates a small local inverse difference, not merely the propagated
comparison above. It does not eliminate or explain the accumulated 18–21%
full-model gap. The earlier matched output projection input is also verified
byte-identical to the native inverse capture, so those output-chain reports
did not accidentally feed pre-inverse attention into either projection.
Reports: `/tmp/lunaflux-deepseek-one-row-inverse-components-20261012-v1`.
No production kernel or arithmetic law was changed by this observation.

## Original generated sparse-attention arithmetic — 2026-10-12

The offline original adapter now has an optional `source` observation. Its
ordinary three-argument numerical replay is unchanged. Observation obtains
the generated CUDA from the same installed `sparse_attn_kernel(16,512,scale)`
factory used by the original wrapper's four sixteen-head calls; it does not
substitute another attention implementation or alter the original inputs.

The bounded source run completes in
`/tmp/lunaflux-deepseek-window-source-20261012-v5` on `.179`. All
**4,718,592 original BF16 outputs are bitwise identical** to the earlier
literal-native-operand original replay. CUDA owners retire. The C++ bridge
compiles with `-Wall -Wextra -Werror`; the MoonBit runner regression passes.
Limits are 8 GiB/no swap. The tiny user-unit peak is the Docker client, not
complete GPU/container residency. Installed ThreadSync warnings are retained.

The retained `original-window.cu` shows:

- QK: BF16/F32 `mma_sync` with 16×8×16 instructions, iterating 32 sixteen-wide
  reduction slices across the 512-channel head.
- Softmax: 64-key online blocks; per-thread two-value partial sums, then
  explicit cross-warp and four-lane `AllReduce` stages. Its denominator
  update is expressed as `old_sum * rescale + tile_sum`.
- P is cast to BF16 before PV. PV uses four sixteen-key MMA slices per tile,
  rather than the current native per-key scalar FMA loop.
- The generated original source uses **`expf`**, not an explicit `exp2`
  algorithm. The existing unsuccessful intrinsic-exp experiment therefore
  does not establish exponent choice as the full-model cause.

The current native online64 source uses ordered scalar QK accumulation and a
serial probability sum. Thus agreement on BF16 probability storage and tile
length does not mean agreement on QK/PV accumulation or denominator reduction
order. These are concrete source differences to isolate; this source audit
does not measure their individual error contributions, prove the compiler's
final FMA contraction, or resolve the accumulated 18–21% gap. Generated CUDA
is **not** selected-kernel SASS or a hardware-counter capture.

Earlier `v2`–`v4` attempts to retain the loaded binary fail only at metadata
export: the installed TVM-FFI adapter has no Cython `libpath`, its CUDA module
does not support object-file writes, and it has no Python `save_to_bytes`
method. Those outputs/logs remain preserved. No rebuilt binary is mislabeled
as the originally loaded kernel; unsupported binary extraction is removed
from the committed diagnostic. Production arithmetic and selection remain
unchanged. This observation runs independently of the active MiniMax
complete-stack experiment on `.178`.

```sh
moon run scripts/inspect-deepseek-window-source.mbtx prepare NEW_SOURCE_ROOT RETIRED_OPERAND_REPLAY_ROOT
moon run scripts/inspect-deepseek-window-source.mbtx run NEW_SOURCE_ROOT RETIRED_OPERAND_REPLAY_ROOT
moon run scripts/inspect-deepseek-window-source.mbtx collect NEW_SOURCE_ROOT RETIRED_OPERAND_REPLAY_ROOT
```

## Explicit tiled F32 candidate — local attention difference eliminated

The precision IR now records `AttentionAccumulationLaw` independently of P
storage. The ordered default stays unchanged; `TiledFloat32` explicitly permits
F32 reassociation for QK/PV and softmax sums. CUDA-only physical lowering uses
matrix-MMA, shared KV operand reuse, parallel reduction and row-owned F32
fragment rescaling. It does not infer opaque fragment register ownership.
The policy flows through the complete decoder constructor chain and both
window/compressed source launch metadata. No NVIDIA type enters the semantic
policy or scheduler. The diagnostic `replay-layer-tiled` explicitly selects it
for both export and execution; ordinary serving remains ordered/default.

Fresh root `/tmp/lunaflux-deepseek-matrix-window-20261012-v3` consumes the
identical captured native post-rotary Q, simulated KV and original sink bias.
Across **144 rows × 64 heads × 512 channels**, the candidate now matches the
installed original **4,718,592/4,718,592 BF16 outputs**, zero relative L2,
zero maximum error and zero nonfinite values. The earlier ordered candidate
had 1,076 differing outputs and 0.006273834% relative L2 on these operands.
This is an actual eliminated local difference, not a relaxed tolerance or
replacement original implementation.

The same compiled kernel produces bitwise-identical all-row prefill and
one-row incremental results, including real ring rollover after 128 positions.
An 18-head/80-column shape control checks both physical tail dimensions and
prefill/incremental equality. That tail control is not an independently
reconstructed original-model mathematical oracle.

Memcheck/full leak check reports zero errors and zero leaked bytes; racecheck
zero errors/warnings/hazards; synccheck zero errors. These cover the ordinary
and incremental candidate paths and the tail control, not every possible
geometry. All owners release. The failed `v1` C++ mixed-`auto` compile is
preserved, then corrected. `v2` ordinary execution and memcheck pass, but
racecheck is killed at its 2-GiB instrumentation cap. `v3` repeats in fresh
directories with an **8-GiB racecheck-only cap**, no swap; ordinary, memcheck
and synccheck retain 2-GiB caps. This is not a higher inference memory budget.
Reported small wrapper peaks are not complete CUDA/tool residency.

The higher-level default and explicit ordered exports remain byte-identical;
regressions verify policy propagation into window and compressed source and
grouped launch geometry. Complete-layer, compressed-read-set physical parity
and the accumulated 18–21% discrepancy remain pending. This local match alone
does not select the candidate for production or establish generation quality.

The affected decoder/kernel checks pass with the existing toolchain warning
exclusions, and source propagation tests cover four layers including a
compressed layer. The attempted full local native suite reports
**5,035/5,040**, with five failures explicitly `No space left on device` while
writing checkpoint fixtures, not failed numerical assertions. This desktop
has less than 1 GiB available; concurrent large fixtures exhaust it. A complete
clean-host rerun is still required. No exclusions, fixture shapes or numerical
tolerances are changed to hide those failures, and unrelated dirty toolchain
files remain untouched by the task commits.
