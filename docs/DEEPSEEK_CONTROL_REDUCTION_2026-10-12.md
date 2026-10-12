# Matched DeepSeek control reduction — 2026-10-12

This follows the output-A correction and residual-publication attribution in
[the layer-stream report](DEEPSEEK_ORIGINAL_STREAM_REPLAY_2026-10-11.md).
It is a numerical investigation, not a throughput or full-model quality result.

The original bridge observes `mixes` and `rsqrt` at the return of unmodified
`Block.hc_pre`. All 144 literal predecessor rows use the original one-row GEMM
geometry. The observer changes no arithmetic; post/combination outputs remain
bitwise identical to the previous no-internals bridge. Original root:
`/tmp/lunaflux-deepseek-original-hyper-layer0-rows1-20261012-v2` on `.179`.

## Isolated causes

The diagnostic actual-renderer production control reproduces both native F32
control arrays bitwise. Its inverse RMS differs on **all 144 rows**, relative
L2 **0.000320213%**. Projection results after normalization differ on 3,455/3,456
values, relative L2 **0.000365954%**. Post-control error is **0.000590706%**.

| Intervention, identical original input | Post-control relative L2 |
| --- | ---: |
| Actual ordered production source | 0.000590706% |
| Original inverse only; native ordered projection | 0.000194052% |
| 32 striped projection partials; native inverse | 0.000388130% |
| Original normalized mixes; native nonlinear controls | 0.000021391% |
| Compensated square sum; ordered projection | 0.000194147% |
| Compensated square sum + 32-partition projection, actual new lowering | **0.000021539%** |

Original-value substitutions are **offline causal controls**, never serving
data. Fixing projection alone leaves normalization error. A more accurate
square sum alone leaves projection error. Even exact original mixes leave a
smaller nonlinear-control/Sinkhorn difference, which is not waived.

Compensated F32 square summation makes **141/144 inverse values bitwise exact**,
relative L2 **0.000001076%**; three differ by at most 9.54e-7. A 256-partition
square tree matches 140/144 and is not more accurate in this capture. F32 FMA
changes projection details, but has no meaningful additional post-control norm
benefit over separated products here. No intrinsic-exp production selection.

## Actual compiler/decoder implementation

The existing `SquareReductionLaw` and `DotReductionLaw` now flow through pure
`HyperConnectionPrecision` into private CUDA lowering. No extra numerical IR,
model-specific kernel branch or vendor library is introduced. The refined
candidate uses compensated square sums and `StripedTreeFloat32(partitions=32)`
with separate F32 product rounding. Both attention and FFN envelopes receive
the policy; tests cover four window/compressed layers and all eight consumers.
The numerical source contract explicitly carries the selected laws instead of
mislabeling refined code as ordered. Workspace and weight budgets stay equal.

Ordinary callers retain ordered control laws. Their generated source is
byte-identical to the retained source, independently checked with `cmp`.
The offline `prepare-refined`/`replay-layer-refined` route selects the candidate
at both export and execution. No literal reference or Python/Torch enters it.

Actual source probe root:
`/tmp/lunaflux-deepseek-hyper-arithmetic-20261012-v4`.
It observes **141/144** exact inverse values, **1,025/3,456** exact normalized
mixes, **15/576** exact post controls and **572/2,304** exact combinations.
Post norm error decreases about **27.4×**, mix norm about **35.8×**; combination
error is **0.000004574%**. These F32 arrays are not yet bitwise parity.

All twelve diagnostic routes and the actual refined source finish memcheck
with zero errors/leaks, racecheck with zero hazards/errors/warnings and
synccheck with zero errors. These all-144-row control probes use 2-GiB/no-swap
units and release CUDA. The earlier `v1` compile failure from a diagnostic
helper-name replacement is retained; `v2` corrects it and adds a regression.
`v3` isolates square reductions; `v4` uses the actual new renderer.

## Commit-pinned complete layer

The `0be5ea5b` ARM release build passes on the clean source, then the same 144
original one-row layer-0 replay completes, zero exit, empty application stderr,
no swap and CUDA closed. Logs explicitly record compensated control squares
and striped control dots in addition to tiled attention/complex rotary/striped
output-A. It is not the default policy. Root:
`/tmp/lunaflux-deepseek-production-layer-0-refined-rows1-20261012-v1`;
reports: `/tmp/lunaflux-deepseek-layer-0-refined-frontiers-20261012-v1`.

| Propagated boundary | Before control refinement | After control refinement |
| --- | ---: | ---: |
| Attention input | Bitwise | Bitwise |
| Sparse attention, inverse, output-A | Bitwise | Bitwise |
| Output-B differing BF16 / 589,824 | 11 | 11 |
| Residual differing BF16 / 2,359,296 | 2,386 | **64** |
| Residual relative L2 | 0.009148849% | **0.000704269%** |
| FFN-input differing BF16 / 589,824 | 2,131 | **56** |
| FFN-input relative L2 | 0.021675558% | **0.001278160%** |
| FFN-output relative L2 | 0.198810072% | **0.031858964%** |
| Complete-layer relative L2 | 0.153870184% | **0.025374703%** |

Complete-layer error falls **83.5%** from the previous candidate, about **93.2%**
from the first 0.3749400% matched baseline. This proves propagated numerical
benefit, not a speedup. It retains 12,377 differing complete-output BF16 values;
remaining nonlinear controls, FFN local arithmetic and propagated drift are
not waived. Affected tests pass 114/114 plus the new metadata/unsupported-law
package tests 12/12; the offline mode tests pass 2/2. Native check succeeds with
unchanged toolchain exclusions. The whole-module clean TLS blocker remains.

The residual publisher itself is unchanged. Speed, compressed-path parity,
full-model 18–21% drift and MiniMax media/encoder differences remain open.

The aggregate improvement is not a per-row parity claim. Position 143 improves
from **0.324717%** complete-output error to about **9.54e-8%**. But row 47 moves
from **0.246268% to 0.253464%**, a small regression despite FFN-input error
decreasing from **0.010825% to 0.001572%**. Its FFN-output error increases from
**0.322967% to 0.336369%**. A matched latest-input FFN intervention is required
to distinguish local arithmetic from quantization/routing sensitivity. These
rows remain visible; the new candidate is not a blanket serving selection.

The same committed executable also completes the original-input **compressed
layer 3**, one-row geometry, in
`/tmp/lunaflux-deepseek-production-layer-3-refined-rows1-20261012-v1`.
Complete-output relative L2 is **0.155833791%**, versus earlier ordered
**0.231644100%** and tiled-only **0.232746659%**. This is about **32.7% less
error** than ordered, not parity. Only **1,210,863/2,359,296 BF16 values** match;
all are finite, maximum absolute error 0.018493652. The candidate combines
control, rotary and output changes, so this is not an isolated attribution to
one of those changes. It exits zero and closes CUDA; default selection stays
unchanged and a dedicated compressed-attention intervention remains necessary.

## Latest-input FFN intervention

Root `/tmp/lunaflux-deepseek-original-layer-stream-refined-ffn-20261012-v1`
executes the installed original MoE on the **exact latest native FFN inputs**,
all 144 original IDs and one-row geometry. The original weights/arithmetic stay
unchanged; this is a causal input intervention, not a replacement reference.
Compared with native FFN on those identical inputs, **587,960/589,824 BF16
values match**, relative L2 **0.007706382%**, maximum error **0.00390625**.
Original runs retire, exit zero and release CUDA; installed warnings remain.

In **row 47, all 4,096 FFN outputs are bitwise identical** on the latest shared
input. Running that same original FFN on native versus original inputs produces
aggregate relative L2 **0.030914458%**, close to the propagated **0.031858964%**.
Thus this row's larger difference and small regression are input/quantization
sensitivity, **not** a demonstrated native FFN kernel bug. The original input
comparison is retained; local error and propagated error are not conflated or
assumed to add linearly. Other rows retain a small local FFN discrepancy. The
next priority is remaining control/publication-input differences, plus separate
compressed-attention isolation; blindly rewriting row-47 experts is unjustified.

## Nonlinear and affine isolation

Fresh root `/tmp/lunaflux-deepseek-hyper-arithmetic-20261012-v5` adds seven
offline counterfactuals to the twelve retained routes. All 144 original streams,
F32 weights, scales and bases remain identical. Default controls reproduce the
retained native arrays bitwise. Original normalized mixes are substituted only
in explicit offline causal controls, never in the actual candidate.

| Identical original mixes, post controls | Exact F32 / 576 | Relative L2 |
| --- | ---: | ---: |
| Stable sigmoid, separate affine products | 18 | 0.000021391% |
| Direct reciprocal sigmoid only | 303 | 0.000021347% |
| Fused affine multiply-add only | 154 | 0.000000648% |
| Direct reciprocal sigmoid + fused affine | **576** | **0%** |

This isolates a real numerical-order mismatch: strict `expf`, direct reciprocal
sigmoid and a single F32 affine FMA reproduce every original post control.
No approximate/intrinsic exponential is selected. With native refined mixes,
the same pair gives **430/576** exact post controls, relative L2
**0.000003151%**, versus 15 exact and 0.000021539% before. Remaining mix error
is visible; this is not whole-layer parity.

Applying FMA indiscriminately to the matrix logits does **not** improve
Sinkhorn combination: refined combination relative L2 increases from
0.000004574% to 0.000005847%. Vector affine and matrix affine must not be
conflated. An `rsqrtf` intervention raises exact inverse count from 141 to 142
of 144, but leaves post error essentially unchanged; it is not a demonstrated
solution to downstream drift.

All nineteen routes finish memcheck/leak, racecheck and synccheck with zero
errors/hazards/warnings in 2-GiB/no-swap units and release CUDA. Serving sources
and policy are unchanged. Next: represent vector affine and sigmoid arithmetic
explicitly in numerical IR, retain matrix arithmetic, and measure propagation.

The implementation now represents `Float32AffineLaw` and
`SigmoidEvaluationLaw` in the existing precision IR. `HyperConnectionPrecision`
records **vector** affine rounding independently of matrix logits. Pure decoder
planning forwards both laws through attention and FFN envelopes to their private
CUDA lowering; no new buffers, queues, host effects or reference inputs appear.
Default source remains byte-identical to the preceding capture. The explicit
`replay-layer-nonlinear` mode selects direct sigmoid and fused vector affine on
top of the refined reduction candidate, at both export and execution.

Actual renderer root `/tmp/lunaflux-deepseek-hyper-arithmetic-20261012-v6`
reproduces the **430/576** exact post controls and **0.000003151%** relative L2.
Combination remains **572/2304** exact and **0.000004574%** relative L2,
unchanged from the refined baseline: matrix-FMA regression was not incorporated.
All twenty routes pass memcheck/leak, racecheck and synccheck with zero reports
under 2-GiB/no-swap limits. Native check passes with unchanged toolchain warning
exclusions; affected native tests **118/118**, runner tests **3/3**. Tests verify
both window/compressed layers and both envelopes receive the vector policies.
Complete-layer propagation and full-model quality remain separate next gates.

## Commit-pinned nonlinear propagation

Clean ARM build of `0c17d345` succeeds under the same 2-GiB/no-swap CPU build
limit (716.2-MiB peak), without using the active media GPU. Actual layer replay
logs explicitly show `FusedMultiplyAdd` vector affine and `DirectReciprocal`
sigmoid at export and execution; ordinary serving is still unchanged. The
144 matched original one-row layer-0 replay exits zero, closes CUDA and records
no application stderr. Root:
`/tmp/lunaflux-deepseek-production-layer-0-nonlinear-rows1-20261012-v1`;
frontiers `/tmp/lunaflux-deepseek-layer-0-nonlinear-frontiers-20261012-v1`.

| Propagated boundary | Refined reduction only | Plus vector nonlinear laws |
| --- | ---: | ---: |
| Attention input, sparse attention/inverse/output-A | Bitwise | Bitwise |
| Attention residual differing BF16 | 64 | 53 |
| Attention residual relative L2 | 0.000704269% | 0.000762259% |
| FFN-input differing BF16 | 56 | 61 |
| FFN-input relative L2 | 0.001278160% | 0.003477828% |
| FFN-output relative L2 | 0.031858964% | 0.019480324% |
| Complete-layer relative L2 | 0.025374703% | **0.016411788%** |

Complete-layer norm decreases **35.3%** from refined, about **95.6%** from the
initial 0.3749400% result. Row 47 changes from 0.253464% to **0.00428665%**,
about **59.1× less error**. The exact-output count is 2,352,495/2,359,296. This
is a numerical benefit, not throughput or full-model parity. Increased
aggregate FFN-input norm and remaining 6,801 complete-output differences remain
visible: BF16/FP4 thresholds make propagation nonlinear.

Compressed layer 3 also completes in
`/tmp/lunaflux-deepseek-production-layer-3-nonlinear-rows1-20261012-v1`, but
slightly **regresses** from 0.155833791% to **0.156272325%**, 1,202,877 exact
BF16 values. Its attention branch outputs remain bitwise identical to the
preceding refined run; altered vector controls affect the subsequent FFN
input/trajectory. This rules out blanket enabling this policy. The remaining
full-model 18–21% drift is not fixed by either layer experiment.

Fresh installed-original observation now captures all eleven compressed layer-3
frontiers at `/tmp/lunaflux-deepseek-original-layer-stream-compressed-frontiers-20261012-v1`.
It executes original layers 0–3 sequentially, one complete block resident at a
time, under 16-GiB/no-swap limits. Literal predecessor `hidden-3` and complete
`hidden-4` are both bitwise identical to the retained matched input/reference,
independently checked with `cmp`: the newly observed original does not alter
the comparison. No original arithmetic, weights or tolerances are edited.

### Compressed attention on identical native operands

The installed original attention now runs independently on the exact native
layer-3 attention inputs, with its own fresh retained/compressed/indexer KV and
the unchanged one-row positional sequence. Root:
`/tmp/lunaflux-deepseek-original-layer-stream-compressed-matched-attention-20261012-v1`.
Its observer captures eight complete frontiers without changing original math.
The input arrays match native bitwise; output extents are complete. Reference
substitution remains an offline causal experiment, not a serving mechanism.

| Boundary | Local native/original, identical native inputs | Original native/original input propagation |
| --- | ---: | ---: |
| Normalized low-rank query | 0.000048151% | 0.016786876% |
| Q after RoPE | 0.001580096% | 0.090942609% |
| KV after simulation | Bitwise | 0.027888525% |
| Sparse attention | 0.001824029% | 0.052158547% |
| Output-A | 0.007378225% | 0.126846301% |
| Attention output-B | **0.034442365%** | **0.379403555%** |

The originally observed propagated output-B error is 0.379357699%. Original
attention on native inputs reproduces nearly the same drift. This isolates
upstream input/quantization sensitivity as the larger source here, not a claim
that native attention is exact: local query/output differences remain. These
norms do not add linearly and are not a decomposition into percentage shares.
Reports: `/tmp/lunaflux-deepseek-layer-3-matched-attention-frontiers-20261012-v1`
and `/tmp/lunaflux-deepseek-layer-3-attention-input-propagation-20261012-v1`.

### Layer-3 control collapse versus branch normalization

`/tmp/lunaflux-deepseek-original-hyper-layer3-rows1-20261012-v1` observes
unmodified original `hc_pre` and RMSNorm on the literal layer-3 predecessor,
all 144 one-row inputs and actual layer-3 checkpoint weights. Its normalized
branch input matches the original full-layer observation **bitwise**. Its
native default, memcheck/leak, racecheck and synccheck finish with zero reports.

Fresh `/tmp/lunaflux-deepseek-hyper-arithmetic-20261012-v7` additionally captures
actual collapse and weighted normalization for each of the twenty routes,
without altering the published default or refined source. All twenty finish
bounded memcheck/leak, racecheck and synccheck with zero reports. The actual
refined collapse/normalization reproduces the retained complete-layer-3 native
frontiers, including **63** differing normalized BF16 values. Frontend captures
are separately collected under `frontend/`, without overwriting older outputs.

| Layer-3 control route | Collapse exact BF16 / 589,824 | Normalized exact BF16 / 589,824 |
| --- | ---: | ---: |
| Ordered production arithmetic | 589,212 | 589,196 |
| Refined reduction | 589,759 | 589,761 |
| Refined + vector affine/direct sigmoid | 589,766 | 589,768 |
| Literal original mixes + fused vector affine/direct sigmoid, **offline only** | **589,824** | **589,822** |

The literal-mix intervention makes all post controls and the entire BF16
collapse bitwise exact, while two weighted-normalization outputs still differ.
Thus the residual error is not an unexplained failure of the collapse layout:
normalized control projection and branch RMS arithmetic are separate remaining
frontiers. Layer-3 compensated control squares match only **90/144** original
inverse values (unlike layer 0's 141/144); the `rsqrtf` intervention raises this
to **113/144**, without normalized-branch parity. The existing variants do not
resolve this and are not promoted. Next probes must use the observed original
means/projection operands and exact normalization rounding, not replace model
data or change the comparison tolerance.

### Exact RMS reduction order — 2026-10-12

The new offline observer retains actual original RMSNorm `var` and replays the
installed control square/mean operation on the original flattened `x` operand.
That replay does not feed back into original computation. Source root
`/tmp/lunaflux-deepseek-original-hyper-layer3-rows1-20261012-v2` preserves both
original collapse and normalized outputs bitwise against `v1`.

The installed ATen reduction source explains the missing order: input vectors
use four independent accumulators per partition, joined in component order;
512 partition totals then reduce in a descending binary tree. This is different
from native stripe-256's single accumulator and from a compensated scalar sum.
The new independent GPU probe implements that explicit F32 ordering, not a
Torch production dependency or a copied reference result. Root:
`/tmp/lunaflux-deepseek-rms-arithmetic-20261012-v1`.

| Matched original operands | Control means exact / 144 | Control inverses exact / 144 | Weighted-normalized BF16 exact / 589,824 |
| --- | ---: | ---: | ---: |
| Stripe-256, reciprocal sqrt | 95 | 110 | 589,822 |
| Four-component vector fold, 512 partitions, reciprocal sqrt | **144** | **144** | **589,824** |
| Literal original means, reciprocal sqrt, offline causal control | 144 | 144 | 589,824 |

The vector fold also matches all **144 original branch variances**, while
unmodified collapse/weights and BF16 publication are retained. Seven variants
pass memcheck/leak, racecheck and synccheck, zero reports, 2-GiB/no-swap and
balanced release. This demonstrates the needed RMS arithmetic on these operands,
not control-projection/full-layer/full-model parity or a performance gain.
Next: encode vector-fold and reciprocal-square-root choices in immutable
precision contracts and verify actual lowering and complete-layer propagation.

The implemented alternative now lives in the existing `SquareReductionLaw`
(`VectorizedTreeFloat32`) and a separate `ReciprocalRootLaw`, not another IR
layer or a model/backend switch. Both control RMS and weighted branch RMS
propagate through attention and FFN envelopes. The prefix lowering carries
`launch_threads`; the frame consumes that requirement instead of always
launching 256 threads. Vector tails are explicit. Unsupported numerical laws
are rejected by lowerings that do not implement them, never silently replaced.
Default generated source is independently byte-identical to the retained
pre-change source; matrix arithmetic, global frames and serving defaults stay
unchanged.

Actual generated-source probe
`/tmp/lunaflux-deepseek-hyper-arithmetic-20261012-v8` executes the serial
control fold and cooperative 512-thread weighted norm, not just the independent
prototype. It matches **144/144 original control inverses bitwise** on layer 3.
Actual normalized native input has 589,775/589,824 exact BF16 values, relative
L2 **0.003034660%**, versus previous nonlinear 589,768 and 0.003129244%.
Control projection still differs, so these native inputs are not parity.
All twenty-one actual-source diagnostic routes pass memcheck/leak, racecheck
and synccheck with zero reports, 2-GiB/no-swap and released CUDA. Native check
passes with unchanged warning exclusions; affected native tests **130/130**,
replay/probe runner tests **3/3**. Actual complete-layer propagation remains the
next check; no throughput or full-model benefit is inferred from these counts.

## Complete vector-RMS replay

Clean pinned ARM build of **c0467273** completes. Export and execution both
record vectorized control/branch square folds (512 partitions, four components),
reciprocal control sqrt, striped-32 control dots, direct sigmoid and vector
affine FMA. Both one-row complete-layer replays exit zero, close CUDA and report
no swap; application stderr is empty (systemd wrapper messages are retained).

| Complete matched layer | Nonlinear candidate | Vector-RMS candidate | Exact BF16 / 2,359,296 |
| --- | ---: | ---: | ---: |
| Layer 0 relative L2 | 0.016411788% | **0.015280325%** | 2,353,501 |
| Compressed layer 3 relative L2 | 0.156272325% | **0.156252161%** | 1,203,147 |

Layer 0 improves a further **6.9% in numerical error**, about 95.9% below the
initial 0.3749400% error, not a throughput gain. Its attention input, sparse
attention, inverse and output-A remain bitwise. FFN input differs on 58 values,
relative L2 0.002561457%; FFN output is 0.018099565%. Row 47 remains 0.00428665%
complete-output error. Layer-3 normalization improves 56 to 49 differing inputs,
but low-rank query and all downstream attention outputs are bitwise identical
to the preceding nonlinear candidate. Its complete-layer benefit is tiny and
remains worse than the earlier 0.155833791% refined-reduction result. The source
fix is verified at its RMS boundary, but **does not resolve the larger compressed
projection/input-sensitivity problem**. Do not blanket-enable it.

Roots `/tmp/lunaflux-deepseek-production-layer-0-vector-rms-rows1-20261012-v1`
and `/tmp/lunaflux-deepseek-production-layer-3-vector-rms-rows1-20261012-v1`;
frontiers `/tmp/lunaflux-deepseek-layer-0-vector-rms-frontiers-20261012-v1` and
`/tmp/lunaflux-deepseek-layer-3-vector-rms-frontiers-20261012-v1`.
Actual layer-3 control mixes still match only **1,007/3,456 F32** values,
relative L2 0.000010158%. With RMS now exact on the original control input, the
next isolated target is the **control projection's F32 dot order**, rather than
repeating RMS experiments or blaming layout/cache ownership. Branch weighted
RMS is exact on identical original collapsed input; upstream collapsed-input
differences remain. Full-model 18–21% drift and MiniMax correctness/quality are
still open; ordinary serving stays unchanged.

## Control dot: two reduction levels, not another RMS defect

On the same original layer-3 streams and F32 function weights, a fresh offline
probe computes its own RMS inverse (144/144 bitwise) and compares 64 single-level
dot orders. None matches every original mix. Pedantic cuBLAS GEMV and GEMM
controls match **3,456/3,456 F32 mixes bitwise**; they are diagnostic controls,
not references, production dependencies or substituted model outputs.

The selected vendor execution was captured with the previously authorized
administrator access, in a 4-GiB/no-swap user unit. It uses a **two-stage dot**:
18 groups × 128 partitions per output; each partition retains two independent
FMA accumulators. Group totals reduce by descending halves. The second kernel
adds group totals in four ordered stripes, then joins the four stripes in order.
Selected launch is `(18,1,24)×(128,1,1)`, followed by `(1,1,24)×(128,1,1)`.
The SASS and source-correlated instruction counts corroborate the independent
accumulators, FMA updates and both reduction levels. This is arithmetic order,
not a throughput claim or a general recommendation to increase launch count.

| Native probe on identical original operands | Exact mixes / 3,456 | Relative L2 (%) |
| --- | ---: | ---: |
| Previous stripe-32 separate products | 1,007 | 0.000010158 |
| Two-level 18×128, two components, FMA, four-stripe publication | **3,456** | **0** |
| Same groups/components/FMA, tree-32 publication instead | 2,085 | 0.000006347 |
| Same groups/components/publication, separate products instead | 2,202 | 0.000004699 |
| Sixteen groups instead of eighteen | 1,673 | 0.000007490 |

Roots `/tmp/lunaflux-deepseek-control-dot-20261012-v1`–`v3`. Captured report:
`v2/profile-admin/control.ncu-rep`; imported SASS `v2/profile-source.stdout`.
All 74 routes pass full-144-row memcheck/leak. Native 72-route racecheck and
synccheck pass on **one row**, not all 144; the original full-row race attempt
hit its enforced 2-GiB cap and remains preserved as `v1/racecheck.stderr`.
Unprivileged profiler attempts failed with `ERR_NVGPUCTRPERM`; no machine-wide
driver setting or memory cap was changed. All probe owners have retired.

The implementation expresses this as `DotReductionLaw::GroupedVectorFloat32`:
immutable group/component ownership, explicit multiply-add rounding, local
tree and publication order. It contains no vendor/model names or CUDA thread
semantics. Private control lowering consumes the law; lowerings without this
arithmetic reject it instead of silently reverting. The diagnostic replay
mode selects the observed parameters explicitly, never as a universal/default
policy. Actual generated-source controls and complete-layer propagation must
still verify before claiming the control/frontend discrepancy is fixed.

Actual generated-source `v9` now confirms the control/frontend boundary on all
144 literal original layer-3 predecessors: **3,456/3,456 mixes**, **144/144 RMS
inverses**, **576/576 post controls**, **589,824/589,824 collapsed BF16** and
**589,824/589,824 weighted-normalized BF16** are bitwise exact. This is native
generated arithmetic, not the vendor control or literal-mix intervention.
Sinkhorn combination is still only 501/2,304 F32 exact, relative L2
0.000006108%; residual publication can therefore still differ. All 22 actual
source routes pass full-144-row memcheck/leak, racecheck and synccheck with zero
reports, 2-GiB/no-swap and released CUDA. Default source remains independently
byte-identical. Warning-denied native check and 122 affected native tests pass;
the eight attention/FFN envelopes across window/compressed layers consume the
new grouped dot law in the source regression. Runner tests 4/4 pass. Complete
layers and full-model drift remain separate, pending checks.

### Complete hierarchical-control propagation

Source-pinned ARM build **a6e2ad1a** completes in the isolated 2-GiB/no-swap
CPU unit on `.178`; the active MiniMax GPU run is not interrupted. Complete
matched replays on `.179` record the grouped-dot policy at **both export and
execution**, use the original 144 predecessors/IDs with one-row calls, exit
zero and close CUDA. All compared values are finite; no serving default changes.

| Complete layer | Vector-RMS relative L2 (%) | Hierarchical-controls relative L2 (%) | Exact BF16 / 2,359,296 |
| --- | ---: | ---: | ---: |
| Layer 0 | 0.015280325 | **0.015244816** | 2,353,681 |
| Compressed layer 3 | 0.156252161 | **0.031195749** | 2,281,992 |

Layer 3's error falls **80.0%**, about fivefold. Its attention input is now
589,824/589,824 BF16 bitwise, instead of 49 differences. This resolves the
demonstrated matched control/frontend defect, not all compressed attention or
full-model drift. Remaining local frontiers include three low-rank query values,
25 rotary query values and output-B relative L2 0.033465559%; FFN-input
0.040668498% and FFN-output 0.419555492% include propagated upstream differences.
They must not be added or interpreted as isolated expert-kernel errors. Layer 0
changes only slightly; its remaining FFN-input has 53 BF16 differences,
FFN-output 0.018087820%, residual 48 differences. Sinkhorn combination and local
projection/rotary arithmetic remain open. These are **numerical corrections,
not speed or quality gains**; full-model 18–21% drift has not been remeasured.

Roots `/tmp/lunaflux-deepseek-production-layer-0-hierarchical-controls-rows1-20261012-v1`
and `/tmp/lunaflux-deepseek-production-layer-3-hierarchical-controls-rows1-20261012-v1`;
frontiers `/tmp/lunaflux-deepseek-layer-0-hierarchical-controls-frontiers-20261012-v1`
and `/tmp/lunaflux-deepseek-layer-3-hierarchical-controls-frontiers-20261012-v1`.
Original data/tolerances and earlier positive/negative probe results are retained.

## Sinkhorn: matrix-affine FMA and butterfly sums

The fresh original one-row observer
`/tmp/lunaflux-deepseek-original-hyper-layer3-rows1-20261012-v3` additionally
inspects the installed TileLang kernel's own emitted source, without changing
its operations or inputs. Original normalized/combination outputs remain
bitwise against `v2`. The captured `original-sinkhorn.cu` and installed
`reduce.h` show descending XOR-butterfly sums: for four entries the sum is
`(a+c)+(b+d)`, not `((a+b)+c)+d`. Matrix logits also use affine FMA. Strict
exponential and rounded division can remain unchanged.

Eight exact-control probes in `/tmp/lunaflux-deepseek-sinkhorn-20261012-v1`
separate affine, reduction and division. All use strict `expf` and unchanged
original mixes/weights/epsilon/20 iterations.

| Matrix arithmetic on identical original controls | Exact combination F32 / 2,304 | Relative L2 (%) |
| --- | ---: | ---: |
| Separate affine, ordered sum, rounded quotient | 501 | 0.000006108 |
| Separate affine, butterfly sum, rounded quotient | 1,349 | 0.000001913 |
| FMA affine, ordered sum, rounded quotient | 521 | 0.000006126 |
| **FMA affine, butterfly sum, rounded quotient** | **2,304** | **0** |
| FMA affine, butterfly sum, reciprocal then multiply | 434 | 0.000005493 |

All eight routes pass full-144-row memcheck/leak, racecheck and synccheck with
zero reports, 2-GiB/no-swap and balanced release. No intrinsic exponential,
tolerance change or literal result enters production. The implementation
adds the independent matrix-affine law and `SumReductionLaw` to the existing
immutable precision plan. Both initial softmax and iterative row/column
Sinkhorn lowering consume that sum law. It propagates through both
attention/FFN envelopes, while ordinary defaults and the earlier explicit
hierarchical replay policy remain unchanged. A distinct
`replay-layer-butterfly-controls` mode selects the new candidate. Generated
source and complete-layer propagation remain to be checked; this section is
an isolated arithmetic proof, not full-model parity or a throughput gain.

The actual generated envelope in
`/tmp/lunaflux-deepseek-hyper-arithmetic-20261012-v10` now matches **all 2,304
original F32 combination values bitwise**, alongside all mixes/inverses/post
controls and both 589,824-element BF16 collapse/weighted-normalization outputs.
The probe calls the **selected envelope's own Sinkhorn symbol**, not the former
default diagnostic Sinkhorn. All 23 actual-source routes pass full-144-row
memcheck/leak, racecheck and synccheck with zero reports, 2-GiB/no-swap and
released CUDA. Default emitted source remains byte-identical to the retained
original default source. Warning-denied native check passes; 125 affected
native tests pass, including both attention/FFN envelope propagation across
window/compressed layers. Original-source runner tests5/5 and sinkhorn/replay
runner tests3/3 pass. Complete-layer propagation is still pending, and matrix
parity on these original controls does not establish full-model parity.

### Complete butterfly-control propagation

Source-pinned ARM build **e9cf0537** passes in the CPU-only 2-GiB/no-swap unit;
the active MiniMax media GPU run is undisturbed. Export and execution both
record `matrix_affine=FusedMultiplyAdd` and `matrix_sum=ButterflyFloat32`.
Complete matched one-row layers on the original 144 predecessors/IDs exit
zero, release CUDA, and retain zero swap. No ordinary serving policy changes.

| Complete matched layer | Hierarchical relative L2 (%) | Butterfly relative L2 (%) | Exact BF16 / 2,359,296 |
| --- | ---: | ---: | ---: |
| Layer 0 | 0.015244816 | **0.006187106** | 2,358,217 |
| Compressed layer 3 | 0.031195749 | **0.027058402** | 2,316,157 |

Layer 0's FFN input is now **589,824/589,824 bitwise**, compared with 53
differences before this correction. Residual publication has one remaining
BF16 difference, down from 48. Its FFN output still differs on 1,800 values,
relative L2 **0.007635825%**, now on identical original FFN input. That is a
genuine local FFN frontier, not unexplained input propagation. Rotary queries
retain 139 differences and output-B retains 11; sparse attention, inverse
and output-A remain bitwise. Complete-layer numerical error falls 59.4%, not
execution time.

Layer 3 improves 13.3% in complete-layer numerical error. Attention input
remains bitwise; its three low-rank query values, 25 rotary values and
output-B error **0.033465559%** are unchanged. Residual relative L2 becomes
0.004460128%, FFN input 0.039682893%, and FFN output 0.372156653%.

A fresh installed-original layer-3 FFN replay on those **latest native literal
FFN inputs**, preserving actual IDs and one-row execution, distinguishes:

| Layer-3 FFN comparison | Relative L2 (%) | Exact BF16 / 589,824 |
| --- | ---: | ---: |
| Native vs original on identical latest native input | **0.015284652** | 588,405 |
| Original latest-native-input vs original original-input | **0.371842650** | 569,958 |
| Native latest-native-input vs original original-input | **0.372156653** | 568,539 |

Most propagated FFN drift is caused by changed input, while the smaller local
FFN differences remain visible. Norms do not add linearly. The offline adapter
now targets an arbitrary base layer for literal FFN replay, rather than
silently restricting this intervention to layer zero. It changes no original
model arithmetic. Roots are
`/tmp/lunaflux-deepseek-production-layer-{0,3}-butterfly-controls-rows1-20261012-v1`,
frontiers `/tmp/lunaflux-deepseek-layer-{0,3}-butterfly-controls-frontiers-20261012-v1`,
and original matched FFN
`/tmp/lunaflux-deepseek-original-layer-stream-butterfly-ffn-layer3-20261012-v1`.
Remaining local query/RoPE/projection and FFN arithmetic still require matched
isolation. Full-model 18–21% drift has not been remeasured or fixed; these are
not full-model correctness, throughput or media-quality results.

### Remaining query and local FFN isolation

The offline query observer now selects an explicit original layer and one-row
submission geometry. Layer-3 capture
`/tmp/lunaflux-deepseek-original-query-layer3-rows1-20261012-v4` calls installed
original quantization, FP8 matrix and normalization operations unchanged.
Its normalized-query output is **bitwise equal to the actual complete original
layer-3 observer**, confirming that this component replay has matched semantics.
All 144 normalized attention inputs are literal original values, not synthetic
or previously drifting native inputs.

| Query/KV component on identical original operands | Exact BF16 / values | Relative L2 (%) |
| --- | ---: | ---: |
| Query-A projection | 147,454 / 147,456 | 0.000035698 |
| Learned query RMSNorm | 147,455 / 147,456 | 0.000034048 |
| Query-B projection | 4,718,566 / 4,718,592 | 0.001786253 |
| BF16-expression head normalization | 4,718,592 / 4,718,592 | **0** |
| KV projection | 73,728 / 73,728 | **0** |
| Learned KV RMSNorm | 73,728 / 73,728 | **0** |

Head normalization and KV are not demonstrated local defects in this capture.
The learned query RMSNorm has a separate, reproducible reduction-order
frontier. Generated-source probe
`/tmp/lunaflux-deepseek-original-query-layer3-rows1-20261012-v5` tests six
immutable vector/tree folds on the **same original Query-A output**:

| Partitions × vector components | Exact normalized BF16 / 147,456 |
| --- | ---: |
| 128 × 4 | **147,456** |
| 256 × 4 | **147,456** |
| 512 × 4 | **147,456** |
| 256 × 2 | 147,455 |
| 512 × 1 | 147,455 |
| 512 × 2 | 147,455 |

All six alternatives plus the existing six-stage query prefix pass full-input
memcheck/leak, racecheck and synccheck in a 2-GiB/no-swap unit, with balanced
release. These are actual generated prefix-RMS sources consuming the existing
generic square-reduction law. They are **not yet selected in the complete
query prefix**; do not report that propagation as finished. Query-A/B matrix
rounding remains a separate cause. Failed observer attempts `v1`/`v2` are
preserved: the new row extraction initially sliced the batch rather than the
sequence axis. `v4` explicitly retains `[1,1,width]`; this was a diagnostic
bridge bug, not evidence of a production shape defect. Prepared `v3` was not
executed after identifying the axis mistake.

The worst layer-0 local FFN row, **54**, uses actual token **45948** and the
now-bitwise original/native FFN input. In
`/tmp/lunaflux-deepseek-original-expert-row54-20261012-v1`, all six routed
experts' weighted activations and outputs, using the **same original route
weights**, are bitwise in both matched-down
and propagated modes (12,288 activation and 24,576 output values per mode).
This excludes those routed expert kernels on these identical operands, as the earlier row-47
intervention did for row 47. Native router logits/weights, the shared FP8 expert
and final MoE combination are not covered by this intervention and require
isolation; other rows remain
unproven. No tolerance/reference edits or production selection changes follow
from these diagnostic results.

## Independent query-normalization law propagation

The existing `QuantizedQueryKvPrecision` now carries an independent immutable
`query_square_law`. Its default is unchanged (256 partitions, one component);
the KV fold and BF16-expression head normalization are not overloaded with
this option. The source generator consumes the law for learned query RMSNorm
only and publishes that kernel's actual launch threads. The existing device
frame already consumes source launch geometry, so no runtime rediscovery or
token-step branch is added.

Propagation is through the query owner → attention inputs → window/compressed
attention → enclosing attention envelope → decoder block/slice. Regression
tests cover both attention routes, unchanged weights/frame budgets, untouched
other five kernels, and unsupported-law rejection without silent fallback.
Ordinary generated query source remains byte-identical against retained `v5`.
The distinct `replay-layer-vector-query` diagnostic selects 256 partitions ×
four components on top of the previous butterfly-control candidate; the
earlier replay mode and production defaults stay unchanged. Both export and
execution report the selected query fold. A source-pinned complete-layer
replay is still required before claiming propagated numerical benefit.

Warning-denied native check and **106/106 affected native tests** pass,
including independent query geometry/source and window/compressed owner
regressions. Replay-runner tests2/2 pass. Targeted `moon info` completes with
936 existing migration warnings and zero errors; only affected public
interfaces are retained with this implementation. This does not replace the
unresolved clean full-suite/whole-model validation.

### Complete vector-query replay: local fix, no downstream gain

Clean source-pinned **58836d3b** ARM build succeeds. Complete layer-0 and
layer-3 one-row replays select `query_square=VectorizedTreeFloat32(256,4)` in
both export and execution, close CUDA and use no swap. Layer-3 low-rank query
mismatches fall **three → two**, removing the learned-query fold difference;
the remaining two originate in Query-A projection. Rotary, attention and
complete hidden output are **byte-identical to the preceding butterfly run**.
Layer 0 also remains byte-identical at complete output, including its already
exact learned query/FFN inputs. Complete-layer errors therefore remain
**0.006187106% / 0.027058402%**, not a further whole-chain improvement.
This correction reaches execution but is erased before changing downstream
quantized work; it does not establish a speed/quality gain or justify blanket
policy selection. Roots
`/tmp/lunaflux-deepseek-production-layer-{0,3}-vector-query-rows1-20261012-v1`;
frontiers `/tmp/lunaflux-deepseek-layer-{0,3}-vector-query-frontiers-20261012-v1`.

### Worst local FFN row: routing weights are a demonstrated cause

The fresh installed-original router observer now supports explicit **one-row
submission** rather than comparing a 144-row GEMM with production increments.
Root `/tmp/lunaflux-deepseek-original-router-layer0-rows1-20261012-v1` retains
all 144 literal layer-0 FFN inputs and original IDs. Layer-0 expert identities
are bitwise; identity-aligned route weights differ (relative L2
**0.000042622%**, maximum absolute error 0.0000005364418). Router logits and
scores differ too. Its layer-3 run uses this layer-0 component corpus, not the
actual layer-3 FFN input; no propagated layer-3 conclusion is inferred.

The original shared FP8 expert on actual row **54**, token **45948**, is
bitwise in both matched and propagated modes: 2,048 weighted activations and
4,096 outputs. Root `/tmp/lunaflux-deepseek-original-shared-row54-20261012-v1`.
Together with the preceding routed controls, no local expert discrepancy has
been demonstrated on this row with identical original routing weights.

In `/tmp/lunaflux-deepseek-route-weight-effect-row54-20261012-v1`, changing
**only the literal route weights to the observed native values** changes two
BF16 weighted activations in slot 0 and one in slot 1. Slot 0's down output
stays bitwise; slot 1's output changes **1,455/4,096 values**, relative L2
**0.215495331%**. All other routed outputs remain bitwise. This is a measured
quantization-sensitive amplification, not an assumed expert-kernel failure.

The actual ascending expert-identity/F32 combination, plus the bitwise shared
output, provides the decisive control:

| Row-54 combination comparison | Exact BF16 / 4,096 | Relative L2 (%) |
| --- | ---: | ---: |
| Original expert outputs/weights vs actual original FFN | **4,096** | **0** |
| Native-weight intervention vs actual native FFN | **4,096** | **0** |
| Native-weight intervention vs actual original FFN | 3,637 | **0.128003179** |

Root `/tmp/lunaflux-deepseek-route-weight-combination-row54-20261012-v1`.
Thus the router-weight change reproduces the **entire actual native FFN row
bitwise**; final combination is not an unexplained difference for that row.
This does not establish every other row, router arithmetic parity, full-model
parity or media quality. Next isolate router projection order, softplus/sqrt
evaluation and selected-weight sum on **identical observed operands** before
changing production arithmetic. No original data, tolerance or serving default
is altered by these offline interventions.

The repeated weight-only intervention `row54-v2` passes memcheck/full leak,
racecheck and synccheck with zero reports in serialized 2-GiB/no-swap user
units. It uses the same observed weights and preserves the first result.
Router/weight-intervention/combination runner regressions **5/5** pass.
The application releases CUDA resources; no new router arithmetic is selected
in production from this causal experiment.

### Router arithmetic isolated and implemented without vendor dependencies

`/tmp/lunaflux-deepseek-router-arithmetic-20261012-v2` isolates each boundary on
the preceding literal layer-0 corpus, retaining all 144 rows and expert slots.
Strict direct `log1pf(expf(x))` (threshold 20) plus `sqrtf` matches **36,864 /
36,864** original scores on original logits. The stabilized expression matches
35,574; replacing log1p with log matches only 20,271. No intrinsic/approximate
exponential is selected. On original scores, neutral-padded descending
butterfly sums and rounded division match **864 / 864** selected weights;
serial sums match 549, and reciprocal multiplication is not equivalent.

Both offline pedantic vendor projection controls match all 36,864 original
logits. A scoped administrator Nsight capture of the matching GEMV has
grid 32, block `(32,8,1)`, one FMA chain per 32-way striped partition and
descending-half shared additions. The native reproduction uses the existing
`GroupedVectorFloat32(1,32,1,1,FusedMultiplyAdd)` law, not a vendor call or
literal output substitution. Root `v3` reproduces **all logits, scores and
864 weights bitwise** through the independent native chain.

The actual compiler-generated sources are also exercised: `v3/source-layer-
{0,3}-output` has exact logits 36,864, scores 36,864, selected identities 864
and weights 864 for both hash and learned selection. Layer 3 here is still
the **layer-0 component corpus**, not the actual latest layer-3 FFN input.
Native arithmetic and both actual generated routes pass full-144-row
memcheck/leak/racecheck/synccheck, serialized in 2-GiB/no-swap units, with
balanced CUDA release. The earlier vendor-inclusive `v2` racecheck exhausted
the 2-GiB bound; its failure is retained, and vendor controls are excluded
from the subsequent native sanitizer run. `v1`'s offline C++ compile failure
and the first no-kernel Nsight name filter are also retained.

Immutable routing precision now carries independent projection, softplus and
selected-sum laws; neutral padding is explicit and does not add logical
expert slots. Unsupported dot lowering rejects rather than falling back.
Propagation reaches model routing plans, the FFN envelope, decoder block and
slice, and the distinct `replay-layer-router-arithmetic` mode. Export and
execution log all three policies. Ordinary defaults remain unchanged and
their generated router source was independently byte-compared with the
retained original corpus export. This is numerical correctness work, not a
speed result. Complete-layer improvement and whole-model parity still require
the next source-pinned replay; the preceding row-54 causal result alone does
not prove every FFN row or complete-model correctness.

Warning-denied native check succeeds with the existing warning exclusions;
**99/99 affected tests** and three `.mbtx` runner tests pass. Targeted
`moon info` completes with 936 pre-existing migration warnings and no errors.
Only task-owned interfaces are retained. No payload hashes were introduced
or computed; these numerical probes compare actual arrays directly.

### Source-pinned router replay and complete propagated chain

Implementation **f9d6dc2c** is pushed. Its clean ARM build completes under the
2-GiB/no-swap CPU-only boundary (745.3 MiB reported peak). Source export and
execution both select the independent router laws; ordinary serving is still
unchanged. Complete one-row, 144-predecessor replays close CUDA without swap:

| Boundary | Before router correction, relative L2 (%) | Corrected (%) | Exact BF16 |
| --- | ---: | ---: | ---: |
| Layer-0 FFN | 0.007635825 | **0.000746736** | 589,810 / 589,824 |
| Layer-0 complete hidden | 0.006187106 | **0.000808384** | 2,359,272 / 2,359,296 |
| Layer-3 complete hidden | 0.027058402 | 0.027008126 | 2,317,277 / 2,359,296 |

The old worst row **54** now has bitwise FFN output. Layer 3's input-driven
frontier remains: FFN input 0.039682893%, FFN output 0.371844611%. Router fixes
do not remove preceding attention/input differences.

The new native layer-major replay executes **all 43 base layers** on `.179`:
`/tmp/lunaflux-deepseek-native-layer-chain-router-20261012-v1`.
It starts at original hidden-0 and feeds each subsequent layer the **previous
native output**, retaining all 144 rows and the original token IDs. Only one
bank is resident; runs are serialized in 12-GiB/no-swap user units. CUDA owners
retire between layers. The independently computed original position-143 mean
matches the actual original observer bytes before every comparison.

| Retired layer | Position-143 stream-mean relative L2 (%) |
| --- | ---: |
| 0 | **0, all 4,096 values bitwise** |
| 1 | 0.432739059 |
| 2 | 1.546022872 |
| 3 | 2.204441841 |
| 9 | 11.278133835 |
| 40 | 17.704235998 |
| 42 | **15.878691075** |

This is not a quality pass. Final row-143 **complete streams** differ by
16.527097826%; final all-144-stream aggregate L2 is only 0.307705020%, heavily
dominated by very large original entries (maximum 69,632). The aggregate cannot
hide the late-token discrepancy. The older 18–21% measurements also used
different topology/physical-row geometry; this fresh curve does not justify a
pure improvement claim against those runs. No model head, generated tokens,
TP2 serving or throughput is validated by this hidden-chain diagnostic.

### First propagated divergence: a shared-FP8 output, not routed experts

Layer 1 replayed on **original hidden-1** has bitwise rows 6 and 143, while its
all-row complete error is 0.025672617%. On previous-native hidden-1, row 6 is
1.276885949% different. The predecessor row differs at exactly **one of 16,384
BF16 stream values**, channel 12,919, by 0.00390625. Root
`/tmp/lunaflux-deepseek-layer0-stream-row6-difference-20261012-v1.txt`.
Fresh original layer-1 boundaries use
`/tmp/lunaflux-deepseek-original-layer-stream-layer1-frontiers-20261012-v1`;
its hidden-1/2 are bitwise the retained full reference. The original run emits
an installed ThreadSync warning; it is not described as empty stderr.

On row **6**, actual token **45948**, all six routed experts' activations and
outputs are bitwise on original weights in both matched and propagated modes:
`/tmp/lunaflux-deepseek-original-expert-row6-20261012-v1`.
The shared FP8 expert in `...original-shared-row6-20261012-v1` differs at
activation 1,555 and down-output **631**. Crucially, down-output 631 still
differs on the **identical original activation**: 4,095 / 4,096 exact, relative
L2 0.012854112%. Its native and original values are -0.439453125 / -0.4375.
Thus the down difference cannot be dismissed as propagation of the single
activation mismatch; both frontiers must be retained.

The shared-only offline combination intervention keeps all original routed
outputs and the actual ascending expert order. Original shared output gives
the actual original FFN **4,096 / 4,096 bitwise**; substituting only the observed
native shared output gives the actual native FFN **4,096 / 4,096 bitwise**.
Root `/tmp/lunaflux-deepseek-shared-combination-row6-20261012-v1`.
This proves the entire remaining FFN discrepancy on that first-divergence row
comes from the shared expert, not router selection, routed experts or combine.

Source review identifies a concrete **arithmetic-domain difference to test**:
installed original FP8 GEMM resets an unscaled F32 dot accumulator every
128-wide block, applies the block scale, then updates a separate scaled F32
accumulator. Native packed-expert lowering reconstructs scaled BF16 operands
before a whole-width striped dot. Those orders are not equivalent. Next probe
unscaled-block reductions, scale application and publication on the captured
original activation, and independently compare gate/up before SiLU. Do not
blindly change scale/reduction policies, transplant HC dot laws, relax tolerance
or promote the candidate because a global norm looks small. A general scaled-
dot arithmetic contract belongs in precision IR and backend lowering, not a
row/token/model-specific exception. All `.179` probes have retired; MiniMax
`.178` remains in denoising with zero swap and empty application stderr.

### Shared FP8: the missing boundary is raw matrix accumulation

The new bounded operand probe falsifies the simple "move scaling after the
dot" explanation for the first divergent row. In
`/tmp/lunaflux-deepseek-shared-block-row6-20261012-v2`, all 32 scalar alternatives
(1–128 independent partial sums, separate/FMA inner updates and separate/FMA
scaled publication) fix the one up-projection discrepancy, but **none fixes
down-output 631**. Those negative results remain available.

The installed original quantizer is independently observed in
`/tmp/lunaflux-deepseek-original-shared-row6-observed-20261012-v2` without
modifying or feeding observations into `Expert`. Input payload/scales are
**4,096/4,096 and 32/32 bitwise**; matched activation payload/scales are
**2,048/2,048 and 16/16 bitwise**. Stateless original down on this activation
is bitwise the complete original shared output. The original F32 terminal
also rounds to all 4,096 original BF16 outputs.

Output 631 is exactly rounding-sensitive: original F32 is **-0.4384765625**,
the BF16 midpoint; native is **-0.438476592302**, one F32 ULP lower. A full-dot
F64 diagnostic still rounds to the native BF16 value, so merely "increase
precision" does not reproduce the reference arithmetic. No midpoint snapping,
tolerance waiver, input replacement or special-case output correction is used.

Original unscaled 128-column blocks are captured independently at
`/tmp/lunaflux-deepseek-original-shared-row6-blocks-20261012-v3`. Scalar folds
match only about 61,500/65,536 F32 block values. A native raw-E4M3 matrix-block
probe at `...shared-block-row6-20261012-v5` matches **65,536/65,536 F32 blocks**
and **4,096/4,096 BF16 down outputs**, plus both 2,048-value gate/up outputs.
Its register ownership follows the documented
[NVIDIA PTX matrix-fragment ABI](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832).
The original's two accumulators are two **F32** levels, not F64. The early
compile failure in block-probe v1 is retained and was repaired using the public
BF16 raw conversion, not by changing model arithmetic.

The implementation expresses `ScaledDotReductionLaw` in immutable precision
IR: decoded-operand F32 remains default; an explicit unscaled matrix block
and scaled-publication law are selectable. This independent policy reaches
the shared expert, FFN owner, decoder block/slice and diagnostic CLI
`replay-layer-shared-block-matrix`. Routed FP4 remains unchanged. Unsupported
formats/grouping reject instead of silently falling back. CUDA warp/MMA
details remain private lowering; no vendor runtime dependency is introduced.

Actual generated-source replay
`/tmp/lunaflux-deepseek-shared-source-row6-20261012-v1` now matches **all 2,048
activations and 4,096 outputs bitwise** in both matched-down and propagated
modes. Actual source memcheck/full leak, racecheck and synccheck report zero,
in serialized 2-GiB/no-swap user units with balanced CUDA release. Launch
descriptors, not old replay constants, select the matrix work decomposition.
Affected native tests **99/99** pass. Ordinary serving defaults are unchanged.
This is a demonstrated local numerical correction, **not throughput, a
43-layer parity result or media quality**; the next source-pinned complete
layer must establish whether the first propagated residual difference is
removed. MiniMax remains independently active on `.178` without concurrent
GPU work. All comparisons use array values directly, with no payload hashes.

### Committed shared-matrix complete replay and next projection frontier

Implementation **16fb7184** is pushed. Its clean ARM build on `.178` is
CPU-only, warning-denied with unchanged migration exclusions, completes at
746.3 MiB reported memory and zero swap. The full layer-0 replay on all 144
original predecessors/IDs proves the selected law in **both export and
execution**:
`/tmp/lunaflux-deepseek-production-layer-0-shared-matrix-rows1-20261012-v1`.
All **589,824 FFN outputs are now bitwise**. Complete hidden error is
**0.000199292%**, down from 0.000808384%, with 2,359,280/2,359,296 exact.
Row 6 is bitwise across all 16,384 hidden values, removing the demonstrated
first shared-output divergence. The remaining 16 hidden differences follow
attention/output boundaries; FFN input and output are bitwise.

Layer 1 using that **corrected native predecessor**, not an original-input
substitution, has bitwise row 6 and row 143. Root
`/tmp/lunaflux-deepseek-production-layer-1-shared-matrix-propagated-rows1-20261012-v1`;
all-row hidden error 0.027826625%. Layer 3 on original predecessors remains
0.027008023%, with preceding attention/FFN-input drift unchanged. The small
layer-0 result is not extrapolated into compressed attention or all layers.

The new full native propagation run completes all 43 layers, one bank at a
time, serialized under 12-GiB/no-swap boundaries:
`/tmp/lunaflux-deepseek-native-layer-chain-shared-matrix-20261012-v1`.
Every derived original position-143 mean is again bitwise the actual observer
before comparison. The earlier router-only run is retained unchanged.

| Retired layer | Router-only mean error (%) | Shared-matrix mean error (%) |
| --- | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 0.432739059 | **0** |
| 2 | 1.546022872 | 0.731341504 |
| 3 | 2.204441841 | 1.805925927 |
| 9 | 11.278133835 | 10.693824442 |
| 21 | 17.713524434 | 14.480403024 |
| 40 | 17.704235998 | 14.473616808 |
| 42 | 15.878691075 | **13.196633236** |

Final row-143 complete-stream error is **13.656348080%**, versus 16.527097826%.
All-row aggregate is 0.301482656%; it is still dominated by large reference
entries and is not a quality/parity claim. No model head, generated tokens or
TP2 serving correctness follows. These are numerical-error reductions, **not
performance improvements**.

The next identical-operand projection test reuses the actual compiler's raw
matrix-block helper, without changing learned/head normalization:
`/tmp/lunaflux-deepseek-query-block-layer3-20261012-v1`. It matches **Query-A
147,456/147,456, Query-B 4,718,592/4,718,592 and KV 73,728/73,728 BF16 values
bitwise** on the retained layer-3 original one-row corpus. The old scalar
Query-A/B differences are therefore an executable arithmetic gap, not an
unexplained norm/layout/cache issue. This probe is **not yet a selected query
owner/complete-layer implementation**. Next propagate an independent scaled-
dot policy through the generic FP8 projection family, query owner and output-B
chain, consume actual launch metadata, then test complete execution. Do not
claim propagation from this helper-level result or blanket-promote all laws.

### Query projection law reaches generated prefix execution

The independent `QuantizedQueryKvPrecision.projection_scaled_dot_law` now
reaches the generic FP8 source, its launch descriptors, query frame/owner,
window/compressed attention, envelope, decoder block/slice and explicit
`replay-layer-query-block-matrix` diagnostic. Learned-query reduction remains
independent; KV/head normalization and ordinary serving defaults are unchanged.
The private raw matrix primitive is shared with the proven shared-expert
lowering. A fixed 128-element quantization scratch preserves all supported
input widths instead of adding a full-width shared-memory limit. Unsupported
block/publication alternatives reject, never silently substitute.

Actual source root `/tmp/lunaflux-deepseek-query-source-layer3-20261012-v4`
matches all six captured stages on **144 rows**, both with matched original
operands and a fully propagated native prefix: Query-A/query norm147456,
Query-B/head norm4718592, KV/KV norm73728 BF16 values are bitwise. Full144
memcheck/leak reports zero. Racecheck/synccheck on the **first captured row**
report zero, with balanced release in serialized2GiB/no-swap user units. These
are actual generated-kernel results, not an unselected helper prototype.

The negative runs are retained: v1 full144 racecheck hit2GiB OOM; v2/v3
one-row racecheck found a real WAR hazard. Line-correlated v3 establishes that
CUDA aliases dead maxima storage with the encoded scale while other threads
still read the maxima. The missing reader-completion barrier now closes that
scratch lifetime before publication. v4 retains bitwise results after repair;
no suppression, tolerance edit, approximate exponential or reference change.

Affected native tests124/124 and warning-denied check pass with existing
migration exclusions. Targeted `moon info` reports936 migration warnings,
zero errors. Ordinary query source is independently byte-compared against its
retained pre-change export and is unchanged. Complete-layer execution remains
the next propagation test; this local prefix correction is not a throughput,
43-layer parity, generated-token or media-quality claim. No payload hashes.

The shared helper refactor is separately rerun in actual generated source at
`/tmp/lunaflux-deepseek-shared-source-row6-20261012-v2`:2048 activations and4096
outputs remain bitwise in matched/propagated modes; memcheck/full leak,
racecheck and synccheck remain zero. This validates the common primitive without
silently assuming the earlier shared-expert result survives a source refactor.

### Commit-pinned query replay and independent output-B proof

Implementation23dea2e7 is pushed. Its clean ARM build completes on178 under
2GiB/no-swap CPU-only limits (750.3MiB reported peak). Complete original-input
one-row replays cover all144 predecessors/IDs at
`/tmp/lunaflux-deepseek-production-layer-{0,3}-query-matrix-rows1-20261012-v1`.
Export AND execution explicitly select the query scaled-dot law.

Layer0 query norm/RoPE, KV, sparse attention, inverse rotary and output-A are
now **all bitwise**, including removal of139 qRoPE differences. Output-B still
has11 BF16 differences, hidden still16; complete error0.000199292% is unchanged.
Layer3 lowrank is147456/147456 exact; qRoPE has just1/4718592 differing value,
while sparse attention, inverse and output-A are all bitwise. Output-B retains8
differences. Complete hidden error decreases from0.027008023% to**0.001508918%**;
this is numerical error reduction, not throughput. Frontier report roots are
`/tmp/lunaflux-deepseek-layer-{0,3}-query-matrix-frontiers-20261012-v1`.

The next leaf is tested independently, not inferred from the query result:
`/tmp/lunaflux-deepseek-output-b-source-20261012-v1` replays actual generic
8192→4096 FP8 matrix source on all144 original output-A operands from the
original one-row output observer
`/tmp/lunaflux-deepseek-original-output-native-all144-20261012-v1`.
All589824 output-B BF16 values are bitwise; full144 memcheck/leak and first-row
race/sync checks report zero. No output-A/reference substitution enters serving.
The same proof survives decoder-helper factoring at source-rootv2. The new
independent output scaled-dot policy and `replay-layer-fp8-projections` profile
must next establish complete-layer propagation; defaults remain unchanged.

The output-B policy now reaches immutable grouped-output precision, actual
launch metadata, output owner, window/compressed sublayers, envelope and
decoder block/slice independently of query and output-A policies. Generic
numeric helpers are shared without changing scalar-default bytes. Nativecheck
and131 affected tests pass; targetedinfo936 existing migration warnings,0errors.
Query-default source and output-B-default leaf are byte-compared unchanged.
An initial whole-historical-output-bundle byte check failed only on older
inverse-rotary casts/formatting; it is not a numerical mismatch or a reference
rewrite. The corrected regression compares the actual unchanged leaf, leaving
all original arrays intact. New source-rootv2 again has full144 bitwise output-B,
full144 memory/leak checks and first-row race/sync checks zero. Complete-layer
FP8-projections replay is still pending; no new full-chain/performance claim.

### Complete combined projection replay — bfec7a6a

Clean commit-pinned ARM build
`/tmp/lunaflux-checkpoint-build-fp8-projections-20261012-v1` completes CPU-only
on178 at752.2MiB reported peak and zero swap. Original-input, physical-row1
replays execute all144 literal predecessor/ID rows with actual export AND
execution selection at
`/tmp/lunaflux-deepseek-production-layer-{0,3}-fp8-projections-rows1-20261012-v1`.

Layer0 is now **2359296/2359296 hidden values bitwise**, and every captured
attention/control/FFN frontier is bitwise. This removes the remaining16 hidden
differences; it is a complete local layer result on the original corpus, not
whole-model generation correctness.

Layer3 is2359293/2359296 hidden exact, relativeL2**0.000056015%**, versus
query-only0.001508918% and shared-only0.027008023%. Sparse attention, inverse,
output-A/B, residual and FFN input are all bitwise. One qRoPE value differs
without affecting the captured attention output; two FFN-output values remain
different on an **identical original FFN input**, producing three hidden
differences. These remain visible local frontiers, not waived tolerances.
Reports `/tmp/lunaflux-deepseek-layer-{0,3}-fp8-projections-frontiers-20261012-v1`.

The chain runner now has an explicit `--fp8-projections` option, with prior
router/shared modes unchanged and2/2 runner regressions passing. A fresh native
propagation chain, starting only from originalhidden0 and thereafter using
previous native hidden, is the next test. No updated43-layer result, model
head/generated-token, TP2 serving or performance claim is made yet. MiniMax
remains active on178, empty application stderr and zero swap; no GPU overlap.

### Fresh native chain: first remaining divergence is ratio-4 layer2

The fresh43-layer controller reserves179 until retirement at
`/tmp/lunaflux-deepseek-native-layer-chain-fp8-projections-20261012-v1`.
Already-retired layers0 AND1 match all2359296 hidden values bitwise. The
independent `layer2-input-original-proof.txt` compares the actual previous native
hidden against originalhidden2 and is also wholly bitwise. Therefore the first
remaining divergence is local to layer2 on this corpus, not earlier input drift.
Model config identifies layer2 as the first ratio4 compressed/indexed attention
layer; ratio128 layer3 is a different case. No operator cause is yet assigned.

All eleven native layer2 fronts are downloaded without overwrite from the
retired child using `collect-boundaries`; existing hidden/comparison files are
not rewritten. The unmodified original observer for3 layers, boundary2, is
CPU-prepared at
`/tmp/lunaflux-deepseek-original-layer-stream-ratio4-frontiers-20261012-v1`.
Its GPU execution waits for the chain to release179. Future original observer
units/containers use12GiB/no swap, aligning with the requested complete-layer
limit; older16GiB results and arithmetic remain untouched. Collector tests3/3,
original observer runner tests2/2 pass. No new final43-layer outcome or operator
fix follows from this preparation alone.

Default shared source is independently byte-compared with the pre-change
retained export; it remains unchanged. Native affected tests99/99 and offline
source/chain/query runner regressions pass. Actual shared source sanitizers are
the preceding one-row proof; the new full-chain runs are deterministic numeric
replays, not full-chain sanitizer campaigns. `.179` owners have retired after
the query probe. MiniMax `.178` is still denoising, zero swap and empty
application stderr; no media terminal or quality result is available yet.

### Completed FP8 chain and revised interpretation

The FP8-projection chain above is now terminal (`COMPLETE.txt` exists), not
reserving179. All43 children retired. Its final position143 mean relativeL2 is
14.134457856729%, versus retained shared-matrix13.196633236%; full row143 streams
are14.551407124895%, versus13.656348080%. Several exact local repairs therefore
did **not** produce a final-hidden improvement. The all144 aggregate0.299083838%
is dominated by large reference values and must not be presented as model quality.
No final head, generated-token, task-quality or speed improvement is established.

The independent original ratio4 observer subsequently completed; hidden0..3
retain the full-reference values. Layer2 begins with exact original inputs and
first diverges in the captured KV-simulation boundary. Native query/RoPE are
bitwise. Complete frontiers are retained at
`/tmp/lunaflux-deepseek-layer-2-fp8-projections-frontiers-20261012-v1`.
The later matched query probe
`/tmp/lunaflux-deepseek-original-query-layer2-rows1-20261012-v1`
shows exact QueryA/query norm/QueryB/head norm/KV projection, but2/73728 learned
KV-norm differences on **identical** original projected inputs. At row30/column40
native is-0.91015625 versus original-0.90625; the later quantized KV value is
-0.9375 versus-0.875. This proves one local arithmetic discrepancy, not that it
alone explains compressed attention or final-model drift. No new KV-norm policy
has been selected or physically proved by the prepared alternative probe.

The [revised correctness plan](NUMERICAL_CORRECTNESS_RETHINK_2026-10-12.md)
now prioritizes prediction distributions, within-framework controls and held-out
quality alongside causal operator tests. Exact local compatibility remains
useful evidence, not the universal optimization objective. The new CPU
teacher-forced comparator is infrastructure, not a new four-framework model run.
