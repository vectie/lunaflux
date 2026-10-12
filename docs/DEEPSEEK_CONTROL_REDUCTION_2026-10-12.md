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
