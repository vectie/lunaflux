# Functional compiler integration work

Scope: complete the seven optimization/integration directions discussed after
`a1adf4cc`, without removing supported shapes or moving CUDA policy into model
or scheduler code. This is a work ledger, not a performance completion claim.

The authoritative path is pure semantics → immutable strategy/layout choice →
explicit effect/lifetime plan → backend lowering. Offline physical observations
are inputs to selection; they are never collected inside a token step.

| Work | Completion criterion | Current work |
| --- | --- | --- |
| Projection pipeline search | Generate legal stage/window alternatives, measure and select them; backend implements the selected lifetime plan | Resource filtering and source-bound offline selection integrated; 19 MLP and 40 QKV/output/head combinations physically measured; real-record export retains baseline; no faster long-input alternative found in this search |
| Operand segmentation | Shared semantic address/segment plan consumed by projection lowerers; preserve useful schedules rather than force losing hoists | Partial: concatenated weights are now immutable request/program semantics, identity-bound and preserved by normalization; QKV scalar, single-token, matrix-map and pipeline lowerings consume that view. Transfer intervals use the same row planner and an exact caller-controlled expansion budget. Full QKV transfer-worker schedule integration remains open; previous hoist regressed |
| Attention ownership/history | Preserve existing c322 semantics and correctness through new selection | Existing path; regression required |
| Attention shape buckets | Query/history/batch-specific measurements choose admitted artifacts through bounded runtime dispatch | V6 exporter/runtime binding and bootstrap classifier integrated; real baseline record passed 48 physical serving trials; no measured c324 production win is claimed |
| Resource feedback | Measured budgets/register facts reach resource-aware compiler selection, including expanded schedules | Prefill/decode/partitioned exporter connected; decode resource-only choice regresses, measured override physically restores baseline source |
| Fusion chain selection | Compare full versus partial chain cost, select through a generic immutable policy, validate both | Paired measured selector integrated; full-chain serving is slower and has last-token divergence requiring diagnosis; partial retained |
| Execution diagnostics | Latest runtime trace includes actual work, padded work, selected bucket/route, mixed steps and GPU gaps | Latest worker trace correlated with all CUDA graph launches; capacity slack, final owners and route cost measured; maximum launch-envelope issue found and correction under GPU test |

No item is complete merely because an interface exists. Completion includes
public behavior tests, integration tests, exact-source physical correctness,
relevant sanitizer checks and matched end-to-end timing. A slower experiment
may establish a cause but does not become the production choice by default.

Latest clean-source closure campaign:
[current compiler validation](BENCHMARK_CURRENT_COMPILER_CLOSURE_2026-09-15.md).
The `d50f3b90` runtime completed the four-vector uninstrumented matrix and seven
mixed workloads. Current frontier measurements again select c322, whose
selected source is unchanged. 4096/64/C16 is 230.06 output tok/s, essentially
unchanged from the prior 230.63. The 3072/32 last-token variability persists;
fresh serving completion is not completion of numerical acceptance or all
common strategy/effect integration.

## Query-owned attention effects, 2026-09-15

The query-owned attention lowering now consumes an immutable, backend-neutral
`AttentionQueryEffectPlan` for transfer issue, readiness, publication, slot
reuse and early-exit draining. CUDA lowering renders these actions without
deciding their ordering. The three existing legal lifetime modes are preserved:
cooperative single-slot, split K/V single-slot, and joined double-slot.

A differential test against the previous renderer passed all 48 combinations
of twelve candidates, dense/history mode and partitioned mode, with identical
source bytes. The old renderer was then removed and source-digest snapshots
retained. A finite asynchronous memory model covers zero through eight tiles,
every early-exit position, missing readiness and premature reuse. Native
warning-denied check passed; schedule tests 22/22 and source tests 33/33 passed.
This is a behavior-preserving compiler refactor, not a new GPU speedup or a
claim that all attention families now use the common effect plan.

The unused `AttentionFoldFences`/`plan_attention_fold_fences` API was removed
after semantic reference lookup found only its own test, not a lowering
consumer. Its replacement regression checks the actual consumed effect plan:
value readiness/publication precede next-key issue, value readers release
before next-value issue, and next-key readiness remains an explicit wait.
This avoids maintaining a second, disconnected synchronization policy.

## Grouped decode effects, 2026-09-15

Grouped decode now also consumes `AttentionGroupedEffectPlan`. The common
plan orders validity initialization, separate key/value readiness, uniform
error exit, alternate-slot prefetch, reader release and partial-fold publication.
It preserves cooperative single-slot and asynchronous two-slot schedules;
CUDA supplies transfer addresses and instruction spellings only. The redundant
CUDA-side lifetime eligibility rule was removed in favor of the common constructor.

Eighteen pre-refactor source snapshots (three candidates, three batch sizes,
two history lengths, including partitioned output) remain byte-identical.
The finite asynchronous model covers zero through nine tiles and every invalid
tile exit; negative tests reject omitted waits, publication and reader release.
Schedule tests 25/25, source tests 34/34 and warning-denied native check pass.
These unchanged generated kernels do not establish a new speedup. Shared-score
prefill and remaining operand-schedule integration are still open.

## Scalar selected-row scatter correction, 2026-09-15

The scalar projection lowering gathered `row_offsets[logical_row + 1] - 1`
but stored into packed `output[index]`. The semantic selected-row result and
the matrix lowerings retain original token-row positions. The scalar path now
scatters to `row * output_width + column`; all-token scalar source is unchanged.
This is a general non-matrix fallback correction, not the cause established
for the current Tensor Core runtime's last-token variability.

The actual exported CUDA kernel passes five cases on RTX 5060 Ti: noncontiguous
ends, contiguous decode, one long row, and an invalid selected row. The probe
checks every output value, including untouched sentinel rows. A negative
control restoring only the previous packed store fails at trial 0 / row 0 /
column 0. Memcheck (including leak check), racecheck and synccheck pass with
zero errors; no speedup is claimed. Native warning-denied check and projection
tests 75/75 pass. Reproduction uses `export_scalar_scatter.mbtx` and
`scalar_scatter_probe.cu`; remote results are in
`/tmp/lunaflux-scalar-scatter.6QkWjs`.

Clean Linux archive `6668c725` subsequently passed all 3,083 native tests with
`--deny-warn` in `/tmp/lunaflux-clean-6668c725.3XxR95`. The dependency C compiler
still reports an implicit declaration warning for
`posix_spawn_file_actions_addchdir_np` in async's thread pool; this is not a
warning-free C build claim. The later unused-policy removal passed local
warning-denied check and the same 22 schedule / 33 source tests.

Latest numerical follow-up: [actual QKV activation capture](BENCHMARK_ACTUAL_QKV_ACTIVATIONS_2026-09-15.md)
finds 29 differing output pairs among 77 exactly equal actual-model input pairs,
all crossing single-token/multi-token execution. A subsequent FP64 dot-product
reference finds the single-token result closer in all 82 changed-component
observations. This localizes and measures a numerical boundary; final-token
causality remains open; the fresh current-source performance campaign above
has completed and still reproduces last-token variability.

Latest follow-up: [actual-row replay and V6 validation](BENCHMARK_ATTENTION_ROW_REPLAY_2026-09-14.md).
The corrected 34-shape replay reduces the isolated c324 advantage to 1.112%,
matching the prior serving attention trace's 1.03%. Counters show 29–31% more
instructions at nearly equal L2 bytes for large shapes. This is a measured
KV-tile tradeoff, not a missing generic async pass. Model-token divergence
diagnosis now excludes a greedy-selector mismatch across 30,720 outputs:
[logit-margin diagnosis](BENCHMARK_LOGIT_MARGINS_2026-09-14.md). First divergent
ranks are tied or separated by 0.125. Upstream numerical acceptance remains
open; no candidate promotion or relaxed token contract follows from that fact.

## Prefill resource integration, 2026-09-14

The exporter accepts `--attention-resources PATH SHA256 DEVICE` alongside
`--attention-tuning`. Budget/register data are immutable offline inputs. The
budget expands the legal frontier before register observations and latency
records are matched; actual timing retains priority over estimated residency.
No profiling or file access was added to token execution.

On the RTX 5060 Ti with the pinned CUDA 13.1.115 compiler, actual cubin register
counts were 181/228/210/200/216 for candidates 319/322/324/318/323 respectively.
Resource-only selection chose 318. Combining these observations with the
existing measured latency table selected 322, with exactly the previously
benchmarked kernel source digest
`d2d6598adff474dae4997538c7384a1b50d5d8d5167d878047709b220f46bbed`.
This is an integration check, not a new performance measurement.

Validation: warning-denied native check; exporter 11/11, tuning parser 1/1,
attention compiler 15/15, CUDA attention AOT 5/5. Exporter preparation and
publication were split into focused files (largest 417 lines).
The full local warning-denied native suite also passed: 3,742/3,742.

Remote run: `/tmp/lfresources.mANp0Z`. Downloaded archive:
`/tmp/lunaflux-resource-feedback-20260914.tar.gz`, SHA-256
`0929eb86326a660ba85ce3ab1c4632426792c6abb90ab383a63a5284720ed922`.
It contains the validation driver, device query, actual resource output,
candidate exports and results. The earlier baseline latency observations are
reused explicitly, not represented as fresh measurements.

## Projection lifetime experiments, 2026-09-14

The compiler now accepts immutable fold choices (matrix, sibling and
intermediate roles), stage count, transfer window and fragment lookahead.
Composition is canonical; tests verify the semantic program digest is unchanged.
The four projection lowerers consume one finite ring renderer. The established
two-slot scheduling seam remains the default. A finite slot model checks that
publication precedes consumption and no live slot is overwritten.

Targeted local validation: projection compiler 53/53; projection CUDA AOT 68/68;
warning-denied source-probe compile. This is not a full-suite or GPU sanitizer
completion claim.

GPU exploration is at `/tmp/lffold.Tnkvuy`. Wider 64-element, 3/4-stage gate
variants exceed the static shared-memory ceiling and fail compilation. The
32-element variants compile but require opt-in shared-memory launch capacity;
the unchanged launch fails with CUDA error 1. An explicitly opt-in diagnostic
launch passes bitwise gate-workspace comparison and independent sampled scalar
checks at 32/1024/2048 tokens. At 1024 tokens the baseline is approximately
349 us versus 479/482 us for 3/4 stages; at 2048 tokens approximately 680 us
versus 944/954 us. These are gate-only exploratory measurements, not down,
whole-MLP or end-to-end throughput. The diagnostic sets the dynamic shared-memory
attribute on each launch, so it is not the final performance harness.

No experimental variant is selected for production. Resource-aware eligibility,
full family coverage, sanitizer checks, per-shape measured records and exporter
selection are still required. More stages alone do not demonstrate a speedup.

## Legal MLP search and decode feedback follow-up

The legal fold search now accounts for static ring storage, epilogue aliasing,
and each launch's own dynamic shared memory. Nineteen legal baseline/alternative
modules passed 32/1024/2048-token numerical checks and memcheck, racecheck and
synccheck in `/tmp/lffoldlegal.uaX4DB`. Five interleaved trials measure gate,
down and the complete MLP chain. At 2048 tokens the baseline chain is 1030.54 us;
no alternative improves it. Stage count alone is not a missing speedup.

`--projection-folds PATH SHA256 DEVICE` consumes immutable source-bound fold
measurements offline. Unmeasured shapes retain their existing strategy. This
does not yet constitute complete QKV/output/vocabulary physical search coverage.

Decode and partitioned resource observations now reach their exporters. Real
decode register counts are 56/31/71 across the frontier. Resource-only selection
changed candidate 441 to 400, but physical comparison showed a severe regression:
the aggregate median over 25 context/batch cases was 2946123 ns for 441 versus
61889381 ns for 400 (five trials, separate kernels, not serving throughput).
Both passed the independent numerical oracle and preserved KV bytes. The
exporter now also accepts `--decode-tuning PATH SHA256 DEVICE`, so actual latency
can override the resource estimate. Candidate 400 is not promoted.
The real exporter with both resource and latency inputs restored candidate 441's
exact baseline CUDA source in `/tmp/lfresourceall.E8FxK3/tuned`.

Local full native suite: 3748/3748 passed. Existing allocation-probe macro warnings
from the C toolchain remain; MoonBit warning-denied tests passed. End-to-end
throughput has not been remeasured by this follow-up, and shape-bucket/fusion/
timeline completion remains open.

Downloaded follow-up archives (SHA-256 verified locally):

- `/tmp/lunaflux-legal-folds-20260914.tar.gz`:
  `8c5e1e66a02278275abc19144e641ec41ed82bc967aad5d5e01f65b4e4b5d9fc`.
- `/tmp/lunaflux-resource-decode-20260914.tar.gz`:
  `fdd05dafbbc4ac7605d6a70966fca0d32ae3963412f5c0c38defd6f983c65155`.

## QKV, output and selected-row head search

`/tmp/lfmatrix.wbCwip` covers 40 legal modules, each with three workload cases,
five interleaved timing trials, numerical comparison and memcheck/racecheck/
synccheck. QKV/output use 32/1024/2048 tokens; head uses 2048 input tokens and
1/8/32 selected output rows. The source-only fixture uses the production lowerer
but these are isolated kernels, not an end-to-end release benchmark.

At 2048 tokens the baseline QKV is about 437 us and output 227 us. The
three-stage/window-four/lookahead-two alternatives take about 485 us and 252 us.
Head baseline is approximately 738/741/805 us for 1/8/32 selected rows. None of
the alternative lifetime plans establishes a long-input improvement; ordinary
two-stage/window-four behavior remains selected. Small sub-percent differences
between identical/default-equivalent schedules are not treated as gains.

Downloaded `/tmp/lunaflux-matrix-folds-20260914.tar.gz`, verified SHA-256
`fd7b4b42b574629fc6b9968b9aad079772a4d7c0bfb92e6ef274ac2292824848`.

## Measured selection and full/partial serving comparison

`/tmp/lffinalselect.Iy6cJP` consumes the combined real fold table. With a 1%
minimum-improvement margin, the entire exported candidate directory is identical
to baseline. Thus a noisy sub-percent head result does not replace the kernel.
The full/partial exporter accepts `--ingress-pair`, backed by the generic pure
fusion policy; no fabricated unfused timing is needed. The measured pair selects
partial fusion and reproduces its runtime bundle byte for byte.

`/tmp/lffusionchain.di9JcR/retry-cwd` compares sequential GPU-exclusive serving
runs, each with one warmup and five measured trials at four input/output lengths
and C1/C8/C16. Current full/partial generated CUDA sources were checked identical
to those in the reused runtime. This isolates the fusion choice; it is not a
fresh build of all current runtime code or a new three-framework campaign.

| Input/output/C | Partial median ms | Full median ms | Partial output tok/s | Full output tok/s |
| --- | ---: | ---: | ---: | ---: |
| 4096/64/1 | 549 | 625 | 116.58 | 102.40 |
| 4096/64/8 | 2365 | 2973 | 216.49 | 172.22 |
| 4096/64/16 | 4443 | 5547 | 230.47 | 184.60 |

Full fusion is slower throughout the tested matrix. At 3072/32, 44 request
comparisons differ only at the final token (ID 16 versus 22). Both paths also
vary across repetitions, so this is not yet attributable solely to full fusion;
fixed-shape/logit diagnosis remains necessary. Do not claim bitwise equivalence
or promote full fusion from these measurements.

The follow-up [fixed selected-graph repetition](BENCHMARK_FIXED_GRAPH_REPEAT_2026-09-15.md)
separates replay of an unchanged execution plan from changes in batching and
kernel schedules. Initial partial/full runs each compare 3072 logits rows
without changed bytes. These diagnostic historical-runtime results do not
close cross-batch numerical acceptance or current-source performance testing.

Downloaded archives, SHA-256 verified locally:

- `/tmp/lunaflux-fusion-pair-20260914.tar.gz`:
  `9520c3d42cf78932f1b5ccfd804b9a4789afb439404bc09839088ac922440364`.
- `/tmp/lunaflux-measured-selection-20260914.tar.gz`:
  `30a750dbb5269e252f9eb78ef48a12a51339e4526fcd9ed1f2972b7553b017f7`.

The new selection machinery intentionally keeps the faster established kernels.
It is not itself an end-to-end speedup; bucket-specific attention dispatch and
the new diagnostic trace remain open.

## Current worker trace and maximum-bucket launch gap

`/tmp/lftrace-current.0zPUUu/profile-v2` runs the current worker source with
diagnostic-only row/bucket/owner/execute markers and retained stderr. The first
attempt failed because the worker deliberately closes stderr; that diagnostic
failure and original binary are preserved. Production descriptor isolation was
not changed. All 352 observed steps in the successful rerun correlate one-to-one
with `cuGraphLaunch`; all 82560 kernel nodes correlate to those launches.

| Measured cell | Steps | Prefill tokens | Decode tokens | Query bucket slack | Initial route cost, total | Kernel time | Between-step GPU gaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096/64/C8 | 80 | 32768 | 504 | 43 | 88.353 us | 2267.9272 ms | 100.9466 ms |
| 4096/64/C16 | 96 | 65536 | 1008 | 43 | 132.654 us | 4281.9496 ms | 179.7420 ms |

These are instrumented attribution runs, not replacement uninstrumented
throughput numbers. Query slack measures capacity, not padded FLOPs. The trace
includes final output-demand owner selection, but route time covers only initial
bucket/phase selection. No timestamp offset is guessed: CUDA correlation IDs
and validated launch order join worker events to device nodes.

The selected prefill attention still launches 158 x 16 CTAs at the maximum
2048-token bucket. Code inspection shows that maximum fallback launch lists
bypass the launch-bound contract used by smaller buckets. The correction applies
the same contract at the profile maximum (and phase-specific maxima), preserving
fixed launches without such a contract. Grid-stride testing is reported below;
this is not a demonstrated physical speedup. Preparation was split
into focused files instead of growing the existing 798-line file.

Local validation before the maximum-bucket change: full native suite 3749/3749.
After the change: device-step package 192/192 and warning-denied native check.

### Maximum bucket: measured row-tail correction

Reducing the maximum grid to `ceil(total_queries / tile)` alone serialized
independent CSR row tails: C16 kernel time increased to 4309.0192 ms. The
corrected pure launch rule reuses `AttentionMetadataLayout.bucket` capacity,
including independent row tails. At 2048 queries / 32 rows / tile 64 the bound
is 63 rather than 32 (the original artifact launched 158). This is generic
metadata geometry, not model-specific tuning.

The corrected trace at `/tmp/lfrowbucket.uISyET/profile` has C16 kernel time
4276.7488 ms, between-step gaps 164.8558 ms, and the same 96 steps / 65536 prefill
tokens / 1008 decode tokens. All 3072 generated tokens across 48 requests match
the original trace. These small timing differences do not establish a speedup.

The uninstrumented rebuilt worker at `/tmp/lfrowbucket.uISyET/plain` completed
four input/output vectors at C1/C8/C16, one warmup and five measured trials.
4096/64 medians are 116.79 / 217.04 / 230.63 output tok/s at C1/C8/C16.
The previous partial-fusion run was 116.58 / 216.49 / 230.47; this is essentially
unchanged performance, not a closed baseline gap. No competitor was rerun here.

The baseline trace archive was downloaded and SHA-256 verified:
`/tmp/lunaflux-execution-trace-20260914.tar.gz`,
`24ec7fdfda5af096b8200403982a64fae913dde4a22c8f8ddac1b95d4d24f246`.
Measured bucket-table parsing/selection is implemented, but production bundle
binding and runtime selection remain unfinished. Full-fusion last-token
variation also remains open; neither is counted as complete.

Both maximum-grid experiments and the uninstrumented matrix were downloaded
without model duplication in `/tmp/lunaflux-rowbucket-results-20260914.tar.gz`;
local and remote SHA-256 agree:
`a79c630029861f46aac6e638aab105ee6c24ba8dc397ba328aa182c28581d416`.

### Measured-route integration and workload coverage

Commit `01ba1c46` closes the code-level bundle-to-owner connection described
above: bundle V6 carries scoped observations, startup creates immutable owner
indices, and dispatch applies measured baseline or wide-prefill choices before
heuristics. Packaging accepts V6. Native check and all 3,757 tests pass; Linux
release worker and exporter compile. A physically measured V6 routing table is
still pending, as is the full-fusion last-token diagnosis. Do not conflate code
integration with validated measured routing.

The new [workload diversity investigation](BENCHMARK_WORKLOAD_DIVERSITY_2026-09-14.md)
separates equal token counts from equal attention work and adds request-level
TTFT/tail measurements. Uniform regression cases alone cannot characterize
mixed workload behavior. Conversely, their uniformity does not explain the
lack of speedup when timing-based selection retains byte-identical kernels.
