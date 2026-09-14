# Functional compiler integration work

Scope: complete the seven optimization/integration directions discussed after
`a1adf4cc`, without removing supported shapes or moving CUDA policy into model
or scheduler code. This is a work ledger, not a performance completion claim.

The authoritative path is pure semantics → immutable strategy/layout choice →
explicit effect/lifetime plan → backend lowering. Offline physical observations
are inputs to selection; they are never collected inside a token step.

| Work | Completion criterion | Current work |
| --- | --- | --- |
| Projection pipeline search | Generate legal stage/window alternatives, measure and select them; backend implements the selected lifetime plan | Resource filtering and source-bound offline selection integrated; 19 legal MLP combinations physically measured, baseline wins; other projection families remain |
| Operand segmentation | Shared semantic address/segment plan consumed by projection lowerers; preserve useful schedules rather than force losing hoists | Pending; previous QKV hoist regressed |
| Attention ownership/history | Preserve existing c322 semantics and correctness through new selection | Existing path; regression required |
| Attention shape buckets | Query/history/batch-specific measurements choose admitted artifacts through bounded runtime dispatch | Pending |
| Resource feedback | Measured budgets/register facts reach resource-aware compiler selection, including expanded schedules | Prefill/decode/partitioned exporter connected; decode resource-only choice regresses, measured override physically restores baseline source |
| Fusion chain selection | Compare full versus partial chain cost, select through a generic immutable policy, validate both | Pending |
| Execution diagnostics | Latest runtime trace includes actual work, padded work, selected bucket/route, mixed steps and GPU gaps | Prior trace exists; new coverage pending |

No item is complete merely because an interface exists. Completion includes
public behavior tests, integration tests, exact-source physical correctness,
relevant sanitizer checks and matched end-to-end timing. A slower experiment
may establish a cause but does not become the production choice by default.

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
