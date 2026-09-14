# Functional compiler integration work

Scope: complete the seven optimization/integration directions discussed after
`a1adf4cc`, without removing supported shapes or moving CUDA policy into model
or scheduler code. This is a work ledger, not a performance completion claim.

The authoritative path is pure semantics → immutable strategy/layout choice →
explicit effect/lifetime plan → backend lowering. Offline physical observations
are inputs to selection; they are never collected inside a token step.

| Work | Completion criterion | Current work |
| --- | --- | --- |
| Projection pipeline search | Generate legal stage/window alternatives, measure and select them; backend implements the selected lifetime plan | Pending |
| Operand segmentation | Shared semantic address/segment plan consumed by projection lowerers; preserve useful schedules rather than force losing hoists | Pending; previous QKV hoist regressed |
| Attention ownership/history | Preserve existing c322 semantics and correctness through new selection | Existing path; regression required |
| Attention shape buckets | Query/history/batch-specific measurements choose admitted artifacts through bounded runtime dispatch | Pending |
| Resource feedback | Measured budgets/register facts reach resource-aware compiler selection, including expanded schedules | Prefill exporter connected and physically checked; decode/partitioned export integration remains |
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

Remote run: `/tmp/lfresources.mANp0Z`. Downloaded archive:
`/tmp/lunaflux-resource-feedback-20260914.tar.gz`, SHA-256
`0929eb86326a660ba85ce3ab1c4632426792c6abb90ab383a63a5284720ed922`.
It contains the validation driver, device query, actual resource output,
candidate exports and results. The earlier baseline latency observations are
reused explicitly, not represented as fresh measurements.
