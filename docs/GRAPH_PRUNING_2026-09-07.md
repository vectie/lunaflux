# Effect-only captured graph specialization

Implementation: `e541841`.

## What changed

The backend-neutral output-demand analysis already identifies the last
persistent effect. The ordered-executor consumer now specializes a second
immutable launch list ending at that effect and captures that shorter list.
Previously only eager execution omitted the suffix; captures still launched
the suffix's predicated kernels.

The hot path selects a startup-owned executor by index when output rows are
zero. It does not rebuild a graph, inspect files, allocate storage, or change
kernel arithmetic. Mixed frames with any producing row retain the full graph.
Diagnostic canaries and FP8 envelopes also retain their existing full graph.

`ExecutionGraphVariantBudget::reserve` is a pure, backend-neutral transition.
Ordinary AOT preparation allows at most 64 optional captures and 32,768 extra
kernel nodes. These are structural limits, **not** measurements or bounds on
CUDA's private graph-storage bytes. Existing specializations with declared
graph-memory bounds do not acquire extra captures: their byte contracts must
be extended before enabling this optimization. Unsupported capture remains
unchanged. Exhaustion or optional capture creation failure retains the full
executor; an eager fallback is never substituted for an existing capture.

All variants lease the existing functions, stream, and allocations. Ownership
is recorded before releasing a failed optional capture, so normal construction
cleanup can retry a failed close.

## Validation

- Native warning-denied check and formatting passed.
- `engine/device_step`: 158 local tests passed. GPU-gated tests return without
  CUDA on the local machine; this is not a physical-validation claim.
- Execution-graph strategy: 3 tests passed, including immutable budget
  reservation and independent graph/node exhaustion.
- Linux warning-denied check and release build passed for the exact committed
  source archive.
- Physical effect-only capture test passed on RTX 5060 Ti. The test alternates
  full and pruned captures, verifies effect counters and omitted head/sample
  counters, exercises budget exhaustion, and closes all shared leases.
- CUDA memcheck, racecheck, synccheck, and initcheck all passed. The first test
  build emitted an existing dependency C warning about
  `posix_spawn_file_actions_addchdir_np`; sanitizer runs reported zero errors.

## Qwen3-0.6B BF16 comparison

Runtime `e541841`, reusing the unchanged aligned partial-ingress kernel set
from `8b1f8f5`. Same RTX 5060 Ti, token IDs, greedy seed 0, ignored EOS, prefix
reuse disabled, one warmup and two timed repetitions. Baseline is the saved
aligned partial-ingress run, not a fresh interleaved control. Numbers are the
arithmetic mean of output tokens/second, including prefill time.

| Input / output | Concurrency | Before | Pruned graphs | Change |
|---|---:|---:|---:|---:|
| 59 / 256 | 1 | 232.52 | 232.73 | +0.1% |
| 59 / 256 | 8 | 971.77 | 969.70 | −0.2% |
| 128 / 128 | 1 | 222.22 | 222.22 | 0.0% |
| 128 / 128 | 8 | 892.77 | 895.50 | +0.3% |
| 512 / 64 | 1 | 179.78 | 180.04 | +0.1% |
| 512 / 64 | 8 | 500.98 | 503.44 | +0.5% |
| 1528 / 32 | 1 | 86.60 | 87.20 | +0.7% |
| 1528 / 32 | 8 | 125.95 | 127.14 | +0.9% |

These small differences do **not** establish an end-to-end speedup. The
functional optimization is implemented, but this workload remains dominated
by other costs. All output sequences belonged to their baseline cell's
previously observed set. Both runs have three unique sequences at 59/256 C8
and one in each other cell; this is not request-ordinal equality or a claim
that the existing batch-dependent variation is fixed.

No new GPU kernel was introduced: this change specializes the execution graph
around existing kernels. Competitor frameworks were not rerun in this test.

Remote results: `/dev/shm/lunaflux-graph-pruning-e541841-20260907-r1` and
`/dev/shm/lunaflux-baselines-20260906-r1-lunaflux-graph-pruned-e541841`.

Result archive: `/private/tmp/lunaflux-graph-pruning-results-20260907.tar.gz`.
SHA-256: `51b337c6177b94faa28b5058dd32c0234c55aa49180428902c32a4b9f33d8dcb`.
The isolated server was stopped after the run; the final GPU process list is
empty. Production deployment was not changed.

## Remaining scope

This is output-demand specialization, not per-bucket projection-artifact
selection. The latter still needs an exporter and runtime representation for
multiple concrete kernel variants. It also does not establish batch-invariant
floating-point reduction, resolve the existing C8 token variation, or replace
the larger matrix/attention schedule work.
