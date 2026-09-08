# Performance parity work — 2026-09-08

Target: close the measured Qwen3-0.6B BF16 gap to vLLM and SGLang on the
same RTX 5060 Ti, without changing the model, input tokens, output-token
budget, prefix-cache policy, or numerical correctness requirements.
This is a target, not a performance claim.

## Independent optimization tracks

| Track | Change | Isolated comparison |
| --- | --- | --- |
| Prefill pipeline | Represent cross-iteration storage lifetimes and lower two-stage transfer/compute overlap | Synchronous vs asynchronous matrix attention at identical query/KV geometry |
| Projection buckets | Export measured head/MLP row variants and bind their exact launch geometry at startup | Same shape and input values, varying work distribution/residency only |
| Mixed attention | Pure startup rewrite into disjoint prefill/decode attention domains | Original mixed attention vs specialized mixed owner, identical CSR rows and KV writes |

Strategies and effect ordering remain immutable planning values. Device-specific
copy instructions and matrix execution stay in CUDA lowering. Token execution
must not perform tuning, code generation, module lookup, or heap allocation.

The projection token-row bound is unchanged: a record measured at 256 tokens
does not become a 1024-token record. A multi-row record also must not silently
replace the single-token GEMV schedule.

## End-to-end comparison matrix

Use input/output vectors `(59,256)`, `(128,128)`, `(512,64)`, `(1528,32)`,
each at concurrency 1 and 8. Report output tokens/s, time to first token,
inter-token spacing, completed requests, actual emitted token counts, and
output-token agreement. Include every cell, not only the best improvement.

First compare each change independently against its exact-source control;
then compare the combined implementation against both baselines. Reuse the
existing request corpus and model. Run GPU work serially, with warmup excluded
from timing, and retain repeated measurements rather than only a best sample.
Sanitizer/instrumented timings are not serving-performance results.

Existing September 8 measurements in
`docs/QWEN_PERFORMANCE_DISTANCE_2026-09-08.md` are the starting reference, not
measurements of these changes. The 1024-token profile also changed LM-head
residency selection, so its difference from the compact profile is not a pure
prefill-chunk experiment. Refresh matched baseline measurements before claiming
parity; literal equality of noisy timing samples is not an acceptance criterion.

## Completion boundary

Each track needs focused software regressions, exact generated-kernel physical
correctness checks where affected, and an isolated timing result. End-to-end
parity is only established by the complete matched workload matrix. A slower
candidate remains available for investigation but must not replace a faster
default merely because it implements a more sophisticated compiler schedule.

Production deployment is outside this work item. Preserve unrelated worktree
changes and commit the independent changes separately.

## Current checkpoints

- `9190fee`: asynchronous prefill compiler schedules; affected packages 55/55.
- `590ecef`: measured head/MLP row-variant export and startup consumers;
  exporter 19/19, focused projection consumer tests 12/12.
- `59f822e`: mixed-phase immutable execution rewrite; device-step 169/169 and
  device package 18/18. These ordinary tests do not run the opt-in GPU fixture.

Isolated physical results, not end-to-end serving improvements:

| Kernel case | Control | New | Speedup |
| --- | ---: | ---: | ---: |
| LM head, 8 rows | 2487.602 us | 1530.648 us | 1.625x |
| LM head, 32 rows | 4956.419 us | 1649.326 us | 3.005x |
| MLP pair, 8 rows | 56.410 us | 30.485 us | 1.850x |
| Prefill Q64/context4096, Q32 tile | 604.322 us | 480.309 us | 1.258x |
| Prefill Q128/context4096, Q32 tile | 823.442 us | 954.831 us | 0.862x |

Projection results are medians of three warm trials on deterministic synthetic
tensors with the actual generated Qwen-sized kernels. Output/workspace equality,
untouched tails, memcheck, and bounded racecheck passed. The earlier unbounded
head racecheck was interrupted and is not counted as a pass. No new production
tuning file was installed from these diagnostic selector fixtures.

Projection results archive:
`/private/tmp/lunaflux-projection-variants-results-20260908-r1.tar.gz`,
SHA-256 `38f8d95b78565298b642787498b62e09786e0475e6a6c96290383572f41ce7a4`.
See `docs/PREFILL_TRANSFER_PIPELINE_2026-09-08.md` for the complete prefill matrix,
correctness checks, shared-memory tradeoff, and result archive.

The runtime-only mixed-phase Qwen A/B uses committed source `59f822e` and
unchanged earlier 1024-profile kernel artifacts, at scheduler budgets 256 and
1024. This isolates execution routing; it does not measure newly exported
projection variants or enable the async prefill candidates. Its source archive
SHA-256 is `42d464d989f3d41fd32f2f6dfa10fb1b2269d867b6f095ba3bed9f9147c5a1bf`.
Results are pending; no parity claim is established.
