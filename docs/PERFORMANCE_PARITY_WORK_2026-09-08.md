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
That initial runtime A/B is **invalid as a performance improvement**: long-input
C8 output sequences diverged substantially. It exposed a descriptor-publication
bug: the narrowed phase counts were placed beyond the variable-length region
actually copied to the GPU. The synthetic executor fixture copied its complete
buffer directly and did not exercise this seam.

Fix `07a3c33` moves phase counts into fixed prefix cells 15–19 and shifts the
output-row suffix to cell 20. The same single upload now includes them, without
another transfer. A real descriptor `stage`/`stage_frame` upload regression was
added; device-step tests pass 170/170. The original r1 results remain preserved
and excluded from improvement claims. The corrected r2 runtime A/B uses
source archive SHA-256
`b4d9700a91993bcfa81c67c64361506a37aa5e3cc3ab51383c0c84672c726040`.
The corrected GPU descriptor/executor tests pass 2/2, including actual submitted
and wire-frame uploads, projected/unprojected output modes, zero-output frames,
eager/captured execution, and budget fallback. Memcheck and racecheck report
zero errors/hazards. The fixture results remain on the host under
`/dev/shm/lunaflux-parity-mixed-20260908-r2/phase-fixture`; downloading its archive
was permission-blocked and was not retried through another route.

### Corrected runtime-only Qwen comparison

Same 1024-profile kernel bundle, scheduler budget/chunk 256/256. Means of the
two timed trials (one warmup excluded). The prior same-day control is `510d07b`;
this is not a fresh vLLM/SGLang measurement.

| Input/output tokens | Concurrency | Control tok/s | Fixed tok/s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 59/256 | 1 | 200.392 | 200.235 | 0.999x |
| 59/256 | 8 | 859.242 | 859.601 | 1.000x |
| 128/128 | 1 | 184.438 | 183.776 | 0.996x |
| 128/128 | 8 | 794.724 | 794.106 | 0.999x |
| 512/64 | 1 | 152.019 | 152.205 | 1.001x |
| 512/64 | 8 | 460.646 | 461.890 | 1.003x |
| 1528/32 | 1 | 76.832 | 77.295 | 1.006x |
| 1528/32 | 8 | 112.825 | 117.377 | 1.040x |

All 72 timed request sequences match a sequence observed in the corresponding
control cell; all emitted token counts match their requested budgets. The
59/256 C8 control had three distinct sequences, while this run had one (present
in that control set), so this is set membership, not a claim of deterministic
request-index equality across different arrival orders. Long C8 had one exact
sequence in each run. Long-C8 mean TTFT improves 1136.313 to 1103.250 ms;
mean inter-token spacing improves 35.222 to 33.464 ms.

Same comparison with scheduler budget/chunk 1024/1024:

| Input/output tokens | Concurrency | Control tok/s | Fixed tok/s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 59/256 | 1 | 200.942 | 201.179 | 1.001x |
| 59/256 | 8 | 863.771 | 862.502 | 0.999x |
| 128/128 | 1 | 184.173 | 183.908 | 0.999x |
| 128/128 | 8 | 815.936 | 813.022 | 0.996x |
| 512/64 | 1 | 155.718 | 155.531 | 0.999x |
| 512/64 | 8 | 486.692 | 489.017 | 1.005x |
| 1528/32 | 1 | 82.904 | 82.581 | 0.996x |
| 1528/32 | 8 | 132.988 | 133.682 | 1.005x |

All 72 timed sequences in this arm also belong to their corresponding control
set; counts are correct. The long-C8 control had seven distinct sequences and
this run one, present in the control set. Long-C8 TTFT is 1281.688 to 1278.125 ms,
spacing 20.111 to 19.873 ms. Changes below 1% in this two-trial experiment are
not persuasive performance gains. Both benchmark servers stopped and the final
GPU process inventory was empty.

Raw corrected results remain in these remote directories:

- `/dev/shm/lunaflux-baselines-20260906-r1-lunaflux-parity-mixed-b256-c256-20260908-r2`
- `/dev/shm/lunaflux-baselines-20260906-r1-lunaflux-parity-mixed-b1024-c1024-20260908-r2`

## Remaining distance

This work establishes useful isolated projection/prefill wins and a modest
runtime-only mixed-phase gain at budget256, not end-to-end parity. It does not
multiply kernel speedups into a hypothetical serving result. The prior same-day
long-C8 baselines were 443.7 tok/s (vLLM) and 419.7 tok/s (SGLang), versus this
runtime-only 133.682 tok/s at budget1024: approximately 3.32x and 3.14x faster
than LunaFlux respectively. Those baselines were not rerun in this work item.

Next, generate the serving bundle with genuine scope-bound measured projection
records (retaining the existing head256 record without relabeling), choose
prefill pipeline schedules only for winning shape regimes, and measure the
combined implementation against freshly repeated baselines. The isolated test
selector records are not production tuning measurements. No parity claim is
established.
