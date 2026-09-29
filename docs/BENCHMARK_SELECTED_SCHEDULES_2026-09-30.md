# Selected schedules: fixing the gap between compiler and serving

**Result:** the selected 32-row ingress plus c322 prefill improves 4096/64 C16
from **165.64 to 190.48 tok/s (+15.0%)**, without the 64-row finalist's
short-input regression. This closes part, not all, of the baseline gap.

## What was wrong

The prior compiler change expanded the legal schedule domain, but the matched
serving run still selected the old ingress and prefill schedules. It therefore
did not exercise the larger-row or asynchronous alternatives. A larger search
space alone is not a performance improvement.

Real calibration then exposed two integration defects:

1. The metadata-aware bundle builder's second export omitted the full-ingress
   route, silently changing the modules while attaching metadata. It now
   preserves that route and verifies the resulting selection.
2. Runtime ingress admission and graph preparation assumed a 16-row CTA.
   Compiler-selected row ownership now travels through the AOT recipe and
   module metadata into admission and all prefill/decode/split-prefill bucket
   rules. Missing metadata retains the legacy 16-row contract; mismatched grids
   are rejected. Bucket tests cover both sides of 16/32/64-row boundaries.

Commits: `b982cbda`, `bb3dd18d`, `d67cb93e`. The runtime uses the completed
`d13ac969` benchmark snapshot with these fixes, retaining the isolated Spark
sm121 exporter adaptation. Unrelated working-tree changes are not included.

This preserves the functional architecture: pure compiler schedules determine
physical ownership, immutable artifact metadata carries it, and runtime graph
preparation consumes it. There is no model-specific performance branch added to
the scheduler, request-path compilation, or token-path artifact checking.

## Measurement contract

- Spark GB10, sm121, CUDA 13.0.88, GPU
  `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
- Qwen3-0.6B BF16; source weights SHA-256
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Token-ID requests, greedy generation, EOS ignored, prefix reuse disabled;
  input/output vectors 128/32, 4096/64, 4096/256 and concurrency 1/8/16.
- One warm-up and three measured trials per cell. The table uses median
  completion time, not a best trial. Every requested output length is checked.
- Engines and diagnostic probes run sequentially. A 32 GiB available-memory
  reserve is enforced; serving units have a 64 GiB limit and no extra swap.
- vLLM/SGLang columns are the completed matched baseline campaign, not new
  simultaneous runs. Same machine, model, workload generator and test matrix.
  Short-run variation and ordering remain limitations; this is not a general
  superiority claim across models or hardware.

Baseline images are `nvcr.io/nvidia/vllm:26.01-py3` (local image ID
`73e1b0f3230a9377a3109d20be7f8607963a41a715eb3000bc647f5f73c198ff`)
and `nvcr.io/nvidia/sglang:26.01-py3` (local image ID
`3a9d399434923689565e1a72f5c3fc409323ce039b65bdbe5c2d6ed14abad3a9`).
Their complete launch settings and container inspection are in the baseline
archive; these results do not claim comparison with every newer upstream build.

## Calibration and actual selection

The bounded search measured 12 ingress schedules, 16 prefill schedules and
four decode schedules. Five paired trials are retained for each probe case.
Ingress/prefill vectors were `(tokens, rows, prior history)`:
`(128,1,0)`, `(2048,1,0)`, `(2048,1,2048)`, `(2048,8,2048)`.
The probe uses each recipe's maximum launch envelope; these aggregate costs
are not estimates of small runtime graph buckets or end-to-end speedups.

The aggregate ingress winner was `ingress-g4-a2-s2-t1-f1-r4` (64 CTA rows).
Prefill selected c322 (query64/KV64, asynchronous). KV128 was not the measured
winner. Decode retained c441 among the tested alternatives; this is not a claim
that the remaining decode algorithm gap has been eliminated.

The existing source/device/toolchain-scoped pure selectors consume these real
observations. No invented timings or globally forced NVIDIA schedule is used.
Whole-serving comparisons of finalists remain necessary: the isolated kernel
winner can lose on the many small decode steps.

## Final 32-row selection: end-to-end result

Whole-serving measurements select `ingress-g4-a2-s2-t2-f1-r2` (32 rows),
with c322 prefill and c441 decode. The isolated aggregate winner was 64 rows,
but the 32-row alternative is a better balanced serving choice. The observed
0.15% difference between them at 4096/64 C16 is within the run spread.

Completion time in milliseconds; lower is better.

| Input/output | C | Previous LunaFlux | Selected LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 246 | 246 | 298 | 290 |
| 128/32 | 8 | 323 | 322 | 283 | 284 |
| 128/32 | 16 | 400 | 388 | 324 | 325 |
| 4096/64 | 1 | 802 | 747 | 772 | 780 |
| 4096/64 | 8 | 3251 | 2845 | 2275 | 2340 |
| 4096/64 | 16 | 6182 | 5376 | 4207 | 4237 |
| 4096/256 | 1 | 2589 | 2530 | 2705 | 2732 |
| 4096/256 | 8 | 8033 | 7707 | 6618 | 6831 |
| 4096/256 | 16 | 15028 | 14261 | 11819 | 11963 |

For 4096/64 C16 this is **13.0% less completion time / 15.0% more output
throughput**, from 165.64 to 190.48 tok/s. TTFT p50/p95 falls from 2095/3652 ms
to 1594/2811 ms. It still takes 27.8% longer than vLLM and 26.9% longer than
SGLang. At 4096/256 C16, throughput rises from 272.56 to 287.22 tok/s (+5.4%);
completion time remains 20.7%/19.2% above vLLM/SGLang.

The other whole-chain finalists used the same c322 attention:

| Cell | 16-row/window2 | 32-row/window2 | 64-row/window1 |
| --- | ---: | ---: | ---: |
| 128/32 C8, ms | 328 | 322 | 354 |
| 4096/64 C16, ms | 5516 | 5376 | 5368 |
| 4096/256 C8, ms | 7775 | 7707 | 7906 |

This is why blindly installing the microbenchmark winner is insufficient.
The complete raw nine-cell matrices for all three finalists are preserved.
The selection is specific to this measured workload/device; unmeasured targets
retain their existing fallback, not a hardcoded Spark-wide constant.

## Selected-kernel hardware counters

Paired launches of the exact packaged cubins, 2048 query tokens, one row,
2048 prior tokens. The reference and selected kernels use identical inputs.
These profiler timings are diagnostic and are not substituted for the
unprofiled serving times above.
The probes preserve the recipes' maximum launch envelopes; this is not a
measurement of the runtime's bucket-reduced inactive CTA count.

| Metric | QKV old → selected 32-row | Prefill old → c322 |
| --- | ---: | ---: |
| GPU duration | 1482.432 → 864.224 µs | 1782.688 → 1129.600 µs |
| CTAs | 4096 → 2048 | 2528 → 2528 |
| Warp instructions | 98,107,392 → 80,199,680 | 139,929,344 → 133,637,888 |
| Tensor activity, elapsed % | 11.03 → 18.95 | 27.93 → 44.08 |
| Long-scoreboard contribution, % | 23.73 → 31.91 | 47.38 → 20.68 |
| Barrier contribution, % | 4.62 → 4.03 | 2.91 → 3.79 |
| Registers/thread | 78 → 86 | 209 → 229 |
| Local load/store sectors | 0/0 → 0/0 | 0/0 → 0/0 |

The QKV change reduces repeated CTA work and instructions; it does not remove
all load stalls. Prefill overlaps loads much more effectively without changing
the number of launched CTAs. Higher registers and barrier percentages do not
invalidate the measured latency reduction. DRAM-read bytes were requested but
not returned in this capture and are not inferred.

## Correctness and limits

- 238 affected-package native tests pass; targeted native checks and interface
  generation pass. Known migration warnings are disabled with
  `--warn-list -79-20-29-25`; this is not a warning-clean whole-repository claim.
- Exact packaged ingress/prefill cubins pass memcheck with full leak checking,
  racecheck and synccheck: zero errors, leaks or race hazards/warnings.
- The independent sampled attention oracle's maximum absolute error was
  0.000407418 in the packaged-kernel check. The deterministic probe agrees
  bitwise with the reference for the checked workload.
- All 225 measured generated sequences in the 64-row serving run exactly match
  the previous LunaFlux run; the 16-row and final 32-row runs each also match
  225/225. This is regression agreement, not a broad model quality evaluation.
- Minimum sampled available memory in the selected run was 104,380,120 KiB
  (~99.5 GiB), comfortably
  above the 32 GiB reserve. No out-of-memory event occurred.
- The selected run recorded acknowledged drain, child exit code zero and
  closed child state. The 16-row helper had a redundant stop of an already
  unloaded transient unit after successful measurements/drain. Its failure
  logs were retained and diagnosed; the helper now waits for the supervisor's
  terminal record. The 64-row outer helper stopped before that record was
  saved, so it does not establish clean serving shutdown.
- No deployment cutover or claim of complete parity is made. Decode and the
  unfixed parts of the graph still limit end-to-end gains. Merely selecting a
  faster prefill kernel cannot make their contribution disappear.

## Reproduction and artifacts

`benchmarks/gpu_pipeline/calibrate_selected_policies.mbtx` compiles and probes
the bounded frontier and exports the measured selection.
`select_policy_subset.mbtx` retains the actual reference/finalist observations
in a separate table for whole-chain comparison; it never rewrites the original
table. `run_selected_runtime.mbtx` builds the chosen bundle, checks packaged
cubins and runs the matched serving matrix. `profile_selected_policies.mbtx`
captures two matching launches per family after serving has stopped.

These are offline Spark benchmark helpers, not production runtime dependencies.
They take the prepared matched campaign root (toolchain, source, release, model
receipts and benchmark client); the diagnostic model path and probe geometry
are explicit in the helper. New output directories must be empty.

Remote artifacts:

- Matched baseline: `/home/wlc004s/lunaflux-bench-d13ac969.2dIigB07`.
- Ingress/prefill calibration: `/home/wlc004s/lunaflux-calibration-rows-20260930.9SjBwt3Q`.
- Final calibration/export: `/home/wlc004s/lunaflux-calibration-complete-20260930.PH0tGhdZ`.
- 64-row serving/counters: `/home/wlc004s/lunaflux-selected-e2e-20260930.g7kQaLnJ`.
- 16-row serving: `/home/wlc004s/lunaflux-selected-finalist-20260930.BsoeouBD`.
- Selected 32-row serving/counters: `/home/wlc004s/lunaflux-selected-32row-20260930.jsY7qx3o`.

To reproduce the final bounded selection, use the full calibration root as
`MEASURED_ROOT`, then run the serving helper against that new finalist root:

```sh
moon run benchmarks/gpu_pipeline/select_policy_subset.mbtx BASE_ROOT MEASURED_ROOT NEW_FINALIST_ROOT ingress-g4-a2-s2-t2-f1-r2
moon run benchmarks/gpu_pipeline/run_selected_runtime.mbtx BASE_ROOT NEW_FINALIST_ROOT NEW_SERVING_ROOT ABSOLUTE_BUILDER_PATH
```

The serving root must already contain `selected_policy_probe.cu`. The baseline
root supplies the pinned source/runtime and matched workload helpers. Selecting
a finalist is an offline build step; no request-path search is introduced.

The baseline archive is downloaded at
`/tmp/lunaflux-committed-bench.BXcuiTIJ/results-d13ac969.tar.gz`, SHA-256
`bc5be2650a94d3c55b7156830262988b9de5466c7252200db87a401302033b8f`.

Downloaded calibration and all three serving archives, with locally verified
SHA-256, are in that same local directory:

| File | SHA-256 |
| --- | --- |
| `calibration-results.tar.gz` | `7b5d774040d3f92a03ba8886440d310b080b8203c8e009871d503e401f786d27` |
| `selected-64-row-results.tar.gz` | `d0ea18755e129e60bbb73d64c75c551b850e00fcc5b4a14a65c1149ecb090aca` |
| `selected-16-row-results.tar.gz` | `ce2a4960ae6664d1457194d829001be26472f4c2c982f9cdf7cfdf56d5f2761a` |
| `selected-32-row-results.tar.gz` | `98fc6bbf255b9e12c046b5020f6d426fe0f73730a419fd9a315883de4c939a6d` |
