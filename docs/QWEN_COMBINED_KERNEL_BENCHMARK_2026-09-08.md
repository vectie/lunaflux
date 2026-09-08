# Latest bindable projection kernels: paired Qwen benchmark — 2026-09-08

## Result

Installing the measured head/MLP row variants into the serving bundle improves
C8 output throughput by **36.1%, 32.9%, 17.2%, and 4.0%**, respectively, over
the matched old-kernel control. C1 changes are below 1%. This corrects the
impression that the latest kernel work produced only the previous 0–4% gain:
that earlier experiment reused the old kernel artifacts and tested runtime
routing only.

This is **not** a fully combined async-prefill result. The current Qwen
exporter explicitly disables asynchronous copy, and no production async
selection policy was added in this benchmark. This turn changes no production
kernel, compiler, scheduler, model, or deployment code.

## Matched experiment

- Qwen3-0.6B BF16, identical token-ID inputs, greedy, fixed output budgets,
  ignore EOS, prefix reuse disabled; RTX 5060 Ti only.
- GPU UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI
  `00000000:17:00.0`, CUDA 13.1.115, sm_120.
- Both arms use runtime source `07a3c33`; HEAD `8112868` differs from it only
  in documentation. Unrelated uncommitted model-family work is excluded.
- Both arms retain the same 1024-token kernel envelope and scheduler
  budget/chunk 1024/1024. The control uses the old granularity bundle; the
  optimized arm regenerates the source-derived bundle with measured head256
  and MLP8/group2 records. Both use the corrected mixed-phase runtime.
- One control→optimized pass and one optimized→control pass. Each cell has
  one excluded warmup and two timed trials per pass: **four timed trials per
  cell per arm**. Tables show arithmetic means of per-trial output rates and
  request-level latencies, not confidence intervals.
- The existing MoonBit streaming client runs on the server host; serving
  measurements include client/transport overhead and exclude model startup.
  No profiler or sanitizer instrumentation is included in these timings.
- Same paired runtime, different kernel bundle: this measures the combined
  head/MLP selection change, not separate causal percentages for each kernel.

## Throughput

Output tokens/second; C8 is aggregate throughput.

| Input/output tokens | C | Old kernels | New kernels | Change |
| --- | ---: | ---: | ---: | ---: |
| 59/256 | 1 | 200.824 | 200.667 | −0.08% |
| 59/256 | 8 | 861.772 | 1172.809 | +36.09% |
| 128/128 | 1 | 183.778 | 184.043 | +0.14% |
| 128/128 | 8 | 813.995 | 1081.596 | +32.88% |
| 512/64 | 1 | 154.595 | 155.348 | +0.49% |
| 512/64 | 8 | 487.627 | 571.590 | +17.22% |
| 1528/32 | 1 | 82.423 | 83.010 | +0.71% |
| 1528/32 | 8 | 133.716 | 139.112 | +4.03% |

The forward pass alone showed +36.43/+32.52/+17.45/+4.13% at C8;
the combined order-balanced means retain the same conclusion. Small C1
differences are not persuasive improvements. There is no workload-independent
overall percentage without specifying workload weights.

### Where the latency changed

Mean request TTFT and post-first-token spacing, milliseconds. These are
experienced request latencies, not isolated kernel timings.

| Input/output | C | TTFT old → new | Spacing old → new |
| --- | ---: | ---: | ---: |
| 59/256 | 1 | 20.000 → 21.500 | 4.905 → 4.906 |
| 59/256 | 8 | 53.375 → 53.594 | 9.086 → 6.613 |
| 128/128 | 1 | 27.000 → 26.000 | 5.250 → 5.248 |
| 128/128 | 8 | 91.531 → 90.094 | 9.146 → 6.699 |
| 512/64 | 1 | 63.750 → 61.750 | 5.528 → 5.520 |
| 512/64 | 8 | 310.531 → 311.000 | 11.479 → 9.057 |
| 1528/32 | 1 | 202.000 → 198.750 | 5.944 → 5.935 |
| 1528/32 | 8 | 1278.875 → 1276.813 | 19.834 → 17.538 |

## Why the earlier gain was small, and why long prefill still is

1. **Artifact integration was missing from the earlier test.** The old
   runtime-only A/B did not regenerate head/MLP modules. The new bundle
   contains the head `_rows256`, MLP `_rows8`, and MLP `_rows8_down` symbols;
   startup builds bounded execution variants through
   `engine/device_step/projection_variants.mbt` and
   `paged_ordered_executor_prepare.mbt`. The measured serving gain now includes
   these changes. No new kernel-level profiling trace was collected, so this
   is a paired bundle result, not per-kernel time attribution.
2. **The optimization has a specific shape domain.** The MLP record covers
   at most eight token rows. A one-token graph keeps its existing GEMV path
   unless an explicit row-one record exists. Large prefill steps do not use
   this small-row MLP record. C1 remaining flat is therefore unsurprising.
3. **Long-prefill execution is essentially unchanged.** Long-C8 TTFT remains
   about 1.28 seconds, while spacing falls by about 2.30 ms. Thirty-one
   post-first-token intervals save roughly 71 ms per request; this is modest
   beside the unchanged first-token wait. TTFT includes scheduling and other
   model operations, so do not attribute its entire duration to attention.
4. **Async prefill is not connected to Qwen artifact selection.**
   `cmd/lunaflux_qwen3_bf16_candidate_export/main.mbt` still supplies
   `supports_async_copy=false` in both prefill capability constructions and
   an empty attention autotune table. Compiler candidates 316/317 exist, but
   the exported serving bundle cannot select them. Their isolated gains are
   also shape-dependent; the Q32 pipeline regresses on Q128 workloads, so a
   universal async switch is not justified.
5. **A kernel speedup is not a whole-model multiplier.** Head/MLP improvements
   leave QKV, attention, output projection, prefill scheduling, and transport
   work. The current data supports improving physical schedules and connecting
   their shape-conditioned choices, not adding generic compiler passes
   indiscriminately. Pure semantics and effect-ordered lowering are preserved.

See [the async pipeline matrix](PREFILL_TRANSFER_PIPELINE_2026-09-08.md) and
[the earlier runtime-only result](PERFORMANCE_PARITY_WORK_2026-09-08.md).

## Remaining comparison to vLLM/SGLang

The following competitors were measured earlier on September 8 with the same
model/GPU/token vectors; **they were not rerun in this work item**. This is a
reference comparison, not a freshly order-balanced three-engine campaign.

| Input/output | C | New LunaFlux | Earlier vLLM | Earlier SGLang |
| --- | ---: | ---: | ---: | ---: |
| 59/256 | 1 | 200.7 | 274.9 | 264.6 |
| 59/256 | 8 | 1172.8 | 1848.4 | 1760.2 |
| 128/128 | 1 | 184.0 | 270.6 | 258.9 |
| 128/128 | 8 | 1081.6 | 1750.4 | 1646.3 |
| 512/64 | 1 | 155.3 | 247.1 | 240.2 |
| 512/64 | 8 | 571.6 | 1187.9 | 1131.5 |
| 1528/32 | 1 | 83.0 | 178.3 | 171.6 |
| 1528/32 | 8 | 139.1 | 443.7 | 419.7 |

Short-C8 competitors are now approximately 1.58×/1.50× LunaFlux rather than
2.14×/2.04× for this matched old profile. Long-C8 remains approximately
3.19×/3.02× behind. The older compact 256-profile LunaFlux bundle remains
faster in some C1 cells; this table is not a per-cell blend of the best
historical configurations. [Baseline details](QWEN_PERFORMANCE_DISTANCE_2026-09-08.md).

## Correctness, scope, and reproduction

All 144 timed optimized request sequences match their corresponding control
cell's sequence; each cell has one unique sequence across both passes. All
requested token counts and terminal events are correct. This is same-model
control agreement, not an independent numerical or cross-engine quality
qualification. The prior isolated variant correctness and sanitizer checks
remain separately scoped. The final GPU compute-process list is empty.

The existing head256 measured record is retained unchanged; it is not relabeled
as a 1024-token measurement. MLP8/group2 uses the actual median of three prior
GPU trials: 30.485119, 30.439680, 30.636160 µs, recorded as 30485 ns with
sample_count=3. No synthetic 1 ns selector fixture was installed. The first
assembly attempt rejected noncanonical tuning-record ordering before
execution; it is preserved at `lunaflux-combined-20260908-r1` and excluded.

Remote bundle: `/dev/shm/lunaflux-combined-20260908-r2`.

- `projection-tuning.v1`: `c0cabf0a07d5fc7695b7c877d8a1546abc4d985f01a236dfeb16a24937b6782d`
- `release-bind.stdout`: `9ba787e02e76a3a09eaf63d3893b6bc23a39b98610c7ff86d4dc15d1082a00b3`
- `runtime.v3`: `c66946edca9d728aae2289dfa27f2bca712d9b5e6fb2506dc42f63955c50cb18`
- Runtime source archive: `b4d9700a91993bcfa81c67c64361506a37aa5e3cc3ab51383c0c84672c726040`
- Worker executable: `72f267f4b27ff0ca49e30462a837c23aa475dc8cd58d2f1e4387a3534d60c5b5`

Raw request JSON/SSE/timestamps remain in four remote directories beginning
`/dev/shm/lunaflux-baselines-20260906-r1-lunaflux-` and ending:

- `combined-control-20260908-r2`
- `combined-optimized-20260908-r2`
- `combined-reverse-control-20260908-r2`
- `combined-reverse-optimized-20260908-r2`

MoonBit orchestration/analysis scripts are in `/private/tmp` locally and
`/dev/shm` remotely: `lunaflux-combined-r2-20260908.mbtx` (prepare/setup),
`lunaflux-combined-e2e-20260908.mbtx` (control then optimized),
`lunaflux-combined-reverse-20260908.mbtx`, and
`lunaflux-combined-summary-20260908.mbtx`. Repeating requires new output paths;
the scripts intentionally refuse overwrite. Production deployment was untouched.
