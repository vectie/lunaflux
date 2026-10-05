# AKO small-batch projection follow-up — 2026-10-05

## Decision

Follow-up: [bounded ingress routing](BENCHMARK_AKO_INGRESS_ROUTING_2026-10-05.md)
now propagates the alternative into serving buckets and records fresh A/B
results. It remains opt-in, with serving sanitizer qualification blocked.
The kernel-only conclusions and retained regressions below are unchanged.

The finite gate/up and QKV experiments are complete. **No production schedule
was promoted and no new serving throughput is claimed.** The experimental
gate/up bootstrap lowering was removed after it failed the paired performance
gate; its generated sources, results and sanitizer logs remain archived.

The important correction is benchmark cache state: reusing one layer's weights
can reverse a QKV decision. Distinct-layer operands expose a useful QKV
small-batch alternative, but that alternative regresses on larger prefill.
It must not replace the sole serving ingress schedule globally.

The `lunaflux-ako` skill determined the bounded profile/modify/measure/record
loop and retention of failed experiments. All additions are offline probes or
MoonBit `.mbtx` automation. Production semantics, numeric contracts, functional
IR passes and runtime ownership boundaries are unchanged.

## Exact baseline and measurement boundary

GB10 DGX Spark, CUDA 13.0.88, `sm_121`, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
Pinned `nvcc` SHA-256:
`c3e8741c84713f825ef038de8f80529fbeb55abe9e85066ec89cc7f9aa862857`.
Baseline root:
`/home/wlc004s/lunaflux-domain-serving-v4-20261004.QmE9WZ7s`.

Selected MLP module:
`bf10e22f8213f89a8a99f4530e1928051111f670101b84b0785201918ab949fe`.
Gate/up uses the bounded `rows8`/`rows16` symbols, grid 96, block 64;
down uses grid 64, block 32. Gate/up has 20 KiB static plus 16 KiB dynamic
shared reservation. The dynamic reservation ablation uses identical cubin
bytes with only the launch reservation changed.

Selected full QKV symbol:
`lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_cached_rotary_v3`.
Its actual source SHA-256 is
`82dc995df98f2c1100a09e564e0c91dbfef0df2a00a8e71f6aa2d473291a3019`.
Baseline geometry is 64 query rows, one head/CTA, 128 threads, transfer width 32,
two pipeline stages. Exported `g4-a2-s2-t2-f1-r4` source is byte-identical to
the serving source, checked before the controlled trial. The new schedules
come from the existing compiler/exporter, not a handwritten model-specific
kernel or runtime JIT.

All timing cells contain five alternating A/B pairs. Each QKV sample averages
30 launches; MLP averages 20. Measurements use CUDA events, including both MLP
launches for the complete-chain column. Uploads, allocations, correctness and
rotary preparation are outside the timed interval. Rotary preparation remains
once per decoder step and is reported separately by the QKV probe; these are
consumer measurements, not complete decoder-step times.

QKV geometry is runtime-bucket-derived: tokens 1/2/8/16/17/32 use equally many
request rows; 64/128/512 use one request. Thus this vector varies both request
count and token count; it is not solely a sequence-length scaling curve.

Two cache conditions are deliberately kept separate:

- **Cached:** repeatedly use one operand set.
- **Distinct layer weights:** allocate 28 identical-valued but separately
  addressed operand sets. QKV uses 224 MiB; gate/up/down together use 504 MiB.
  No cache-flush kernel or upload is timed. QKV warms a complete 28-layer pass;
  MLP retains its original three-launch warmup and 20-launch timing boundary.
  All samples, including startup/noise outliers, are retained.

This approximates a decoder's weight working-set pressure. It is not a trace
replay of all intervening operations or a substitute for serving validation.

## Gate/up: both proposed changes rejected

Hypotheses: remove the unused dynamic reservation; or stage the initial input
through the same asynchronous retained effect as subsequent operands, instead
of its synchronous bootstrap branch. Neither produced a robust 3% gain in
every pair. The bootstrap change passed correctness but was not kept in
production.

| C16 measurement | Baseline | Candidate | Decision |
| --- | ---: | ---: | --- |
| Reservation removed, cached gate/up | 12.349 µs | 12.342 µs | Inconclusive |
| Reservation removed, distinct gate/up | 60.648 µs | 60.546 µs | Inconclusive |
| Async bootstrap, cached gate/up | 12.336 µs | 12.346 µs | Slight regression |
| Async bootstrap, distinct gate/up | 60.542 µs | 60.362 µs | Inconclusive |
| Async bootstrap, cached complete MLP | 40.970 µs | 40.421 µs | Inconclusive; minimum paired gain 0.66% |
| Async bootstrap, distinct complete MLP | 97.389 µs | 96.950 µs | Inconclusive; minimum paired gain −3.86% |

The roughly **fivefold cached/distinct gate/up difference** explains why a
12-µs cached probe cannot account for the approximately 60-µs selected serving
launch. Removing a small input bootstrap is not enough to solve this weight
supply pressure. Other rows and all down/whole-chain samples are in
`summary.json`; small negative or noisy outcomes are not silently omitted.

Baseline gate/up NCU replay recorded 90 registers/thread, one wave/SM,
700,032 warp instructions, 2.98% tensor activity, 2.68% issue activity,
long-scoreboard/active-issue ratio 26.89, and zero shared-load/store hardware
conflicts. These are replay counters, not unprofiled event times. DRAM byte
count returned **unavailable** on this capture; no achieved-memory-bandwidth
percentage is inferred from it.

## QKV: row geometry and transfer width must be separated

The first alternative changed both row size **64 → 16** and transfer width
**32 → 64** (`g4-a2-s2-t4-f1`). It is not evidence for row-size change alone.
The controlled alternative keeps width 32 (`g4-a2-s2-t2-f1`).

The table uses separate baseline medians from each paired cell. A positive
delta below means faster; deltas are median *paired* gains, not a comparison
against a baseline measured in another run.

| Tokens / request rows | Cached: rows16/width32 | Distinct: rows16/width32 | Cached: rows16/width64 | Distinct: rows16/width64 |
| --- | ---: | ---: | ---: | ---: |
| 1 / 1 | +3.1%, inconclusive | +1.9%, inconclusive | +4.7% | +1.0%, inconclusive |
| 2 / 2 | −0.1% | +0.1%, inconclusive | −12.5% | **+16.5%** |
| 8 / 8 | +0.3%, inconclusive | +0.6%, inconclusive | −11.1% | **+16.2%** |
| 16 / 16 | Approximately unchanged | +0.7%, inconclusive | −10.1% | **+15.6%** |
| 17 / 17 | +8.3% | +4.0% | See raw results | +17.9% |
| 32 / 32 | About +20% | +7.6%, inconclusive | −6.7% | +22.2% |
| 64 / 1 | +26.6% | +14.3% | −39.0% | +4.5% |
| 128 / 1 | −25.4% | −14.0% | −75.8% | −30.1% |
| 512 / 1 | −51.7% | −58.6% | −155.5% | −154.2% |

Unmarked positive cells passed the predeclared minimum 3% gain in every pair;
“inconclusive” retains the failed minimum or an outlier, even if its median
improved. Negative cells are regressions, not candidates for promotion.

For the practically important distinct-weight C16 cell:

- Width32 row16: **53.751 → 53.492 µs**, no robust improvement.
- Width64 row16: **53.980 → 45.482 µs**, minimum paired gain **14.76%**.
- Width64 cached cell: **20.495 → 22.571 µs**, a regression.

This is a real cache-sensitive kernel result. It does **not** imply 15.6%
whole-engine throughput gain. Multiple layers, other kernels, mixed prefill,
graph dispatch and host boundaries still contribute.

NCU's width64 comparison found registers **116 → 75**, warp instructions
**915,392 → 739,936**, and issue activity **4.96% → 5.68%**. However, shared-load
conflicts increased **294,912 → 688,128**, and short-scoreboard/active-issue
ratio rose **1.12 → 2.87**. Cold/default-cache replay times were
**101.6 → 52.8 µs**, unlike cached unprofiled timing. These measurements show
why cache state and the whole producer/consumer path matter; they do not prove
one source instruction or bank counter causes the full framework gap.

## Correctness, validation and remaining propagation

MLP complete-chain output and intermediate workspace matched bitwise, with the
existing independent sampled scalar checks. QKV output and positioned KV writes
matched the frozen baseline bitwise for all measured shapes. Every rotated
QKV allocation is also checked before timing. The ingress probe's
`oracle_maxabs=0` is **not an independent full QKV reference**: that oracle is
implemented for attention/postprocess modes, not ingress. The ingress claim
here is baseline differential equality, not new model-quality qualification.

Both row16 QKV variants passed memcheck/leak, racecheck and synccheck at the
65-token partial-tile boundary. The distinct-weight QKV mode separately passed
memcheck with **zero bytes leaked**. The bootstrap MLP pair passed all three
sanitizers. The affected projection package passed **105/105** tests after
removing the rejected production changes. New automation passed targeted
checks; the report parser tests retain an outlier rather than selecting only
the median.

GPU jobs were serialized, user-systemd limited to 8 GiB and no swap, with
bounded runtimes and at least the 32-GiB host reserve. Paired QKV trials recorded
more than 116 GiB MemAvailable before/after each cell. No production service,
global profiler setting or model was changed.

Remaining work is **propagation**, not another claim that all QKV is faster:
the current serving bundle exposes one ingress schedule. A legal general
alternative-set plan must preserve separate prefill/decode or token-domain
choices, content-addressed modules and graph ownership, then select through
the existing resource/measured layer. The small-row width64 option must be
tested end-to-end alongside the larger-row option; the larger-shape losses
above prohibit replacing the sole module. No model-name branch or ad hoc
request-path profiler/JIT should be introduced to accomplish that.

The last measured unchanged serving rates remain **1,438.20 output tok/s**
at 128/32 C16 and **1,053.50 output tok/s** at 512/64 C16. They come from the
[preceding fresh serving control](BENCHMARK_AKO_SHORT_INPUT_2026-10-05.md), not
this kernel-only follow-up. vLLM/SGLang were not rerun in this experiment.

## Reproduction and archive

- Remote root: `/home/wlc004s/lunaflux-ako-small-projection-20261005.VtjZyPFt`.
- Local archive: `/tmp/lunaflux-ako-small-results-20261005.64aPTd/measurement.tar.gz`.
- Verified remote/local SHA-256:
  `dc1652f418b71f457266f35ae48273412bedbddec9e3e6a1bf1110181c405593`.
- `summary.json` retains all MLP parts and all four QKV cache/schedule tables;
  trial directories retain numerical/geometry and raw five-pair output.
- `baseline-gate.csv`, `qkv-counters.csv` retain the hardware capture.
- `FILES.sha256` covers archived source, modules, recipes, commands and logs.
  Untested exporter frontier directories and build caches are excluded.
- Drivers: `ako_small_projection.mbtx`, `ako_small_ingress.mbtx`,
  `ako_ingress_working_set.mbtx`, `ako_ingress_qualification.mbtx`,
  `ako_small_report.mbtx` under `benchmarks/gpu_pipeline/`.
