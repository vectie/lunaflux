# Major-contributor repair: bounded MLP down

## Selection of the problem

Fresh full-serving attribution on the 48-SM GB10, Qwen3-0.6B BF16,
4096 input / 256 output tokens, C16, showed the following exclusive totals.
These are diagnostic trace times, not ordinary benchmark latency:

| Category | LunaFlux ms | vLLM ms | SGLang ms | LunaFlux excess vs vLLM ms |
| --- | ---: | ---: | ---: | ---: |
| Projection / post-ops | 2637.114 | 2217.152 | 2330.490 | 419.962 |
| Attention | 9107.500 | 8870.666 | 8936.218 | 236.834 |
| No GPU activity | 380.718 | 207.265 | 35.645 | 173.453 |
| Normalization | 173.712 | 111.180 | 148.324 | 62.532 |
| Vocabulary head | 393.164 | 391.795 | 359.918 | 1.369 |

Projection/post-ops is the largest *GPU category excess*, not the largest
absolute category. Attention remains the dominant absolute cost. Relative to
SGLang, no-GPU time is larger than projection excess. It is not automatically
CPU scheduling time: copies, overlap and profiler overhead must be separated.

Source-ordered attribution of vLLM's complete decoder cycles identified down
as the largest individual projection contributor: approximately 558 ms here,
versus 377 ms in vLLM. Attribution requires attention and SiLU anchors and
28 × 4 projection calls; two incomplete tail cycles remain unresolved rather
than being assigned from grid dimensions. The request is identical, but batch
and history step vectors are not identical, so this is a prioritization ledger,
not an exact isolated-operation speed ratio.

## Controlled repair

The selected 8/16-row down schedule launched 16 CTAs of four warps on 48 SMs.
The offline alternative uses 64 single-warp CTAs and a 128-wide reduction
transfer window instead of 64. This exposes more independent work and halves
the reduction-window iterations. It does not change ordered dot products,
BF16 conversions, the gate/up producer or the large-row schedule.

The existing generic `ProjectionFoldChoice` expresses this as a bounded
`IntermediateFold`; the immutable compiler refines layout/effects and CUDA
lowering spells the result. No Qwen branch, new runtime validation, online
JIT, cryptography or tuning is added to the token path.

| Exact AOT chain, rotating 28 down matrices | Before µs | After µs |
| --- | ---: | ---: |
| 8-row down | 48.480 | 34.736 |
| 8-row gate/up → down | 67.117 | 53.533 |
| 16-row down | 48.482 | 35.619 |
| 16-row gate/up → down | 67.379 | 54.603 |
| 2048-token down | 227.826 | 227.810 |

Five alternating event trials per part, median shown. Producer timing is
unchanged within noise. The down weight working set is 176,160,768 bytes;
the timed interval contains no host weight upload. These are not full-serving
throughput numbers. Bitwise whole-chain equality and the scalar checker passed
at 8, 16 and 2048 tokens. Compute Sanitizer memcheck reported zero errors and
zero leaked bytes on the exact compiled 16-row modules.

### Hardware replay

| Counter, exact 16-row modules | Before | After |
| --- | ---: | ---: |
| CTAs / threads per CTA | 16 / 128 | 64 / 32 |
| Registers per thread | 72 | 130 |
| Warp instructions | 315,456 | 423,872 |
| Tensor activity, % of elapsed peak | 0.886 | 2.047 |
| SM issue activity, % of elapsed peak | 0.711 | 2.206 |
| Long-scoreboard stalls per issue-active ratio | 15.064 | 5.426 |
| Barrier stalls per issue-active ratio | 3.637 | 0.034 |
| Short-scoreboard stalls per issue-active ratio | 2.294 | 1.615 |
| Active occupancy, % | 10.067 | 2.450 |
| Replay duration, µs | 93.248 | 41.184 |

Stall ratios are **not percentages**. Replay uses cold-cache control and is
not interchangeable with the rotating-weight event medians. The win despite
more instructions/registers and lower occupancy supports the work-distribution
and dependency diagnosis; it does not prove each change's independent share.
DRAM-byte metrics were unavailable in this capture and are not fabricated.

## Propagation and rejected experiments

The normal model exporter reproduces the measured source digest
`c6b7f77a4f6c5e6bcd47d0f2c7a24ffcfa947c8d40d36a5217dc1971d2b96397`.
All 28 MLP sources change; other operation sources remain byte-identical.
Normal deterministic compilation and release binding produce module
`bf10e22f8213f89a8a99f4530e1928051111f670101b84b0785201918ab949fe`.
The scoped snapshot is
`benchmarks/gpu_pipeline/profiles/gb10-sm121-20261004.fold-v4`;
it is an explicit measured profile, not a universal device recommendation.
Its MLP record objective is the measured 2048-token whole module versus the
untuned default, not a fabricated small-row number substituted for that objective.
Small-row qualification supplies the incremental comparison to the previously
selected module. Dense records retain their previous measured provenance.

The row-retirement experiment reduced large-down registers 218 → 180 but
increased rotating-weight down time by about 2.6%. It was removed from the
local production compiler; its captured source and measurements remain remote.
Likewise, smaller large-row tiles did not consistently improve the complete
chain. They are not installed as production winners.

Early harness runs omitted gate/up's required 16 KiB dynamic shared allocation.
Those runs are invalid and excluded. Corrected probes record static and dynamic
shared memory separately. Failed preparations and their logs were preserved;
continuations use new log names and do not overwrite captures.

## Campaign locations

- Fresh attribution: `/home/wlc004s/lunaflux-major-step-20261004.2JUYM84Q`.
- Valid small-row search: `/home/wlc004s/lunaflux-major-down-small-valid-20261004.dUKY3iEA`.
- Rejected row lifetime: `/home/wlc004s/lunaflux-major-down-valid-20261004.WYScnFgy`.
- Exact AOT package and counters: `/home/wlc004s/lunaflux-major-down-serving-v2-20261004.vhI1rFLD`.
- Ordinary fresh-start comparison: `/home/wlc004s/lunaflux-major-down-bench-20261004.BxBWDY2e`.

GPU workloads are serialized. Preparation/qualification uses 8 GiB/no swap;
serving uses bounded 64 GiB/no swap and a monitored 32 GiB host-available reserve.
## Ordinary fresh-start serving result

Three Latin-square fresh starts per framework, identical synthetic token
inputs, greedy generation, the same pinned BF16 model and C16. No profiler is
running in these trials. The long-input outputs agree across every framework
and round. The existing short-input 382/624 divergence occurs again, including
one within-LunaFlux repeatability difference; short output equivalence is not
claimed. Three trials do not establish statistical significance.

| Input / output / C | Previous LunaFlux ms | New LunaFlux ms | New tok/s | vLLM ms | SGLang ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 32 / 16 | 367 | 354 | 1446.33 | 323 | 326 |
| 4096 / 64 / 16 | 4564 | 4552 | 224.96 | 4207 | 4212 |
| 4096 / 256 / 16 | 12734 | 12611 | 324.80 | 11842 | 11999 |

The long-256 median completion time improves **0.97%** relative to the previous
campaign, not 26.5%. The current remaining time gaps are **6.49% vs vLLM** and
**5.10% vs SGLang**; long-64 gaps are **8.20% / 8.07%**. This repairs a measured
kernel/selection problem without closing the entire framework gap. The old and
new framework medians come from successive campaigns, not a formal paired
runtime A/B significance test. Long-256 trials are 12,566 / 12,695 / 12,611 ms;
the previous median is 12,734 ms.

The minimum monitored `MemAvailable` is 55,597,024 KiB, above the 32 GiB reserve.
All runtime workers drain and close without stderr. Local native checks and
the full suite pass: **4,362 / 4,362**, using the existing MoonBit migration
warning exclusions `-79-20-29-25-92-14`. Affected compiler/AOT tests pass
208 / 208. No claim of production deployment or complete optimization is made.

## Selected full-serving trace

The subsequent trace at
`/home/wlc004s/lunaflux-major-down-trace-20261004.bUbmb4lY` completes successfully.
Restricting attribution to the timed client window, rather than including
warmup, confirms 6,916 rows16-down calls use **64 CTAs × 32 threads**, 130
registers/thread and 16 KiB shared memory. Their summed duration is
**242.395 ms**, versus **336.744 ms** for the same call count in the preceding
selected trace: a 28.0% reduction. The rows8-down entry also uses 64 × 32;
the primary large-down entry remains 256 × 128, 218 registers and 48 KiB.
All 59,248 observed kernel calls are graph kernel calls. This proves selection
and execution propagation, not a profiler-free latency result.

Exclusive projection/post-op activity changes from 2,637.114 to 2,538.658 ms.
Attention is 9,123.622 ms and no-observed-GPU activity is 406.286 ms in the new
trace. Compared with the previously captured references, projection excess is
still approximately 322 ms versus vLLM, attention excess 253 ms, and idle
excess 199 ms. Against SGLang, idle excess is approximately 371 ms and exceeds
projection excess of 208 ms. These separate captures are a remaining-work
prioritization ledger, not an exact causal decomposition of ordinary latency.
The down repair is therefore complete for the qualified bounded schedules;
the entire framework performance gap is **not** complete.

## Downloaded measurements

Compact archives exclude deployment configuration and credentials. Each
includes a per-file SHA-256 manifest and is created without overwrite.
Local download directory: `/tmp/lunaflux-major-down-20261004.5aM0Zs5b`.

- `aot.tar.gz`: `c68021c21d6029a882d3f5fe47e789928000132eb258f8e84ed6212e7ad5e861`.
- `benchmark.tar.gz`: `dc7764abeae0b366605c5534196ab0401968ab1fd11402be7e68dc5867e15531`.
- `trace.tar.gz`: `227592e626c7781eefc4e8261bac4ab0cf2a63e982172c7df2754d3936271679`.

The bounded profile, AOT propagation regression and qualification tooling are
committed as `e1657059`. The measured profile remains explicitly opt-in and
device-scoped; it is not installed as a universal static default.
