# Measured prefill dispatch on both Sparks

## Result and limitation

The measured startup table removes an expensive split-prefill choice without
changing the worker, AOT kernels, model, or scheduler chunk size. Uninstrumented
ABBA serving on Spark .179 reduces median completion time by **22.8% at 16K/C1**
and **39.3% at 32K/C1**. The tested C1 output vectors are repeatable and match
the measured 8K-chunk control.

**C2 is not a validated production winner.** It becomes faster, but identical
requests still produce differing token vectors between repeats. Both the old
and measured 2K-chunk arms exhibit this problem. The measured table is a
benchmark-only artifact, not authorization for deployment or a quality claim.
This work does not fix all attention kernels or prove optimal schedules.

No new vLLM/SGLang measurements were taken in this experiment. Output tok/s
below includes prompt processing; it is not decode-only throughput.

## Machines, scope and parallel work

- .179: GPU `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`; five paired route
  probes, startup binding, and uninstrumented serving.
- .178: GPU `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`; independent small
  frontier/C2 probes and selected-chain Nsight counters, concurrently with
  .179's separate workload. Full serving was not run on .178 in this round.
- Both: GB10, sm121; CUDA 13.0.88, nvcc SHA-256
  `fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
- Qwen3-0.6B BF16; immutable query experiment
  `/home/wlc004s/lunaflux-ako-query-chunks-20261005.PCwRF4kW`.
- The 8K-capacity AOT artifacts are shared by both scheduler arms. The changed
  table binds the existing attention route scope
  `854fff066aebc53d7a7e6de80744a0bf0142fc9f6792449861018d678bf83ae8`.
- GPU work is serialized on each device. Probes/counters use 8 GiB process
  limits; serving retains bounded engine/bridge units, no swap, and a monitored
  32 GiB MemAvailable reserve. Measured .178 probe reserve never fell below
  122,412,884 KiB; .179 serving memory samples are retained per fresh start.

Finite budget: five .179 probe cells, three .178 cells, five .179 serving
starts, two bounded three-kernel counter captures, and a separate dispatch
diagnostic. No new candidate kernel search or reference rebuild.

## Per-device paired probes

The probe uses the corrected runtime geometry: ordinary attention has metadata
capped grid X; split partial and merge have query-tile-capped grid X. The
partial consumes tile metadata, while merge consumes CSR row offsets. Both
split launches and their workspace are included in the timed chain. Five
alternating-order CUDA-event samples follow numerical/KV checks.

| GPU | Queries / rows / history per row | Unsplit median ms | Split pair median ms | Split / unsplit |
| --- | --- | ---: | ---: | ---: |
| .179 | 2048 / 1 / 8192 | 2.368 | 6.759 | 2.85× |
| .179 | 1792 / 1 / 30720 | 7.210 | 21.291 | 2.95× |
| .179 | 2048 / 2 / 12288 | 3.364 | 9.718 | 2.89× |
| .179 | 2048 / 2 / 28672 | 7.771 | 24.606 | 3.17× |
| .179 | 7936 / 1 / 24576 | 27.410 | 67.147 | 2.45× |
| .178 | 64 / 1 / 32768 | 1.388 | 1.574 | 1.13× |
| .178 | 2048 / 2 / 8192 | 2.193 | 6.614 | 3.02× |
| .178 | 2048 / 2 / 28672 | 7.483 | 24.162 | 3.23× |

All eight cells completed with empty probe stderr, unchanged artifact hashes,
unchanged KV, and differential/sampled independent FP64 error within 0.003.
Split and unsplit outputs are not bitwise equal. This does not establish whole
model quality or request-level numerical repeatability.

Only .179 measurements become .179 table records. Stable graph ID 1 is the
complete baseline graph and ID 5 the complete split-prefill graph; neither is
a compiler candidate ID. Actual representative context is
`history + ceil(total_queries / rows)`, not aggregate tokens across requests.
The existing pure startup mapper selects ID 1 in all five measured buckets.
Unmeasured shapes retain existing policy; split-prefill is not globally removed.

## Uninstrumented serving ABBA

Same worker hash
`dd4e8e2b5ddfb66cc6c901cf96b5ca14f12f9138841644c4fc2c4b4480928622`,
same AOT modules, same generated varied token inputs, greedy sampling, 64
output tokens per request. Four starts alternate old/measured/measured/old
at a 2K step/chunk budget, followed by one measured 8K-chunk control start.
Each start excludes one warmup per cell and retains three measured trials,
per-request token IDs, token arrival times, TTFT, TPOT and wave wall time.

| Input / concurrency | Old 2K completion s | Measured 2K completion s | Reduction | Old → measured TTFT s | Measured output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16384 / C1 | 2.622 | 2.024 | 22.8% | 1.669 → 1.061 | 31.63 |
| 32512 / C1 | 7.611 | 4.617 | 39.3% | 6.148 → 3.147 | 13.86 |
| 32512 / C2 | 15.675 | 9.908 | 36.8% | 9.330 → 4.969 | 12.92 |

C2 timing is diagnostic because repeatability fails. The .179 8K control gives
1.979 / 4.504 / 9.345 seconds for these cells respectively. The C1 difference
between measured 2K and 8K chunks is now about 2.2–2.5%, versus the old large
gap: much of the prior apparent chunk-size benefit was a selected-route change.

C1: zero changed output tokens within each arm and between measured 2K and 8K
controls, across both fresh starts and the retained warmup/measured vectors.
C2: old repeats differ by up to 39 positions; measured repeats differ by up to
41 positions. The 8K control's four repeats are stable in this capture, but
its row-zero vector differs from the first measured-2K vector by 41 positions.
Do not conflate these request-level failures with the synthetic kernel tolerance
gate. Next investigation must trace real C2/mixed execution and first divergent
logits, rather than simply replacing more kernels.

## .178 hardware counters

Exact probe: 2048 queries, two rows, history 28672 per row. Two isolated admin
captures preserve the preceding permission-denied non-admin capture. Basic
and detailed captures are separate; profiled timings are not ABBA serving times.

| Detailed selected invocation | Unsplit | Split partial | Merge |
| --- | ---: | ---: | ---: |
| Profiled duration ms | 7.301 | 23.324 | 0.208 |
| Warp instructions | 1,125,435,264 | 1,270,290,176 | 13,479,936 |
| Tensor active / elapsed | 54.54% | 16.98% | 0% |
| Long-scoreboard stall ratio | 22.55% | 55.89% | 86.82% |
| Barrier stall ratio | 2.19% | 2.71% | 0.07% |
| Registers / thread | 235 | 255 | 34 |
| Allocated shared memory KiB / block | 50.304 | 83.072 | 1.024 |
| Shared-memory residency limit / SM | 2 | 1 | 32 |

Basic capture's active-warps ratio is 16.24% versus 8.11% for unsplit/partial;
its profiled durations are 8.046 / 23.908 / 0.202 ms. The split pair executes
about 14.1% more warp instructions, not three times as many. Its partial has
roughly half the active warps and a much larger load-dependency ratio. The merge
is under 1% of the chain duration: removing the merge alone cannot recover the
gap. These measurements support a resource/latency-hiding regression in the
partial. They do not apportion the full slowdown uniquely between layout,
global-load dependencies and occupancy, or establish equal DRAM traffic.
Stall ratios have their own denominator and are not additive wall-time shares.

## Compiler/runtime boundary and reproducibility

No production kernel, model builder, scheduler or compiler IR is rewritten.
The change uses the existing chain:

`complete AOT graphs → exact-shape offline observations → pure startup owner mapping → immutable token-step dispatch`.

There is no request-path JIT, profiling, filesystem validation, cryptography,
new allocation, model-name special case or globally forced schedule. The
route diagnostic is a disposable worker, separate from timing binaries.
No sanitizer result is inferred from counters; unchanged kernels retain prior
qualification, while this round adds paired correctness checks.

Local tests: each new automation helper, route installer, attention tuning
parser (3 tests), and measured runtime owner mapping (5 tests) passed. The latter
package tests require existing toolchain-migration warning exclusions; the
unexcluded command first fails on pre-existing implicit-promotion warnings.
No unrelated dirty-tree changes were edited or staged.

Raw roots:

- .179: `/home/wlc004s/lunaflux-ako-measured-routes-20261005.XrKdGE3t`.
- .178: `/home/wlc003s/lunaflux-ako-measured-routes-20261005.gK1oe7nW`.
- Local transfer: `/tmp/lunaflux-measured-routes-20261005.kQsBRzBn`.

.178 archive SHA-256:
`f546df06fae48f4188fdfcbcd6f9bd274bcb126a3f654316d115bfc3cf164701`.
Its local archive hash and all 66 manifest entries were verified.

.179 archive SHA-256:
`9358c8f5686ee232f627b80b35e5e270ad83f5d046cc78ec79184cc8b811aeba`.
Its local archive hash and all 722 manifest entries were verified. Both
archives were downloaded without overwriting and extracted under the local
transfer directory as `spark178/` and `spark179/`.

The separate .179 dispatch replay records 12 `kind=17` markers at the measured
owner-table early return: four selections of bucket 245 (one row, 2048 queries,
context bucket 16384), and eight selections of bucket 246 (one row, 2048 queries,
context bucket 32768). All select owner 27, the unsplit baseline graph. No
forced-route flag was supplied. Unmeasured buckets retain their existing
selection policy. The trace is archived under
`lunaflux-ako-chunk-trace-selected/selected-owners.txt`; traced execution is not
used for the serving timing results above.
