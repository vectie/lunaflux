# Mixed attention calibration repair

Follow-up: [exact mixed-work attribution](BENCHMARK_EXACT_MIXED_WORK_2026-10-07.md)
matches all 96 LunaFlux/vLLM logical steps, corrects reference phase
classification, and separates the remaining attention/projection gaps. Its
score-storage experiment does not change the serving results below.

The 8K input, 64 output, C8 serving cliff includes a reproducible selection
bug. Reusing immutable prefill records skipped mixed-chain calibration, leaving
unmeasured mixed buckets on an expensive partitioned-prefill heuristic. The
completed repair reduces complete-wave time by 27.27% and raises aggregate
output throughput by 37.49%, without changing any of the nine executable
kernel modules. The first partial repair gained 15.74%; covering the smaller
mixed buckets removes the remaining main split-prefill fallback calls.
It does not establish performance or numerical parity with vLLM and SGLang.

## Source cause and repair

`benchmarks/gpu_pipeline/measure_decode_routes.mbtx` previously iterated no
prefill or mixed shapes whenever prefill measurements were reused. Prefill
records do not measure the independently sized decode companion of a mixed
graph. The frozen route table contained no mixed R8/Q2048 records.
`engine/device_step/graph_bucket.mbt` correctly gives measured startup owners
priority, but an absent record retains the heuristic. Consequently the new
kernel alternatives existed without being selected for this workload.

The repair separates reusable prefill coverage from mandatory mixed coverage.
It also makes the number of prefill requests explicit in the offline probe.
The former default, R−1 prefill requests and one decode request, does not cover
the scheduler's one-prefill-request, many-decode-request physical domain.
Calibration now checks concentrated and distributed extrema, omits infeasible
distributions, and records a conservative maximum of five-sample medians for
each candidate. These are complete attention chains, not prefill-only timings.
The underlying five-sample measurements and exact distributions are retained.

Selected recipe/module pairs take precedence over older staged module copies.
This prevents calibration from combining a newly selected symbol with a stale
binary when both stages are present. No benchmark records are fabricated.

The production architecture remains unchanged: pure shape/phase policy,
offline measured records, startup graph ownership and constant-time runtime
dispatch. This adds no model-name branch, request-path tuning, JIT, filesystem
check or profiler dependency. The general calibrator covers row buckets
1, 2, 4, 8 and 16; the focused experiments repair observed mixed buckets using
existing qualified alternatives.

## First alternating serving comparison

Four fresh LunaFlux starts run baseline, candidate, candidate, baseline.
Each start uses one warmup and three measured waves per cell, producing six
measured waves per side. Input token-ID vectors, output count, timing boundary,
model, runtime, worker and all nine kernel modules are unchanged. Only startup
route records change. Throughput counts output tokens and includes prefill.

| Input / output / C | Baseline wall ms | Candidate wall ms | Completion time reduction | Baseline / candidate tok/s |
| --- | ---: | ---: | ---: | ---: |
| 8192 / 64 / 8 | 6837.5 | 5761.5 | 15.74% | 74.88 / 88.87 |
| 4096 / 64 / 16 | 4561 | 4567 | −0.13% | 224.51 / 224.22 |
| 128 / 256 / 1 | 1710.5 | 1699.5 | 0.64% | 149.66 / 150.63 |
| 32512 / 64 / 2 | 7565.5 | 7567 | −0.02% | 16.92 / 16.92 |

The 8K paired halves improve by 15.30% and 16.28%. Median request TTFT falls
from 2500 to 2334 ms; median mean TPOT falls from 64.98 to 50.44 ms. The
controls do not establish a meaningful speed change. Minimum host available
memory exceeds 99 GiB, against the enforced 32 GiB reserve.

Within-engine complete output vectors still vary at 8K/C8: 14/40 baseline and
11/40 candidate comparisons differ from their first measured row vector.
16/48 candidate vectors differ from the corresponding baseline row vector.
All inputs and output counts match. These timing results are not a quality
qualification; the existing numerical/repeatability question remains open.

## Executed route verification

Separate Nsight Systems captures use the exact client workload and selected
bundle. Their profiled times are not substituted into the throughput table.

| 8K C8 observed work | Baseline | First repair | Complete repair |
| --- | ---: | ---: | ---: |
| Complete profiled wave ms | 6807 | 5754 | 4969 |
| GPU activity union ms | 6685.63 | 5633.88 | 4843.70 |
| No recorded GPU activity ms | 122.65 | 121.12 | 127.41 |
| Graph steps | 96 | 96 | 96 |
| Main partitioned prefill partial calls | 784 | 336 | 0 |
| Main partitioned prefill partial plus merge ms | 2677.38 | 1148.33 | 0 |
| Ordinary wide prefill calls | 112 | 560 | 896 |
| Ordinary wide prefill ms | 112.81 | 586.95 | 959.56 |

The gain reaches real serving, rather than only a probe or an unselected AOT
file. Nearly unchanged GPU-inactive time rules out a large host-bubble repair
as its explanation. The first repair leaves 336 expensive partial calls in
smaller mixed buckets reached while the batch grows. The complete repair
calibrates R2/R4/R8 with Q2048 through their bounded context limits and removes
those main two-partition calls. A separate eight-partition tail remains:
28 partial calls, 17.88 ms. That is not the original 2.68-second cliff and is
not relabeled as zero total partitioned-attention work. Total kernel count
remains 21,440; the repair changes work distribution, not graph step count.

## Complete alternating serving comparison

A second independent four-start ABBA campaign compares the original baseline
with the complete R2/R4/R8 repair using the same four controls and six measured
waves per side. No kernels, model weights, capacity, accuracy thresholds or
reference commands change.

| Input / output / C | Baseline wall ms | Candidate wall ms | Completion time reduction | Baseline / candidate tok/s |
| --- | ---: | ---: | ---: | ---: |
| 8192 / 64 / 8 | 6836 | 4972 | 27.27% | 74.90 / 102.98 |
| 4096 / 64 / 16 | 4564.5 | 4567 | −0.05% | 224.34 / 224.22 |
| 128 / 256 / 1 | 1707 | 1704 | 0.18% | 149.97 / 150.24 |
| 32512 / 64 / 2 | 7575 | 7587.5 | −0.17% | 16.90 / 16.87 |

The 8K paired halves improve by 27.13% and 27.32%. Median request TTFT drops
from 2503.5 to 1528.5 ms, a 38.95% reduction. Median mean TPOT drops from 64.94
to 50.82 ms, a 21.75% reduction. Small changes in the controls are not evidence
of a meaningful performance regression or gain. Minimum observed host
available memory is 99.46 GiB baseline and 99.67 GiB candidate.

Within-engine changed complete vectors are 11/40 baseline and 9/40 candidate
at 8K/C8, with 15/48 candidate vectors differing from their baseline row.
At 4K/C16 the corresponding counts are 2/80, 11/80 and 9/96. Short-input and
32K/C2 vectors agree within and across both sides. Performance-control stability
does not mean all numerical-control vectors agree; production promotion
remains outside this repair.

The matched baseline trace also includes vLLM and SGLang. Their 8K profiled
wave times are 4503 and 4566 ms. SGLang uses larger prefill chunks, so counts of
individual attention invocations are not interchangeable. The earlier fresh
unprofiled reference medians are 4513.5 and 4616.5 ms, documented in
`docs/BENCHMARK_SHORT_LONG_CONTEXT_2026-10-07.md`; they are not new alternating
reference starts from the repair campaign.

Relative to those earlier unprofiled reference medians, 4972 ms still takes
10.16% longer than vLLM and 7.70% longer than SGLang. Their corresponding
8K/C8 throughput is 113.44 and 110.91 tok/s, versus the repaired 102.98.
The selection cliff is repaired, not every remaining implementation gap.
The final trace still contains 1433.27 ms of the ordinary C8 decode kernel
and 959.56 ms of wide prefill, plus mixed decode, projection and other work.
These groups are not a matched instruction-level attribution of the entire
remaining reference gap. No new claim about bank conflicts, load stalls or
instruction issue follows from timestamp traces alone.

## Validation and scope

Six focused calibrator tests pass warning-denied native checks, including
prefill reuse, context/arena bounds and mixed distribution coverage. The C++
geometry test passes with warnings treated as errors. The new diagnostic probe
compiles on sm121 with the pinned CUDA 13.0 compiler. Complete-chain probes
check all output elements against their paired baseline with maximum absolute
error 0.003, an independent sampled FP64 attention oracle with the same bound,
finite outputs and unchanged KV contents. No tolerance is relaxed.

The concentrated mixed-domain memcheck passes with zero errors. Production
kernels and native ABI are unchanged; a dirty-tree whole-runtime rebuild is
not presented as this experiment's source. GPU work is serialized. Serving
has a 64 GiB cgroup cap, bridge 2 GiB, controller 8 GiB, zero additional swap,
and the 32 GiB available-memory reserve. Every serving arm drains and closes
its child with exit zero and empty runtime stderr. Production is unchanged.

The source fixes are committed as `af4adb2d` and `54f03b50` on `parallel`.
The runtime and model identities are the frozen identities in the preceding
short/long report; archives also retain their independently captured hashes.

## Evidence locations

First route repair:
`/home/wlc004s/lunaflux-batch-cliff-fix-20261007.fPum5Ax6`.
First alternating comparison:
`/home/wlc004s/lunaflux-batch-cliff-abba-20261007.GC1SlmUw`.
First selected-route trace:
`/home/wlc004s/lunaflux-batch-cliff-selected-trace-20261007.rpdcW7L7`.
Complete route repair:
`/home/wlc004s/lunaflux-batch-cliff-fix-20261007.sx0E1lbt`.
Complete alternating comparison:
`/home/wlc004s/lunaflux-batch-cliff-abba-20261007.xvFHOL9t`.
Complete selected-route trace:
`/home/wlc004s/lunaflux-batch-cliff-selected-trace-20261007.aFLlBSBF`.
Matched baseline trace:
`/home/wlc004s/lunaflux-batch-cliff-trace-20261007.R5mZygdX`.

An earlier failed preparation at
`/home/wlc004s/lunaflux-batch-cliff-fix-20261007.d8URrZV4` is preserved.
Its module/symbol pairing failed before measurement; it is not evidence of a
production kernel failure. Corrected attempts use new directories and do not
overwrite the failed logs or artifacts.

All seven completed experiment archives were downloaded without overwrite to
`/tmp/lunaflux-batch-cliff-evidence-20261007.UJLUZMJq`.
Local archive hashes and every archived file-manifest entry verify.
Build/dependency trees and materialized deployments are excluded; executable
identities, selected bundles, calibration modules/recipes, owned source
snapshots, request vectors and traces are retained. No original is deleted.

| Archive suffix | SHA-256 | Verified archive members |
| --- | --- | ---: |
| abba GC1SlmUw | `0e26f5b84f0f41129401648bc7ceb99d9c7e14c707d76849cf5330846b32c4f4` | 1492 |
| selected trace rpdcW7L7 | `370e570b995be70a10c7941ed8dba47b71bfef8986b6de9d2747d76537471a3f` | 200 |
| baseline trace R5mZygdX | `95593793e4e966686d89e83bc692f05a25384a91c8a3eefe6cc24a331271db11` | 602 |
| fix fPum5Ax6 | `9193a205bbe1b8eb3696504028fae78da537c8fb78603b16e00b535144e284f6` | 157 |
| abba xvFHOL9t | `c4744fb59de421296995ba7b16923d75222b95a5adcea07d70bbbfea6a5d1493` | 1492 |
| selected trace aFLlBSBF | `bea4bfec99c13c94869cbf9ddca684830b83420109096b8af38f7779b3efa670` | 200 |
| fix sx0E1lbt | `892a928ab01a1dd635746e47dd5c2868f0fe8cd5ab975361cbfe94f648e82403` | 285 |

The complete repair archive is
`/tmp/lunaflux-batch-cliff-evidence-20261007.UJLUZMJq/lunaflux-batch-cliff-fix-20261007.sx0E1lbt.verified.tar.gz`.
The full alternating comparison is
`/tmp/lunaflux-batch-cliff-evidence-20261007.UJLUZMJq/lunaflux-batch-cliff-abba-20261007.xvFHOL9t/comparison.json`.
