# Dual-Spark identity rescaling and dependency attribution, 2026-10-05

## Decision

Do not enable guarded unit rescaling. It saves physical instructions but does
not produce a robust timing improvement on either Spark. Remove its temporary
production plan/API and CUDA changes; retain the input archive, AOT artifacts,
paired measurements, sanitizers and PC-level analysis. This is selected-kernel
work, not a new end-to-end or vLLM/SGLang comparison.

## Experiment

The closed online fold initializes output to zero and updates it arithmetically.
An experimental immutable rescale plan permits skipping multiplication by one
for this state, without changing softmax order, BF16 rounding, transfer effects
or denominator updates. CUDA uses a uniform `__any_sync` guard around the output
fragment rescale. It does not apply this rewrite to untrusted bitcast values or
to all floating-point multiplication. Its backend conditions include FTZ off and
the current F32 arithmetic NaN behavior, documented in the
[CUDA 13.0 floating-point contract](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#floating-point-standard).
No fastmath, reassociation, request JIT or model-specific branch was introduced.

Budget: one alternative, three Q2048 cells, five alternating timing pairs per
cell per Spark, 30 GPU-event repetitions per member. Both Sparks timed
concurrently; `.178` then ran memcheck/racecheck/synccheck while `.179` profiled.
CPU build: 8 GiB. GPU processes: 16 GiB, zero swap, 600-second runtime,
32 GiB MemAvailable reserve. Frozen approximate c30322 baseline reused.

| Host | Requests/history | Baseline µs | Guarded µs | Paired time change |
| --- | --- | ---: | ---: | ---: |
| .178 | 1/28672 | 7108.89 | 7186.80 | +1.17% |
| .178 | 2/28672 | 7153.31 | 7214.54 | +0.48% |
| .178 | 2/8192 | 2098.44 | 2072.81 | −1.18% |
| .179 | 1/28672 | 7356.81 | 7371.88 | +1.29% |
| .179 | 2/28672 | 7421.56 | 7421.37 | −0.026% |
| .179 | 2/8192 | 2170.57 | 2145.94 | −1.34% |

Paired changes are medians of candidate/baseline−1, not ratios of table medians.
No cell passes the fixed 3% robust-improvement gate.

## What reached the GPU

Matched Q2048/R2/H28672 selected SASS:

| Metric | Baseline | Guarded |
| --- | ---: | ---: |
| Registers/thread | 234 | 254 |
| Executed warp instructions | 936,583,040 | 875,873,984 |
| FMUL | 123,408,384 | 63,705,088 |
| HMMA | 119,668,736 | 119,668,736 |
| Async LDGSTS | 14,958,592 | 14,958,592 |
| Workgroup barriers | 2,808,832 | 2,808,832 |
| VOTE.ANY | 0 | 934,912 |
| MOV | 100,893,696 | 102,802,432 |
| NOP | 25,242,624 | 28,047,360 |

The compiler removes 48.4% of scalar multiplies and 6.5% of all executed warp
instructions, but the long-history time barely changes. It uses more registers,
but residency remains two CTAs: an occupancy collapse cannot explain the result.
Average warp latency is 6.48→6.90; long-scoreboard share 13.80%→13.82%, wait
share 37.84%→38.95%. Instrumented time slightly favors the candidate
(8.142→8.093 ms), which is not a replacement for the unprofiled paired decision.

## Next action changed by PC evidence

The highest sampled load-dependent site in **both** implementations is the
page-capacity check immediately after a page-table global load:

- Baseline PC `0x325b6e540`: `LDG.E R50, desc[UR12][R48.64]`.
- Next PC `0x325b6e550`: `ISETP.GE.U32.AND ... R50, 0x4000`;
  44,290 not-issued long-scoreboard samples.
- Candidate analogous PC `0x325b82860`: 46,050 samples.

These PCs execute with one active lookup owner per subgroup. The complete
historical stage unrolls eight copy slots, but each slot loads and immediately
validates its page before issuing that slot's vector transfer. That is a serial
metadata dependency. The guard does not change it. HMMA also accounts for
105,258/99,601 not-issued wait samples and 54,554/59,807 math-throttle samples.
This is a multi-part critical path, not a promise that one rewrite fills the
entire framework gap. PC samples are not elapsed cycles or additive gap shares.

The next bounded experiment is to consume the existing portable `PagedTileEpochs`
ownership plan with a batch of immutable page-ID producers before vector
consumers. CUDA subgroup lanes can fetch contiguous IDs together, rather than
eight immediate load/check chains. Page bounds, invalid-page publication,
zero-fill, V-address lifetime and all transfer barriers must remain intact.
This does not conflate earlier no-history captures, where paged fallback did
not execute, with this historical-KV workload.

## Checks and records

All 30 paired observations are bitwise equal to the frozen approximate baseline
and pass the fixed sampled FP64 ceiling. Q129/R2/H128 memcheck, racecheck and
synccheck pass with zero errors/hazards. Experimental source passed 119 affected
native tests with existing warning exclusions 20/79/29/25. It is removed from
production source; no new unused public rescale API remains.

Local root: `/tmp/lunaflux-ako-unit-rescale-20261005.xDdt5ilt`.
`dependency-analysis/report.json` adds ranked PC sites and opcode sample totals
from the original matched capture; it does not reprofile or alter raw data.

- `.178`: `/home/wlc003s/lunaflux-ako-unit-rescale-20261005.h922eOTo/experiment`.
- `.179`: `/home/wlc004s/lunaflux-ako-unit-rescale-20261005.o5MEv9r0/experiment`.
- `.178` sealed archive: `8cac43439b8c70f5e012b09eed65ce655ddeba67b58f3ce62d9088c948ba52bd`.
- `.179` sealed archive: `6a37c45ba93ecc67d79bcf8ae955f9fa60a698ee1600cfd16863e35217723b4f`.

Both downloaded archive hashes match. Complete manifests verify locally; source
input is retained separately as `source-patch.tar.gz`. Neither baseline nor
experimental compiled artifacts are overwritten or relabeled as serving wins.
