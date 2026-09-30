# Measured compiler-to-serving integration

This follows the five implementation/selection recommendations in the Spark
source review. It is a general functional planning and CUDA-lowering change,
exercised on Qwen3-0.6B BF16. It is not a claim that every new schedule wins.

## Implementation

| Recommendation | Implemented join |
| --- | --- |
| Parallel blockwise decode | Explicit F32 probability law, partial/merge AOT symbols, exact eight-partition workspace plan, V8 runtime admission and captured graph alternative. The existing numerical family remains eligible. |
| Joint geometry/resource planning | Immutable row/head/K-tile/stage/fragment frontier; shared-memory limits derived from device properties, bounded by the private ABI; configurable compiler register ceiling included in tuning identity. |
| Full versus separated ingress | Both full QKV fusion and independent projection plus QKNorm/RoPE/KV-write epilogue compile and bind. Explicit evaluation is distinct from `--ingress-pair` measured whole-chain selection. |
| Shape-dependent attention dispatch | Measured batch/query/history buckets bind prepared complete graph owners at startup. Ordinary, wide prefill and partitioned alternatives remain distinct; token steps use the prepared lookup. |
| Machine feedback | Version-3 records carry actual instruction/local-sector observations and explicit unknown barrier counts. The typed selector consumes resource and machine constraints before measured latency. |

There is no request-path compilation, measurement, filesystem validation or
cryptography. Compiler plans remain immutable; scratch reuse, copy readiness,
reader retirement and merge writes are explicit effects. NVIDIA implementation
details remain in the CUDA backend, not model semantics or request scheduling.

The bounded benchmark search is not an exhaustive autotuner for every device,
model or workload. Unmeasured cells retain a legal fallback. The V8 route is
available and tested, but is not forced into the default just to count it as
implemented.

## Bugs repaired during integration

- Runtime startup reduced an AOT eight-way partition to four at a shorter
  context, disagreeing with generated source/workspace. Explicit AOT split
  counts now survive binding; the legacy heuristic remains separate.
- The builder appended old V7 split source to the complete V8 module. Modules
  now retain their numerical family rather than mixing incompatible entries.
- Graph measurements used alternate ID 3 without the required matching ID-1
  baseline. Records now include an actual baseline for every measured cell.
- Register compiler flags were missing from ingress tuning identity. Different
  ceilings now invalidate otherwise source-identical measurements correctly.
- Calibration hardcoded a 128-register record budget even during a 255-register
  experiment. Records now preserve the actual budget.
- Measured attention selection collapsed the frontier to one candidate, making
  explicit evaluation of another legal candidate impossible. Evaluation now
  retains alternatives without inventing observations; regression coverage
  checks that the measured winner is unchanged.
- The opposite decode phase silently used the static winner although real
  decode measurements existed. The serving selection helper now supplies that
  table explicitly.
- The independent epilogue probe confused `symbol` with `function_symbol` and
  ordinary row offsets with attention metadata. Both probe contracts are fixed.
- Fresh preparation omitted the probe geometry header. Preparation now copies
  the source and header from the same archive.
- Pure-prefill calibration omitted mixed-phase slots and intervening history
  buckets. Those slots silently took the slower wide heuristic. The offline
  probe now represents mixed descriptor counts and measures each context bucket
  separately; a regression confirms that pure-prefill records do not cover
  mixed slots. No production token-step scan or online timing was added.

## Hardware-counter ablation

Spark GB10, 48 SMs, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`,
CUDA 13.0.88. Query2048/rows8/history2048; five paired unprofiled samples.
The 128- and 255-register campaigns retain separate compiler identities.
Local sector traffic is **not physical DRAM bytes**. Profiled durations are not
substituted for the unprofiled selection objective.

| Ingress schedule | 128 ceiling µs | 255 ceiling µs | Local-sector bytes, 128 → 255 |
| --- | ---: | ---: | ---: |
| Row64 / K32 / stage2 / head1 | 482.370 | 492.625 | 0 → 0 |
| Row64 / K32 / stage3 / head1 | 616.161 | 615.959 | 0 → 0 |
| Row128 / K32 / stage2 / head1 | 2502.731 | 893.513 | 3,579,838,464 → 0 |
| Row64 / K32 / stage2 / head2 | 2016.999 | 835.418 | 2,509,766,656 → 0 |
| Row64 / K64 / stage2 / head2 | 2482.209 | 904.449 | 3,594,780,672 → 0 |

The wider schedules really were spilling. Removing that traffic produces
large isolated gains, but does not beat the row64 winner. All twelve profiled
255-ceiling finalists have zero measured local-sector traffic in this cell.

| 255-ceiling schedule | Registers | Resident CTAs/SM | Warp instructions | Tensor active % | Long-scoreboard % | Barrier-stall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Row64, head1, stage2 | 118 | 3 | 67,461,120 | 25.63 | 19.19 | 4.53 |
| Row128, head1, stage2 | 198 | 1 | 53,737,472 | 14.98 | 16.03 | 2.54 |
| Row64, head2, stage2 | 217 | 1 | 58,865,664 | 16.02 | 15.77 | 2.26 |

Less instruction work and lower stall percentages do not imply a faster
kernel: the wider schedules have substantially less active tensor execution
and one resident block rather than three. This supports a remaining
resource/work-distribution limitation after spill removal, not a claim that
occupancy alone explains every cycle. Three/four stages also lower some waits
without improving the two-stage winner. Barrier instruction counts are
unavailable and stored as unknown, not fabricated from stall percentages.

## Integrated blockwise/full-chain experiment

Source archives and complete logs are retained under:

- v3: `/home/wlc004s/lunaflux-integrated-v3-20260930.fqshBJiL`
- v5: `/home/wlc004s/lunaflux-integrated-v5-20260930.cQ36xBWD`
- v6: `/home/wlc004s/lunaflux-integrated-v6-20260930.INoLaObL`

The v3 full and separated runs use the same source, model, geometry and
ordinary/blockwise decode families. Both pass packaged-kernel sanitizers,
normal shutdown and empty runtime stderr. At input4096/output256/C16:

| Route | Median completion ms |
| --- | ---: |
| Prior fastest LunaFlux | 14,083 |
| V8 full ingress + measured decode routes | 14,507 |
| V8 separated projection/epilogue + measured decode routes | 14,594 |
| Pinned vLLM, preceding same-day measurement | 11,840 |
| Pinned SGLang, preceding same-day measurement | 11,980 |

The 0.6% full/separated difference is small; it is not convincing evidence
that maximum fusion always wins. The V8 experiment regresses versus the prior
package and cannot be promoted as a speedup.

V3 partial+merge improves ordinary blockwise C1/history4095 from approximately
726 to 104 µs, but production already had a partitioned route at small batch.
At C16/history4095 it is approximately 1294 µs; the measured existing c441
ordinary kernel is approximately 1263 µs. At C8 the existing c441 is also
faster than the blockwise partition alternative. Comparing only with unsplit
blockwise decode would therefore exaggerate a serving gain.

The final v6 run consumes actual v3 machine feedback, retains independent
ordinary/wide prefill modules, and measures 45 prefill buckets before binding
their graphs. The first v6 export exposed the opposite-phase selection issue
and is retained separately; the corrected export selects measured c441.

### Actual serving exposed a second selection defect

The initial corrected export completed input4096/output256/C16 in **14,797 ms**.
Its serving trace revealed 504 launches of c2001, totaling **1,009.582 ms** in
that diagnostic capture. The earlier paired normal-prefill counters had not
captured this variant. Pure and mixed phases occupy different immutable graph
slots; sparse pure-prefill measurements were not sufficient coverage.

An exact mixed query2048/rows16/history2048 counter pair quantified the cause:

| Counter | c322 | c2001 |
| --- | ---: | ---: |
| Profiled replay duration µs | 1,102.528 | 3,714.688 |
| CTAs | 1,008 | 1,520 |
| Registers/thread | 230 | 255 |
| Executed warp instructions | 109,589,824 | 142,945,488 |
| Local-load sectors | 0 | 2,049,472 |
| Local-store sectors | 0 | 1,534,848 |
| Tensor active % | 33.24 | 10.01 |
| Long-scoreboard stall % | 23.96 | 16.47 |
| Barrier stall % | 17.22 | 4.87 |

The wide kernel executes 30.4% more instructions and approximately **114.7 MB
of local-sector traffic** despite a 255-register ceiling. Its lower dependency
and barrier percentages are not a speedup: replay takes 3.37 times as long.
This is a concrete spill/resource/work-distribution defect in this alternative,
not an absence of functional IR. The selector must reject it for these shapes;
expanding a legal KV tile is not proof that its lowering is profitable.
GB10 DRAM-byte counters were unavailable in this capture and are not invented.

Calibration now covers **183 pure/mixed prefill cells**, with five paired
unprofiled samples per cell, including every context bucket between its current
query extent and 8192. These are representative shapes, not an exhaustive
certification of every distribution inside each bucket. Mixed batches also
pass independent attention correctness and memcheck/racecheck/synccheck.
The final serving trace confirms c2001 is absent: the same 952 principal
prefill calls use c322, totaling 695.124 ms instead of the previous combined
1415.219 ms. This is a profiled diagnostic comparison, not the unprofiled
serving throughput result.

## Final matched serving comparison

Qwen3-0.6B BF16 on the same GB10; exact input token IDs, greedy, ignore-EOS,
prefix reuse disabled. Three measured trials after warmup. Pinned NVIDIA26.01
vLLM and SGLang containers were rerun serially after the initial v6 serving.
The final route-only retest uses the same source kernels, with the expanded
startup graph table. Lower completion time is better.

| Input/output | C | Previous fastest Luna ms | Final Luna ms | Fresh vLLM ms | Fresh SGLang ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 246 | 245 | 293 | 287 |
| 128/32 | 8 | 321 | 327 | 274 | 283 |
| 128/32 | 16 | 383 | 388 | 320 | 329 |
| 4096/64 | 1 | 738 | 741 | 767 | 788 |
| 4096/64 | 8 | 2,752 | 2,766 | 2,270 | 2,340 |
| 4096/64 | 16 | 5,177 | 5,175 | 4,201 | 4,244 |
| 4096/256 | 1 | 2,529 | 2,536 | 2,699 | 2,761 |
| 4096/256 | 8 | 7,625 | 7,633 | 6,650 | 6,875 |
| 4096/256 | 16 | 14,083 | 14,081 | 11,861 | 12,003 |

The route repair removes the regression, not the remaining framework gap.
Final long-C16 throughput is **290.89 token/s**, versus **345.33/341.25**;
completion time remains **18.7%/17.3% higher**. Long-C16 TTFT p50/p95 is
1495/2630 ms, versus 1147/2259 and 970/1707. ITL p50/p95 is 46/48 ms,
versus 39/61 and 40/42. Short/single-request results have different tradeoffs;
there is no universal speedup.

All 150 long-input output sequences match each reference exactly. Across all
225 sequences, exact agreement is 220 with previous LunaFlux, 219 with vLLM,
and 221 with SGLang; all 225 first tokens agree. Remaining differences occur
on the short-input cases. This is not a bitwise identity or broad quality claim.

Normal selected kernel counter pairs versus the previous package show nearly
identical instructions and replay time: ingress 67.59M→67.46M instructions,
prefill 95.08M→95.08M, decode 120.70M→120.70M; all have zero local-sector
traffic in those cells. Thus integrating search/feedback/dispatch does not
itself reduce mathematical or supporting work in the winning existing kernels.
The larger head/row alternatives remove spills but still lose on tensor
utilization and resource/work distribution; forcing them is not a remedy.
The remaining end-to-end gap is not fully apportioned by these isolated probes,
and old baseline stall percentages are not claimed as fresh measurements.

All five compiler-to-runtime joins are implemented and exercised. This closes
the implementation/integration work, **not** exhaustive tuning, a universal
best schedule, or performance parity with the reference frameworks.

## Reproduction and retained results

The v6 kernel/runtime source is committed `730378cc`, archive SHA-256
`61420f9f21bd1cfd24c68fb6ae1e59767c19ec4b7bbf452d0c17b3125672b906`.
The final benchmark overlays are mixed-descriptor probe `8b6f57ac`, complete
context/phase calibration `31ca9fce`, and resumed-serving mixed sanitizers
`287795a4`. These are offline benchmark changes; the kernel/worker binaries
remain those of the v6 source. Formatting/finalization do not alter their code.

The final v6 subdirectories are `serving-phase-complete`, `fresh-baselines`,
`final-comparison`, `trace-phase-complete`, and `counters-mixed-wide`.
Failed setup runs and slower completed ablations remain separately labeled.
The three-root results archive also retains v3 blockwise/full-versus-partial
campaigns and v5 register/resource/machine-feedback experiments. Source builds,
toolchains and model payload copies are excluded; generated sources, AOT
artifacts, probes, recipes, raw counters, traces and serving results are kept.

Downloaded without overwriting to
`/private/tmp/lunaflux-integrated-final.XtIrSY/integrated-results.tar.gz`;
local and remote SHA-256 both verify as
`915ed2279e5a0aeb20416abcc6ad953861fd596e9d433fc8374ab70928af1d59`.

## Validation

- Full local native suite: 4,213/4,213 pass.
- Affected warning-denied package suite: 314/314 pass; final frontier/export/
  executor regression subset: 220/220 pass.
- The final mixed-route regression passes; attention/graph/device-step subset
  passes 212/212. A redundant full-tree rerun was stopped during compilation
  of unrelated working-tree packages; it is not reported as an additional pass.
- `moon info`, formatting and diff whitespace checks complete.
- The global warning-denied check encounters unrelated warning 92 and warning
  14 in existing uncommitted remote-TLS work. Those files are not modified for
  this task. Suppressing those alerts is not claimed as repairing them.
- GPU work is serialized, serving units have a 64-GiB ceiling/no swap, and
  admission/measurement monitors preserve at least 32 GiB available memory.
  Reference containers use their separately documented 80-GiB ceiling/no swap.

This is instance-level BF16 benchmark work, not production deployment approval,
multi-GPU validation, broad model-quality validation or a universal speed claim.
