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
Final corrected serving and fresh reference results are pending below.

## Validation

- Full local native suite: 4,213/4,213 pass.
- Affected warning-denied package suite: 314/314 pass; final frontier/export/
  executor regression subset: 220/220 pass.
- `moon info`, formatting and diff whitespace checks complete.
- The global warning-denied check encounters unrelated warning 92 and warning
  14 in existing uncommitted remote-TLS work. Those files are not modified for
  this task. Suppressing those alerts is not claimed as repairing them.
- GPU work is serialized, serving units have a 64-GiB ceiling/no swap, and
  admission/measurement monitors preserve at least 32 GiB available memory.
  Reference containers use their separately documented 80-GiB ceiling/no swap.

This is instance-level BF16 benchmark work, not production deployment approval,
multi-GPU validation, broad model-quality validation or a universal speed claim.
