# Gate/up ownership and orthogonal transport

This records the gate/up subtask of the framework-gap investigation. The new
compiler alternatives are implemented and correctness-tested; they are **not a
claim that the gate/up performance gap is closed**. Ordinary selection defaults
remain unchanged until an exact source-bound complete-chain measurement wins.

## General compiler change

The scheduled product of ordered folds is unchanged. A pure physical refinement
now independently binds:

- accumulator/epilogue ownership: split sibling planes, retain the gate producer
  and import its peer, or coown both accumulators through strict SiLU;
- finite consumer-group geometry (4/8/16/32, bounded by the strategy and tile
  ownership), with launch geometry derived rather than overridden by a probe;
- producer transport: segmented load/publication batches or independent complete
  worker traversal per operand, separate from epilogue ownership;
- explicit operand-ring and epilogue publication/retirement effects.

No model-family branch or reassociation is introduced. The CUDA lowerer consumes
these plans; ordered BF16 products, strict `expf` SiLU and BF16 nearest rounding
are preserved. Coowned values require no shared result planes or epilogue fences.
Retained ownership requires one peer plane rather than two result planes.

Offline v2 fold records can encode geometry, `epilogue:retained|coowned` and
`transport:segmented|independent`. The regenerated source digest includes every
bounded row variant. Records require matched baseline measurements; unmeasured,
slow or stale alternatives cannot silently replace the ordinary kernel.

## Why fewer instructions did not win

Paired selected-gate SourceCounters were downloaded without replacing prior
captures to `/private/tmp/lunaflux-gap-fix-sources.JCW6KUwk/` as
`gate-coowned-256-isa-{0,1}-{raw,source}.stdout`. These are instruction-level
profiling observations, not end-to-end framework timings.

| Selected gate metric | Ordinary split baseline | Coowned 64x128 / 256 threads |
| --- | ---: | ---: |
| Total executed instructions | 97.55M | 46.66M |
| `HMMA.16816.F32.BF16` | 6,291,456 | 6,291,456 |
| `LDSM.16.M88.4` | 4,718,592 | 2,359,296 |
| `LDG.E.128` | 1,572,864 | 983,040 |
| `STS.128` | 1,572,864 | 1,179,648 |
| Scalar result-plane `LDS` / `STS` | 393,216 each | absent |
| CTA barrier instructions | 860,160 | 202,752 |
| Registers/thread | 64 | 146 |
| Achieved active occupancy | 53.92% | 16.61% |
| Issue-active fraction of peak | 32.98% | 9.75% |
| Tensor-active fraction of peak | 34.04% | 21.03% |
| Local spilling requests | 0 | 0 |
| `STS.128` long-scoreboard not-issued samples | 17,838 | 54,099 |

Both captures execute synchronous `LDG -> STS`, not `LDGSTS`. Source-level copy
capability must not be mistaken for an executed async instruction. Sharing A
fragments and retaining results cuts real work, but the increased accumulator
footprint reduces resident warps; the producer's shared publication waits on
global-load results. This is not evidence that total instruction reduction is
bad, nor that occupancy alone is a performance target. It motivates independent
transport/residency experiments with identical arithmetic and launch geometry.

## Physical results and confound removal

The root coordinator serialized GPU execution. Each listed ownership/transport
alternative passed deterministic correctness and the campaign's three sanitizer
modes. Timings are unprofiled event medians, in microseconds, for the diagnostic
MLP shape hidden=1024, intermediate=3072. They do not replace real serving or
framework-comparison measurements.

| Experiment | Ordinary/control chain | Alternative chain | Interpretation |
| --- | ---: | ---: | --- |
| Retained producer, 2048 tokens | 875 | 916 | 4.7% slower; not a winner |
| Coowned 64x128 / 512 threads, 2048 | 876 | 927 | 5.8% slower; 102 registers/thread |
| Coowned 64x128 / 256 threads, 2048 | 905 | 1179 | Slower; gate 533 -> 776, 146 registers/thread |
| Same coowned geometry, independent -> segmented transport, 2048 | 1167.389 | 1010.630 | 13.43% less complete-chain time; gate 792.818 -> 671.189 (15.34% less) |
| Coowned 64x64 / 256 threads, segmented, 2048 | 864.438 | 886.541 | 2.56% slower complete chain; gate 511.354 -> 544.450 (6.47% slower) |
| Same 64x64 alternative, 512 tokens | 237.250 | 263.210 | Gate improves 121.184 -> 112.347 (7.29%), but chain is 10.94% slower |

The independent-to-segmented row is the clean orthogonal transport A/B, not a comparison against the
ordinary baseline. Compiler register ceiling, geometry and ownership are fixed.
The independent control's generated CUDA, including all bounded variants, was
compared byte-for-byte with the preserved pre-refactor source and is unchanged.
The segmented producer removes transport traversal/address work, but its gate
time still exceeds the ordinary baseline's roughly 500--560 us in these runs.
It is therefore not installed as a production winner.

Reproduction through the actual production source compiler:

```text
moon run --target native --warn-list -79-20-29-25 tests/projection_pipeline_cuda_source_probe -- gate-coowned transport-independent sibling 2 2 1 4 8 0 8 192
moon run --target native --warn-list -79-20-29-25 tests/projection_pipeline_cuda_source_probe -- gate-coowned transport-segmented sibling 2 2 1 4 8 0 8 192
```

The emitted recipe provides exact primary/down grid, block and scratch
dimensions; launchers must use those dimensions. Lowering all row variants is
essential: a source-only primary change must not silently revert on a smaller
bucket. The guided 64x64 trial reduces the coowned output tile to test the
accumulator/operand residency tradeoff; it remains slower as a complete chain.
No additional geometry is presumed fast.

## Verification and remaining work

### Final matched 64x64 counter capture

The final paired files are `gate-segmented-square-isa-{0,1}-{raw,source}.stdout`
in the same preserved local directory. Both launch **1536 CTAs**; the ordinary
kernel uses 512 threads and the coowned segmented square uses 256. These are
separate Nsight-profiled durations, **not** the unprofiled event medians above:
733.952 us versus 742.592 us (alternative about 1.18% slower).

| Matched selected-gate metric | Ordinary split baseline | Coowned 64x64, segmented |
| --- | ---: | ---: |
| Executed instructions | 97.55M | 53.98M |
| `HMMA.16816.F32.BF16` | 6,291,456 | 6,291,456 |
| `LDSM.16.M88.4` | 4,718,592 | 3,145,728 |
| `LDG.E.128` / `STS.128` | 1,572,864 each | 1,179,648 each |
| Scalar result-plane `LDS` / `STS` | 393,216 each | absent |
| CTA-barrier warp instructions | 860,160 | 405,504 |
| `MOV` | 13,258,752 | 11,476,992 |
| `IMAD` | 13,553,664 | 823,296 |
| `LEA` | 2,039,808 | 3,588,096 |
| Registers/thread | 64 | 102 |
| Register-limited resident blocks | 2 | 2 |
| Shared-memory-limited resident blocks | 2 | 4 |
| Achieved active occupancy | 56.37% | 33.01% |
| Issue-active fraction of peak | 28.43% | 15.35% |
| Tensor-active fraction of peak | 29.33% | 28.62% |
| Local spilling requests | 0 | 0 |
| `STS.128` long-scoreboard not-issued samples | 17,772 | 28,633 |
| `BRA.U` barrier not-issued samples | 16,267 | 10,111 |
| MMA math-pipe-throttle not-issued samples | 2,837 | 6,064 |
| MMA short-scoreboard not-issued samples | 505 | 1,852 |

This resolves the remaining attribution more narrowly than "more instructions":
the segmented square removes substantial address/control work, one third of
fragment loads, one quarter of vector transfers and the shared result planes.
But 102 registers still limit it to two resident blocks; at half the threads
per block that leaves fewer resident warps than the ordinary kernel. The
instruction-local hotspot remains `STS.128` waiting for global-load results,
even with fewer such instructions. Tensor fragment dependency/throttle samples
also rise, while barrier samples fall. The register-retained ownership has moved
work out of shared publication into per-consumer accumulator/fragment lifetimes;
the resulting lower latency-hiding capacity offsets its useful-work savings.
No spilling was observed, and shared-memory capacity is no longer the binding
resident-block limit for the alternative.

Not-issued samples are sampled observations at those instructions, not elapsed
time fractions. Their counts must not be converted into a speedup prediction.
The nearly equal tensor-active fractions and slower complete-chain measurements
confirm that lower total instructions alone do not establish a better schedule.
The new compiler alternatives remain diagnostic/offline-selectable; the
ordinary baseline remains preferred by the measured complete-chain results.

Affected compiler, tuning and projection packages passed 200/200 native tests
with known toolchain migration warnings 79/20/25/29 excluded. Regression coverage
includes deterministic ownership/transport planning, active tail ownership,
primary and scalar-floor launch geometry, unchanged down geometry, bucket policy
inheritance, v2 grammar rejection and source-bound measured selection.
After the coordinator removed the remaining old redundant test annotation, the
affected native check also passed with warning 73 explicitly enabled. The
coordinator's broader aggregate passed 415/415 with that warning configuration.

The implementation remedies are executable and generally selectable. The
remaining performance issue is the register/residency and global-load-publication
balance; the alternatives measured so far cannot close the reference gap. The
guided 64x64 trial comes close at 2048 but still loses; its 512-token gate win
does not translate to a complete-chain win. Keep the ordinary baseline. Do
not describe this subtask as a completed performance fix or promote it on reduced
barrier/instruction counts alone.
