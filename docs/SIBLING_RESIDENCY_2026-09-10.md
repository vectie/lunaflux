# Gate/up residency and synchronous transfer scheduling

## Current result and scope

The integrated gate/up schedule measures **354–356 us** at
T=1024, input width 1024 and intermediate width 3072 on the RTX 5060 Ti.
The r9 same-run comparison against the committed phased implementation gives
559.15 us versus 354.17 us (three-trial medians), about 36.7% less kernel time.
It is also about 9% below the historical 390-us path, although that historical
comparison is not interleaved in this campaign. These are not end-to-end
inference speedups or a claim of global optimality.

The paired reference in the r5 timing probe is a separate older binary
(approximately 500 us at T1024). Eleven token lengths match that reference
bitwise. Two T1024 SourceCounters samples each report 3,538,944 executed shared
instructions and zero excessive wavefronts. Hardware load/store totals remain
nonzero; they are no longer an optimization acceptance target by themselves.

The implementation is now expressed in the normal compiler-to-CUDA lowering:

- Construct shared addresses immediately before publishing, shortening their
  live register range across global reads.
- Consume the current operand slot before preparing the next slot.
- Remove empty async commit/wait instructions from synchronous transfers.
- Remove internal read/publish CTA fences; preserve initial publication,
  every loop-end publication/reuse fence, result publication and the final
  work-item ownership fence. The final K iteration still fences before
  operand storage is reused for gate results.
- Apply a kernel-local two-block register target only to the measured
  512-thread sibling-reuse topology when the caller budget is at least 64.
  Propagate the caller budget to both primary and row-variant generation.
  Smaller workgroups and non-sibling kernels retain the existing policy.
  The companion down kernel retains its file-level register budget.

This is pure immutable schedule inspection followed by CUDA lowering, with no
model-name branching or runtime autotuning. The two-block target is a measured
CUDA resource policy, not a portable promise of residency. This work does not
add a general cross-backend register-pressure optimizer or synchronization
elimination pass to LunaTile IR.

## Incremental r4/r5 experiments

| Transformation | T1024 normal kernel time (us) |
| --- | ---: |
| Deferred register-batched copy, gate-local launch bounds | 371.23–371.39 |
| Also remove empty async group operations | 359.36–359.46 |
| Also remove the internal read/publish CTA fence | **355.82–355.99** |
| Move the fence-free next copy back before current MMA | 379.14–379.20 |

The selected r5 source uses 58 gate registers and 112 down registers, with no
stack frame or register spills. The earlier global register-cap experiment
also reduced down's budget; the kernel-local policy avoids that side effect.
The approximately 0.3% r4 global-cache-policy variant was not selected.

The r9 paired timing vector below reports medians of three trials, each timing
100 CUDA-graph-replayed launches. These descriptive medians are not confidence
intervals. Unlike the older reference in r5/r6, the reference here is the
committed phased implementation. T1 improves versus that implementation but
remains slightly slower than the approximately 9.32-us older reference.

| Tokens | Committed phased gate/up (us) | Current compiler gate/up (us) |
| ---: | ---: | ---: |
| 1 | 13.540 | 9.426 |
| 7 | 35.213 | 18.773 |
| 17 | 36.565 | 19.496 |
| 63 | 51.347 | 35.203 |
| 64 | 51.575 | 35.290 |
| 65 | 62.192 | 40.797 |
| 255 | 153.363 | 99.535 |
| 256 | 153.252 | 99.710 |
| 257 | 165.657 | 106.162 |
| 504 | 281.249 | 180.499 |
| 1024 | 559.145 | 354.171 |

The companion down entry remains at 112 registers and approximately 187.04 us
at T1024 in the same-run old/new comparison. The 175-us down reference used by
the older probe is not the committed phased baseline and must not be used to
attribute a new down regression to this gate-only change.

## Same-binary residency isolation

A separate experiment used the **same** r3 `deferred64.cubin` at all three
points. It changed only the host-requested dynamic shared-memory reservation;
the extra bytes are unused. NCU reported the same actual shared-memory
configuration (102,400 bytes/SM), 60 registers/thread, 512 threads/block and
768 blocks at all points.

| Dynamic shared bytes/block | Shared-resource block limit | Active warps/SM | T1024 median (us) | Hardware load/store, two samples |
| ---: | ---: | ---: | ---: | --- |
| 16,384 | 2 | 31.23–31.24 | 371.282 | 74,651/715,710; 76,154/721,598 |
| 25,600 | 2 | 31.21–31.23 | 369.320 | 52,989/428,416; 56,207/430,732 |
| 25,856 | 1 | 15.95 | 499.186 | **0/0; 0/0** |

Static shared storage (24,576 bytes), the middle dynamic reservation (25,600)
and driver reservation (1,024) sum to 51,200 bytes/block: two blocks exactly
fit 102,400 bytes. Adding 256 bytes prevents two-block residency. Achieved
occupancy falls from about 65% to 33.24%; tensor activity falls from about 67%
to 46%. The profiler does not directly provide an active-CTA count here;
the block resource limit and measured active warps support this interpretation.

The residency threshold makes latency **35.16% worse while making hardware
conflicts zero**. All six Source reports still have zero excessive shared
wavefronts with positive shared-instruction coverage. No recompilation,
register-allocation change or actual shared/L1 partition change explains this
comparison. This strongly supports lost concurrency and latency hiding as a
cause of the earlier zero-count slowdown, rather than a corrected address
layout. It does not identify every physical arbitration event on sm120.

Even the two-block points differ in hardware totals, so CTA count alone does
not explain all remaining hardware events. Shared physical placement and
arbitration can also matter. Do not add unused shared reservation to production
for the small 0.53% timing difference observed between those two points.

## Validation boundary and reproduction

Existing remote results, preserved without replacement:

- `/run/user/1000/lunaflux-transfer-reorder-20260910-r4`
- `/run/user/1000/lunaflux-transfer-reorder-20260910-r5`
- `/run/user/1000/lunaflux-occupancy-isolation-20260910-r1`
- `/run/user/1000/lunaflux-current-gate-20260910-r6`
- `/run/user/1000/lunaflux-current-gate-20260910-r7`
- `/run/user/1000/lunaflux-small-reuse-20260910-r8`
- `/run/user/1000/lunaflux-gate-paired-20260910-r9`

Device UUID: `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`; PCI
`00000000:17:00.0`; CUDA 13.1, `sm_120`, `-O3 --fmad=false`.
NCU uses SourceCounters, SpeedOfLight, LaunchStats, Occupancy and explicit
hardware shared load/store totals. Its single-launch cold-cache durations are
not interchangeable with ordinary graph-replayed timing. Clocks were not
locked. No vLLM, SGLang or end-to-end Qwen measurement was made in this step.

The local compiler export is whitespace-normalized identical to r5
`no_fence.cu`, but not byte-identical. Its fresh r6 compilation produces a
byte-identical CUBIN to the r5 experiment:

- r5 source SHA256:
  `d106ccf313139eb8e2a3cb479ef92e212827afc5f1a72bfd0e08b5946068eebd`
- r5 CUBIN SHA256:
  `0c3283069bf99e9061ffd880d649cedc337b120d99f874249f26905141fc7b62`
- local r6 compiler export SHA256:
  `0aafac11f679d5ffbd1e54fc03aa03cdd81299c4e4919accd9f1c00ac00b7a7f`
- same-binary isolation CUBIN SHA256:
  `a3708a87c739271698661c812e23200573a959e5e4c9f0ac1dd506fbf3ae8980`

Fresh r6 GPU compilation and checks passed after explicit renewed user upload
approval. Gate/up and down each pass the paired 11-length numerical vector.
Gate/up SourceCounters have positive shared-instruction coverage and zero
excess at `[7,17,63,64,65,255,256,257,504,1024]`. Memcheck, initcheck, racecheck
and synccheck pass at `[7,65,1024]`. Hardware totals are not required to be zero.

The r7 supplement uses the same r6 CUBIN with a smaller grid (192 rather than
768) to force four work-items per CTA at T1024. Its 18-length fixture
`[1,2,3,7,15,16,17,31,32,33,63,64,65,255,256,257,504,1024]` matches bitwise.
Racecheck/synccheck pass at `[2,3,15,16,17,31,32,33,63,64,65,1024]` and
memcheck/initcheck at T1024. CUBIN hashes before and after are unchanged.
This is ownership/tail coverage, not a proposal to change production grid size.

The r8 campaign compiles verbatim current exports for 4- and 8-warp gate
workgroups (128/256 threads). Both use the new transfer lowering without the
512-thread launch-bounds qualifier. Each passes 18 lengths
`[1,2,3,7,15,16,17,31,32,33,63,64,65,127,128,129,255,256]` against the
deterministic fixture. At `[7,65,256]`, both pass all four sanitizers and two
SourceCounter samples per length: 24 sanitizer checks and 12 positive-coverage,
zero-excess profiles in total. The companion down entry is compiled but not
launched in r8; down execution coverage comes from r6/r9. The small-topology
T1 export requests an ordered scalar fold while the reference requests a
tree; the dyadic fixture passes bitwise, which is not a claim that different
reduction orders are bitwise equivalent for arbitrary inputs. These small
topologies are coverage tests, not new autotune winners.

The r9 comparison passes the paired 11-length vector for gate/up and down
against the actual committed phased kernel. No serving deployment changed.

Local regression coverage checks transfer ownership and tail initialization,
gate-isolated synchronization assertions, register budgets 32/63/64/128,
non-sibling and smaller-group exclusions, companion isolation, and primary /
row-variant propagation. `moon info`, formatting and warning-denied native
check pass; affected-package tests pass **46/46** and the full native suite
passes **3,660/3,660**. The first sandboxed full run had three loopback socket
permission failures; the approved rerun with local socket access passed the
entire suite without a code change.

Implementation commit: `38f72cb`. The r6/r7/r8/r9 and same-CUBIN occupancy
results, generated sources, CUBINs, reports and reproduction drivers were
archived as `lunaflux-gate-optimization-20260910.tar.gz` and downloaded without
replacing earlier results. Remote and local archive SHA256 match:
`39cc3349f757c1303ce3a72e1230cd4a99af76ae3b9d852a9321af71aafe1214`.
