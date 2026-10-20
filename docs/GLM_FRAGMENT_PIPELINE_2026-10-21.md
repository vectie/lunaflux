# Explicit projection fragment pipeline

Scope: component correctness/counters followed by bounded two-rank tracing.
No overall benchmark, payload hashing, or numerical-contract change.
The serving default remains unchanged while qualification is incomplete.

## Evidence-driven implementation

The retained, matched installed vLLM dispatch is
`nvjet_sm121_tst_mma_128x64x64_4_32x64x64_tmaAB_bz_TNNN`.
Its instruction capture demonstrates swizzled matrix loads, operand reuse,
interleaved loads/MMA, and reader retirement before the final register-only
MMAs. Its library implementation is not available as source; the vLLM/PyTorch
dispatch and captured SASS, not an imagined library source, are the evidence.
See `GLM_TENSOR_TRANSFER_PROJECTION_2026-10-20.md`.

The new path uses a pure `MatrixFragmentPipeline` physical plan, selected by an
explicit `MatrixRowReuse` shape envelope. The numerical IR's ascending F32
matrix fold is unchanged. Private CUDA lowering implements:

- A bijective byte swizzle shared by producers and matrix consumers.
- M32/N64/K64 ownership, four N8 accumulators reusing each A fragment.
- Two shared operand epochs and two register fragment slots.
- Next-fragment loads before current MMA; next-tile copies before the last MMA.
- A wait for the required epoch only, plus separate publication and reader
  retirement barriers. No busy polling and no dependency on an unconsumed copy.

This is `cp.async`, not a claim to reproduce vLLM's TMA implementation. It adds
no device allocation, vendor dependency, split-K output, or runtime JIT.
The single-row arithmetic path is retained. M64 was also measured, not selected.

## Physical qualification

Root: `/tmp/lunaflux-glm-projection-staging-fragments-20261021-v1`.
Actual compiler-exported source, seed713, M64/N4096/K8192 BF16 inputs and F32
output; existing shape-aware 1024-CTA launch contract retained. Five alternating
graph timing pairs, 20 replays per timing; no HTTP throughput measurement.

| Schedule | Median old | Median candidate | Smallest paired improvement |
| --- | ---: | ---: | ---: |
| M32 fragment pipeline | 426.610 us | 310.691 us | 24.65% |
| M64 fragment pipeline | 411.933 us | 311.517 us | 24.20% |

Both candidates preserve all tested active and inactive F32 output bytes
at live rows 0,1,3,22,36,64,65 (invalid counts return without stores).
The selected M32 path passes memcheck/full leak, racecheck and synccheck on
rows1,3,64 with zero reported errors, under a 2 GiB/no-swap owner.
It uses 24 KiB static shared storage versus M64's 32 KiB.
The rows1 timing difference is small and is not a demonstrated decode gain.

These are component results, not proof of complete serving selection, quality,
or full-decoder speedup. Serving AOT propagation and both-rank tracing remain
separate gates. Resource limits remain 100 GiB device/Spark, 16 GiB available
host memory, zero owned swap; unsafe trace loading must abort before the floor.

## Actual instructions and counters

The candidate binary contains `LDGSTS.E.BYPASS.128`, `DEPBAR.LE SB0,1`,
interleaved `LDSM.16.M88.4/.2` and `HMMA.16816.F32.BF16`, and reader retirement
at `0x1960` before the next copies and four final MMAs. Thus these mechanisms
reach SASS, rather than existing only in compiler plans or source strings.
`source-s1-v1.stdout` retains the disassembly.

| Same component capture | Existing | M32 candidate |
| --- | ---: | ---: |
| Profiled duration (not graph median) | 497.824 us | 378.560 us |
| Warp instructions | 16,872,448 | 17,111,040 |
| L2 read sectors | 6,384,378 | 6,294,306 |
| Tensor active | 7.284% | 9.939% |
| Active warps | 44.205% | 22.085% |
| Long-scoreboard contribution | 69.008% | 63.068% |
| Short-scoreboard contribution | 4.930% | 1.129% |
| MIO-throttle contribution | 4.731% | 2.369% |
| Barrier contribution | 13.506% | 21.216% |

The mechanism bundle improves measured completion despite slightly more
instructions and lower occupancy. It does **not** eliminate load/barrier waits.
Stall percentages have different active-cycle denominators and are not wall
time allocations; do not add them or attribute the entire gain to one mechanism.
The retained matched vLLM capture has 2.294M instructions and 3.150M L2 sectors:
the per-thread transport/addressing cost still differs materially from its TMA
schedule. This capture is not a new vLLM full-model run.

The integrated `dense_source` exporter is byte-identical to the physically
tested candidate source. Targeted native tests: 57/57. Native check passes with
the repository's existing migration-warning exclusions. Targeted `moon info`
reports 2137 existing migration warnings, zero errors. No unrelated migration
files are part of this change.

## Integration checks and retained failures

`99742cbe` builds successfully on the pinned ARM toolchain. The initial
`ed39b1c3` build failed because the newer desktop formatter rewrote `Int` as
`Int32`, unsupported by that pinned compiler. Restoring the compatible spelling
fixes the build without changing arithmetic or unrelated toolchain files.

An additional sparse-decoder regression proves policy forwarding and the
unchanged launch domain; affected native tests now pass 61/61. Full strict
native check/test were attempted and stop on existing repository migration
warnings (including deprecated `try?` in `compiler/parallel_ir`), not a claimed
full-suite pass. Logs:
`/tmp/lunaflux-glm-fragment-phase-checks-20261021-v1`.

The same M32 candidate on .178 has median 422.150 → 322.949 us, minimum paired
improvement 18.12%, with active/inactive outputs bitwise identical.
Root: `/tmp/lunaflux-glm-projection-staging-fragments-rank0-20261021-v1`.
Additional .179 shapes also pass identical-output checks, but their generic
M32/K64 baseline is not necessarily each shape's current serving selector;
they are not used to claim serving gains or broaden the selected envelope.

Both serving rank exports select the new policy only for the large sparse
output shape (fragment inner threshold8192, columns4096). Their 11 sparse
output entries contain the actual pipeline; all other projection selections
remain unchanged. Initial full-rank AOT compilation at root
`/tmp/lunaflux-glm-http-fragments-aot-20261021-v1` hit its CPU-only 4 GiB cgroup
cap, with zero owned swap. It did not load a model or exhaust device memory.
A retry at `...-v2` uses an 8 GiB CPU cap after a 24 GiB host-headroom check.
The requested runtime 100 GiB/16 GiB limits are unchanged.

The 8 GiB compile retry succeeds on both ranks. Before executing that artifact,
the physical plan's 16-byte copy-operand alignment is also forwarded through
source launch metadata into the existing startup argument table. Only selected
matrix inputs/weights acquire that requirement; other ports retain their
existing alignment. There is no new runtime scan or token-step operation.
The emitted kernel bytes are unchanged by this metadata correction.
Expanded affected tests pass 81/81; targeted interface generation has 578
existing migration warnings and zero errors.
