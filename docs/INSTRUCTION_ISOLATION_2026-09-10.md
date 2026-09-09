# Gate/up shared replay isolation

## Outcome

The production kernel is unchanged. The experiment localizes two useful
directions without claiming the residual hardware counters are all fixed:

1. The shared producer address map executes with **zero hardware load/store
   totals** when its operand data is synthesized in registers. Restoring
   global operand loads produces about 553,000 store replays with the same
   shared wavefront count. The static shared map alone is therefore not a
   sufficient explanation for this producer's aggregate count.
2. Removing the real epilogue while keeping all MMA results live through a
   checksum reduces load totals from about 57,000 to 3,800. This implicates
   the epilogue or its interaction with other work; it is not an additive
   instruction-by-instruction attribution.

## Controlled scope

RTX 5060 Ti, UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, sm120,
grid 768, block 512, T=1024. Source comes from the final gate/up experiment
`lunaflux-copy-fix-20260909-r2/mlp.cu`. Every variant retains the same launch
geometry, 24,576 static shared bytes and 16,384 requested dynamic shared
bytes. Register counts vary and are reported below. CUDA 13.1 compiler flags
remain O3, fmad=false and maxrregcount=128.

NCU used SourceCounters, SpeedOfLight and WarpStateStats, plus hardware
shared-load/store aggregate counters, cache-control all and clock-control
none. Two profiled launches per variant. Durations are profiler measurements
with unfixed clocks, not end-to-end benchmark speedups or confidence intervals.

## Valid r2 measurements

| Variant | Hardware load | Hardware store | Duration us | Registers |
| --- | ---: | ---: | ---: | ---: |
| Full control | 56,472–56,975 | 482,061–484,145 | 406.91–408.00 | 61 |
| No epilogue, checksum sink | 3,745–3,798 | 499,780–504,753 | 378.43–382.56 | 55 |
| Producer only | 0 | 554,077–556,344 | 253.98–254.98 | 43 |
| Producer without loop CTA barrier | 0 | 516,032–531,516 | 225.15–226.62 | 43 |
| Consumer with fixed operands | 8,316–8,403 | 22,603–23,858 | 282.82–282.94 | 52 |
| Fixed consumer, MMA replaced by checksum arithmetic | 10,188–10,380 | 15,171–15,538 | 132.42–132.70 | 43 |
| Fixed consumer, warp instead of loop CTA barrier | 11,467–11,671 | 22,921–23,835 | 268.90–269.12 | 52 |

All source-attributed copy and other shared excessive wavefronts are zero.
Actual equals ideal shared wavefronts: control 12,189,696; no-epilogue
11,796,480; producer 2,359,296; fixed consumers 9,584,640. This confirms
that r2's consumer shared accesses survived compilation.

Fixed consumers initialize both stages once, retain the initial CTA barrier,
then reuse read-only shared operands for the original 32 iterations. They do
not perform the original matrix product. No-MMA consumes all loaded register
fragments in checksum arithmetic; it is not a zero-cost removal of MMA.
The no-epilogue sink adds a small global output and arithmetic. Resource,
scheduling and dependency changes preclude subtracting durations as exact
component costs. Warp-only synchronization is safe in this read-only
experiment, not necessarily in the production pipeline with shared updates.

## r3 producer data-source isolation

| Variant | Hardware load | Hardware store | Duration us |
| --- | ---: | ---: | ---: |
| Global operands, repeated control | 0 | 552,567–552,807 | 253.18–254.11 |
| Register-generated operands | 0 | 0 | 93.41–93.44 |
| Register operands, no loop CTA barrier | 0 | 0 | 83.17–83.39 |

All three execute 2,359,296 actual and ideal shared wavefronts. Store work
has not disappeared: only the global operand loads are replaced by generated
register values. Counts-buffer access and a minimal output remain. This
supports a global-load/shared-store execution interaction, but does not by
itself identify the exact hardware arbitration rule. Removing required loads
is not a valid production optimization or a predicted 2.7x serving speedup.

## Validation and experiment correction

The first r1 design removed result use. The compiler eliminated consumer
work despite inline assembly in the source: no-epilogue showed only
2,359,296 shared wavefronts, and fixed consumer only 147,456. These r1
consumer comparisons are invalid for intended attribution and are excluded.
r2 adds a live checksum sink; no-MMA also consumes every loaded fragment.

The 10 valid r2/r3 configurations passed 20 racecheck/synccheck invocations
and produced 20 NCU reports. These checks establish experiment synchronization
sanity, not numerical inference correctness. No ablation was deployed or
admitted as a production kernel. The MoonBit agent workflow was used for the
`.mbtx` driver; its local warning-denied build-only check passed.

Next experiments should preserve real operand values while separating load
issue, register staging and shared-store issue, then isolate epilogue
shared reads from global output writes. Any successful scheduling change
belongs in generic transport/epilogue lowering and must regain full numerical
and timing validation before adoption. Another blind swizzle change is not
supported by these results.

## Artifacts

Remote `/run/user/1000/lunaflux-isolation-20260910-r{1,2,3}` contains source,
CUBINs, sanitizer logs, NCU reports, source summaries and detailed metrics.
The corresponding versioned `.mbtx` drivers are preserved alongside them.
Archive: `/run/user/1000/lunaflux-isolation-20260910-results.tar.gz`.
The initial download was rejected by automatic approval. After the user
explicitly authorized download, the archive was saved without overwrite at
`/private/tmp/lunaflux-isolation-results.xcLDVn/lunaflux-isolation-20260910-results.tar.gz`.
Remote and local SHA-256 match:
`1d566d3c3a9343f41c5d937c28b78360be35ccdc5a19fab54a7298f501adda98`.
