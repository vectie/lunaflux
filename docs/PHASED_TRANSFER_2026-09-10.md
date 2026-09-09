# Real-operand phased shared transfer

## Result and scope

The sibling-reuse gate/up lowering now stages real global operands into a
bounded register batch, finishes the CTA read phase, publishes shared vectors,
and finishes the CTA publish phase. Ordered MMA and numerical outputs remain
unchanged. This is a pure source-lowering function of tile geometry, not a
Qwen/model branch. It does not yet replace other projection or attention paths.

On the RTX 5060 Ti (sm120), both hardware shared load/store conflict totals
were zero in two samples at each token length:
`[7,17,63,64,65,255,256,257,504,1024]`. Source-attributed excessive wavefronts
were also zero. T=1 uses a shared-free path and is not counted as positive
shared-instruction coverage. These measurements do not establish zero conflicts
for every kernel, geometry, device, or future execution.

The paired 11-length numerical probe passed bitwise comparison; racecheck and
synccheck passed. At T=1024, normal timing changed from approximately 390 us
to 559 us (about 43% slower). Registers rose from 61 to 75, so the result
cannot be attributed solely to the new barriers: resource residency changes
are also involved. This deliberately accepts the user's zero-conflict-first
tradeoff; it is not an inference throughput improvement.

An additional phased epilogue variant also reached zero at T=1024 but took
approximately 563 us and was not integrated.

## Exact source and reproducibility

Remote experiment roots:

- `/run/user/1000/lunaflux-transfer-phase-20260910-r1`
- `/run/user/1000/lunaflux-transfer-phase-20260910-r2`
- `/run/user/1000/lunaflux-transfer-phase-20260910-r3`

r2 stopped at the T=1 source-coverage check because it correctly found no
shared instructions; this is not a numerical kernel failure. r3 omits T=1
from shared-coverage profiling and retains it in paired numerical testing.
All roots remain preserved without overwrite. They are not contained in the
earlier downloaded synthetic-isolation archive.

The fresh compiler export from the affected-package test
`down fold lowers masked operands through shared transport layout`, using
`LUNA_TEST_EXPORT_DOWN_OPERAND_LAYOUT=1`, is byte-identical to r3 `phased.cu`:
SHA-256 `91ef59e49fc5953871fb1516a9466cded8b4a9c7de5d1ad3c8dfa84c498768a8`.
CUDA compile flags: `--cubin -O3 -std=c++17 -arch=sm_120 --fmad=false
--maxrregcount=128`. Profiling used SourceCounters and SpeedOfLight with
explicit hardware load/store totals, cache-control all and clock-control none.
Profiler duration is not substituted for normal timing.

The generated helper has regression checks for phase separation, zero-filled
tail handling, and exactly-once vector ownership over tile/batch geometries.
Affected-package tests pass (41/41); warning-denied native check passes.

## Remaining work

QKV, output, down, vocabulary, attention and other selected shapes still need
their own real-operand transfer/consumer experiments and hardware counters.
Do not extrapolate this gate/up result to them. After zero-conflict coverage,
jointly reduce register pressure, synchronization, and address work without
reintroducing measured conflicts or altering ordered numerical behavior.
