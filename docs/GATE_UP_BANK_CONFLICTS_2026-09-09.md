# Gate/up bank-conflict correction

Follow-up to [operand reuse](OPERAND_REUSE_AND_LAYOUT_2026-09-09.md).
The previous sibling-axis schedule reused weight fragments but retained
full-row shared-memory pitches. This change maps input, gate and up operands
to compact 16×16 tiles, including both asynchronous producers and WMMA
consumers. The immutable layout and its address renderer are private CUDA
lowering helpers. The compiler's pure sibling/row partition and ordered K
fold remain unchanged; no model-family branch or public API is introduced.

There is no storage growth: static shared remains 24,576 bytes, dynamic
scratch 16,384 bytes, and register count 92 for the measured kernel. The
expired-stage epilogue alias, synchronization and BF16 rounding are unchanged.

## Paired physical results

RTX 5060 Ti sm120, UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`.
Control is the immediately preceding sibling-reuse CUBIN. Three interleaved
trials, three warmups and 30 CUDA-event-timed launches per trial, synthetic
BF16 operands. Median isolated kernel times:

| Tokens | Previous µs | Compact layout µs | Speedup |
| --- | ---: | ---: | ---: |
| 256 | 165.99 | 149.49 | 1.11× |
| 257 | 174.19 | 157.81 | 1.10× |
| 504 | 303.26 | 272.30 | 1.11× |
| 1024 | 612.11 | 546.63 | 1.12× |

The token vector 1,7,17,63,64,65,255,256,257,504,1024 passes complete
bitwise workspace comparison, including untouched tails. Below256 tokens the
unchanged path has no meaningful timing difference. The down entry point in
the combined module passes the same vector and shows no material regression.
These are not whole-model or serving throughput measurements.

## Fresh selected-kernel counters

Nsight Compute full collection, 40 passes per kernel, cache control `all`,
clock control `none`; 1024-token launch. Barrier/stall metrics are average
warp-stalled cycles per issued instruction, not wall-time percentages.

| Metric | Previous | Compact layout |
| --- | ---: | ---: |
| Shared-load bank conflicts | 28,549,516 | 9,542,074 |
| Shared-store bank conflicts | 196,608 | 196,608 |
| Load + store conflicts | 28,746,124 | 9,738,682 |
| Tensor-pipe active | 40.99% | 46.13% |
| Barrier cycles/issue | 2.130 | 1.619 |
| Short-scoreboard cycles/issue | 0.757 | 0.503 |
| Local load/store sectors | 0/0 | 0/0 |

Load conflicts decrease **66.6%**, and combined load/store conflicts decrease
**66.1%**. They are not zero; this fixes the previous lack of aggregate
conflict reduction without claiming optimal memory behavior. This paired
experiment changes only operand layout, retaining sibling distribution and
epilogue order, so it isolates the layout change from the earlier reuse work.

## Checks and reproduction

The two affected packages pass **65/65** tests, including exhaustive compact
address bijection, copy-vector alignment and matrix-fragment mapping across
multiple tile widths. Warning-denied native checking, formatting and interface
generation pass. The full repository suite is not rerun for this scoped
lowering edit; the preceding batch passed3639/3639.

The changed kernel passes memcheck, racecheck, initcheck and synccheck at
1024 tokens: zero errors, and zero race warnings.

Generated CUDA SHA-256:
`6fae69f9861944544216b12f69961e74095850787e6393ba5f09ecc223168d8a`.
Compile flags: `--cubin -std=c++17 -O3 -arch=sm_120 --fmad=false
--maxrregcount=128 --ptxas-options=-v`, CUDA13.1.
Run root: `/run/user/1000/lunaflux-gate-layout-20260909-r1`.
Raw timings, CUBINs, CUDA source, runner and Nsight reports are retained there.
Implementation commit: `b98eede`.
Archive: `/run/user/1000/lunaflux-gate-layout-archive-20260909-r1/results.tar.gz`.
SHA-256: `04e82ad20cba229a85617f5b15267a1e9612371a5f80f2bce5ff4d9d7cd8e4b9`.
After the run, GPU memory returned to the initial15MiB with no compute process.
