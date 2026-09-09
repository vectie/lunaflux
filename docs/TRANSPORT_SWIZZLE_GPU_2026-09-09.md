# Producer/consumer swizzle GPU experiment

Scratch experiment only; production source and deployment are unchanged.
RTX 5060 Ti, UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI
`00000000:17:00.0`; GPU idle before measurement. CUDA 13.1 toolchain, sm120,
O3, fmad disabled, max registers 128, matching the preceding experiment.

The r5 compact operand layout was replaced in scratch QKV/output sources with
`row * 64 + (column ^ ((row & 7) * 8))`. Producer logical assignment and explicit
consumer fragments remain those of r5. Both `.cg` and `.ca` policies were tested.
This jointly changes physical producer destinations and consumer addresses,
without changing the numerical fold.

## Paired timings

Three trials, 30 repeats per trial; values are median CUDA-event microseconds.
Baseline is the retained accepted kernel, not the rejected compact experiment.
The faster tested `.cg` variant is shown:

| Kernel | Tokens | Accepted baseline | New swizzle | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| QKV | 256 | 112.71 | 102.65 | 8.9% |
| QKV | 257 | 120.88 | 108.64 | 10.1% |
| QKV | 504 | 213.11 | 194.66 | 8.7% |
| QKV | 1024 | 420.45 | 382.31 | 9.1% |
| Output | 256 | 57.38 | 51.29 | 10.6% |
| Output | 257 | 57.40 | 51.28 | 10.7% |
| Output | 504 | 110.41 | 100.37 | 9.1% |
| Output | 1024 | 202.89 | 184.44 | 9.1% |

Both kernels under both policies passed bitwise comparisons for
`1,7,17,63,64,65,255,256,257,504,1024`. Short shapes below the pipeline threshold
remain effectively unchanged. The `.ca` medians at 1024 were 385.08 µs for QKV
and 186.47 µs for output; `.cg` is preferable in this measured experiment.
Clocks were not locked; QKV's `.cg` trials ranged 380.39–385.03 µs, with paired
baseline 415.66–420.45 µs. This is an isolated-kernel improvement, not an
end-to-end serving or baseline-framework comparison.

## Transport counters

Full Nsight runs were separate from event timing. For QKV, compared with rejected
r5 `.cg`:

- Async-copy read sectors: 25,165,824 → 12,582,912.
- Source-correlated copy excessive shared wavefronts: 44,040,192 → **0**.
- Source-correlated copy excessive global sectors: 12,582,912 → **0**.
- L2 TEX read requests: 25,166,023 → 3,145,914.
- Long-scoreboard cycles per issued instruction: 4.798 → 0.301.
- Barrier cycles per issued instruction: 1.365 → 0.480.
- Executed instructions: 80,691,200 → 85,491,712.

This is faster despite more instructions: it removes the transport amplification
identified in the prior diagnosis. The zero values are copy-instruction metrics,
not an assertion that every shared access in the kernel is conflict-free.

Output's async-copy read sectors are 6,291,456; L2 TEX read requests 1,572,918;
long-scoreboard ratio 0.160; barrier ratio 0.492. Output register use increases
from 96 to 118 per thread; static shared memory stays at 30,720 bytes. The
resource change must be considered before wider-shape or production adoption.

## Scope and next boundary

No sanitizer, end-to-end serving, down/head/attention, or all-shape qualification
is claimed by this run. Integrating the measured layout into the compiler and
validating its full dispatch coverage remain separate work.

Remote artifacts: `/run/user/1000/lunaflux-transport-swizzle-20260909-r1`.
Local download: `/private/tmp/lunaflux-transport-swizzle-results-20260909-r1`.
Each contains generated sources, cubins, compile logs, paired logs and two
full Nsight reports. Runner: `/private/tmp/lunaflux-transport-swizzle.mbtx`.
