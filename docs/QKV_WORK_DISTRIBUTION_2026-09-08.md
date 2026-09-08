# QKV bounded output distribution — 2026-09-08

The compiler's existing one-output-tile-per-workgroup strategy reduces the
QKV microbenchmark's GPU time by approximately 12% when rotating through
28 distinct layer weights. A repeated single-weight benchmark hides this
benefit: all distributions take approximately 13 microseconds with hot weights.
This is a kernel measurement, not an end-to-end serving improvement or parity
claim against vLLM/SGLang.

## Measured shape and schedule

- RTX 5060 Ti, CUDA 13.1.115, sm_120; fixed UUID
  `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`.
- BF16 QKV projection: input 1024, output 4096, Q 2048, K/V 1024 each.
- Rows 2 and 8; output-group counts 1, 2, and 4 versus existing 8.
- Same ordered K16 WMMA accumulation and separate Q/K/V weight ABI.
- Single resident weight versus 28 distinct 8-MiB weights (224 MiB total).
- CUDA Graph timing, eight iterations per layer, three warm graphs, nine
  event samples per trial; three trials with alternating baseline/candidate order.

Table values are median trial times in microseconds for rows 8 and 28 layers.

| Output tiles/workgroup | Baseline | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 56.077 | 49.346 | 12.0% |
| 2 | 56.113 | 53.199 | 5.2% |
| 4 | 56.274 | 55.766 | 0.9% |

The one-tile schedule uses 256 CTAs rather than 32 at this shape. Rows 2 also
improved, from 56.292 to 49.302 microseconds. Paired outputs matched bitwise,
including untouched output tails; the timing probe released allocations.
These checks do not replace a full serving or sanitizer campaign.

## Bounded integration

Use the measured strategy only in the existing row-8 offline-tuning record.
The QKV regression checks that C1 and rows greater than 8 retain their prior
selection and that the maximum-profile AOT export emits the bounded variant.
No generic default is changed solely from this one-device measurement.

The canonical record payload is:

```text
record	1	1	1024	4096	1024	1001	8	3	16	16	16	1	1	49346	3
```

Functional planning owns the output distribution. CUDA lowering owns warp
mapping and WMMA. There is no change to model semantics or reduction order.

## Reproduction

- Local source modules: `/private/tmp/lunaflux-qkv-streaming-20260908-r2`.
- Probe: `/private/tmp/lunaflux-qkv-streaming-20260908.cu`.
- MoonBit runner: `/private/tmp/lunaflux-qkv-streaming-run-20260908.mbtx`.
- Remote results: `/dev/shm/lunaflux-qkv-streaming-results-20260908-r1`.
- Downloaded results: `/private/tmp/lunaflux-qkv-streaming-results-20260908-r1`.

The runner accepts input directory, new output directory, and probe source.
All timing output is outside the production runtime.
