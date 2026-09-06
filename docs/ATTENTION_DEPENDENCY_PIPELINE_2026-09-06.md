# Query residency and independent K/V readiness

The grouped decode compiler now derives a transfer-readiness plan from its
immutable schedule. The two-stage asynchronous schedule expresses K readiness
for scoring separately from V readiness for the online update. CUDA alone
lowers this into copy groups, waits and barriers. The query is loaded once
outside the KV fold, and K/V transfers share their computed page addresses.
The logical partition grain and arithmetic order are unchanged.

This is a kernel experiment, not a new end-to-end serving result or a change to
the serving default. Matrix scheduling, fusion profitability, work planning and
output-demand elimination remain separate workstreams.

## Paired GPU measurement

RTX 5060 Ti, CUDA 13.1.115, BF16 input/output, FP32 accumulation, 16 query heads,
8 KV heads, head dimension 128. Old is the previous 32-position asynchronous
pipeline; new adds query residency, shared address calculation and independent
K-score/V-update readiness. Both retain the 64-position logical partition grain.
Three paired trials, reversed variant order in trial two, 200 graph launches
per sample; entries are arithmetic means of the three per-trial timings.

| Batch | Context | Direct old → new (µs) | Speedup | Split-8 total old → new (µs) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 59 | 8.14 → 6.14 | 1.32× | 9.32 → 7.52 | 1.24× |
| 1 | 128 | 15.13 → 10.47 | 1.44× | 10.66 → 8.81 | 1.21× |
| 1 | 512 | 54.26 → 35.03 | 1.55× | 11.09 → 9.05 | 1.23× |
| 1 | 1,528 | 159.20 → 100.22 | 1.59× | 26.37 → 19.20 | 1.37× |
| 1 | 4,096 | 415.28 → 264.15 | 1.57× | 64.36 → 44.47 | 1.45× |
| 8 | 59 | 9.24 → 7.22 | 1.28× | 16.51 → 14.83 | 1.11× |
| 8 | 128 | 17.36 → 12.57 | 1.38× | 23.83 → 20.24 | 1.18× |
| 8 | 512 | 63.41 → 42.74 | 1.48× | 70.75 → 54.89 | 1.29× |
| 8 | 1,528 | 203.11 → 136.32 | 1.49× | 202.56 → 145.20 | 1.40× |
| 8 | 4,096 | 535.01 → 343.58 | 1.56× | 508.94 → 358.73 | 1.42× |

The unchanged merge remains approximately 1.37–2.38 µs. These measurements
combine three transformations; they do not isolate each one's contribution.

## Correctness and reproducibility

- Bitwise agreement with the preceding vector kernel at contexts 1, 7, 8, 9,
  59, 63, 64, 65, 128, 256, 512, 1,528 and 4,096, including mixed-row checks.
- Independent reference tolerance, unchanged KV and deterministic resource
  release checks passed in the timing harness.
- Memcheck, racecheck, synccheck and initcheck completed without reported
  errors or hazards for batch 8/context 128. This is not exhaustive validation
  of every shape or a soak.
- GPU UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI `00000000:17:00.0`.
- Source SHA-256: `74e5b5911d3ba00a64a72a06d686a3402f5ea4bc4e79bb160c94d8ebbede2543`.
- Cubin SHA-256: `bb09f8831de4e9b7d5bf09095ce69c3c042a96a62e390207d2228961950f7ace`.
- Downloaded archive: `/private/tmp/lunaflux-dependency-decode-20260906-r1.tar.gz`.
  SHA-256 `e2990c510dff3e8f941817dac65be570a0680696a196a5ee95f9c08503fb07c1`
  matches the remote archive. It contains source, automation and raw timings.

Source-level regression tests check query residency and both readiness points.
The native schedule/lowering/source suites cover the generic plan and its CUDA
consumer. No runtime measurement, allocation or compilation was added to the
token-step path.
