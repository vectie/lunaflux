# Full prefill compiler-path activation

The exported prefill family now requires query-owned fragments and dense-current /
paged-history K/V reads, including single, wide-alias, and partitioned entries.
Bucket specialization preserves those immutable compiler requirements. Small
query buckets do not retain the old ownership solely to avoid a speed regression.
Decode remains its separately compiled implementation; this is not a claim that
every model or every device backend has been physically validated.

Tested runtime: `3ad4a98`. Source archive SHA-256:
`30aa5a57e545de39f69089b7a27c43b4195ae08128fe2782f25624e985d4cd10`.
Control: the `4896771` new-toolchain baseline, rerun with identical requests.
Both use MoonBit v0.10.12, Qwen3-0.6B BF16, one RTX 5060 Ti, prefix caching
disabled, 2,048-token prefill chunks, the same 8,192-page KV capacity, greedy
sampling and fixed output lengths. Inputs repeat/truncate the same recorded
1,528-token sequence. No production deployment was changed.

## End-to-end results

Each cell has one discarded warm-up and five measured trials. Throughput is
the arithmetic mean of complete concurrent-request output tokens / wall time;
TTFT is the mean across measured requests. Engines ran sequentially on the same
GPU. These are measurements, not statistically established universal speedups.
Differences below 1% should be treated as roughly neutral.

| Input/output tokens | C | Control tok/s | Activated tok/s | Throughput gain | TTFT ms: control → activated |
|---|---:|---:|---:|---:|---:|
| 512/64 | 1 | 211.23 | 211.36 | +0.06% | 29.00 → 28.00 |
| 512/64 | 8 | 1013.46 | 1019.92 | +0.64% | 112.25 → 109.58 |
| 512/64 | 16 | 1334.80 | 1346.71 | +0.89% | 190.53 → 184.99 |
| 1528/32 | 1 | 145.47 | 148.16 | +1.85% | 70.00 → 66.60 |
| 1528/32 | 8 | 339.61 | 348.21 | +2.53% | 306.05 → 293.65 |
| 1528/32 | 16 | 371.23 | 381.41 | +2.74% | 560.59 → 540.42 |
| 3072/32 | 1 | 98.10 | 100.89 | +2.84% | 156.00 → 147.20 |
| 3072/32 | 8 | 160.44 | 166.36 | +3.69% | 746.23 → 711.15 |
| 3072/32 | 16 | 168.00 | 174.80 | +4.05% | 1373.94 → 1308.54 |
| 4096/64 | 1 | 107.24 | 109.93 | +2.51% | 226.80 → 209.80 |
| 4096/64 | 8 | 186.02 | 193.08 | +3.79% | 1105.90 → 1044.47 |
| 4096/64 | 16 | 148.23 | 155.09 | +4.62% | 2044.76 → 1932.09 |

For inputs ≥1,528 tokens, throughput gains range from 1.85–4.62% (geometric
mean of cell ratios: 3.18%), and TTFT falls by 3.60–7.50%. Isolated C1 decode
intervals are almost unchanged. Concurrent per-request decode intervals also
include interference from other requests' prefill; they are not isolated decode
GPU timings. The 4,096/C16 input alone equals the entire 65,536-token KV capacity,
before output allocations. Interpret this as a capacity-pressure workload, not
a clean batching-efficiency experiment; this run did not collect a scheduler
trace attributing its C8→C16 throughput drop.

vLLM and SGLang were not rerun here. This does not establish the half-vLLM-time
target or a new comparison against either framework.

## Numerical comparison — not complete token identity

Both versions completed all measured requests and shut down cleanly. However,
only **475/500 complete sequences** match exactly. In the 3,072/32 workload:

- C8: 10/40 new sequences differ from control.
- C16: 15/80 new sequences differ from control.
- Every difference is at output position 32: token ID 16 becomes 22; the
  preceding 31 tokens match. Across all cells, 23,975/24,000 output tokens match.
- The old version has one unique sequence in each of these cells; the new
  version has two. This is a newly observed batch-sensitive numerical difference,
  not the previously recorded C2 discrepancy and not evidence that it is fixed.

The new schedule explicitly permits an alternative softmax reduction tree;
bitwise equivalence was not its numerical contract. Nevertheless, no logits or
top-two margin was captured, so a near-tie explanation remains unverified. Do
not call these runs complete numerical parity or model-quality validation.
Further investigation should capture logits at the first differing position
with matched batching before changing arithmetic or dismissing the difference.

## Actual exported attention module

The activated CUBIN SHA-256 is
`854220710c256816303c8f73c404cec4fe5f5764500894b9ee4d8234b002dd73`.
It is byte-identical between activation-r2's numerical/sanitizer campaign and
activation-r3's complete runtime. It uses 203 registers, zero local-memory and
stack allocation, 1,024 static shared bytes plus 49,168 dynamic shared bytes.
The CUDA compile ceiling is 255, avoiding the old forced-128-register spill
behavior. This is a complete selected-schedule comparison, not an attribution
to a single IR pass.

| Packed queries / rows / history | Control µs | Activated µs | Time reduction |
|---|---:|---:|---:|
| 1528 / 1 / 0 | 557.42 | 472.74 | 15.19% |
| 1528 / 8 / 4096 | 2740.38 | 2575.47 | 6.02% |

These are medians of five paired CUDA-event measurements. The 34-case matrix
covers query tails, ragged rows, fragmented pages, no history and 4,096-token
history, with an independent sampled scalar oracle and unchanged persistent
KV checks. Memcheck, racecheck and synccheck pass. These kernel checks do not
erase the complete-model token differences above.

## Integration fixes and validation

Full-path testing found two integration defects missed by microbenchmarks:

1. Segmented producer lowering emitted unsigned `threadIdx.x >= 0`, rejected
   by release nvcc's warning-as-error policy. Bounds are now partially evaluated.
2. Runtime prefill binding hard-coded 256 workers and rejected the new legal
   128-worker AOT schedule. It now retains checked compiled geometry, while
   preserving partial/merge consistency and semantic row checks. A new test
   failed before the fix and passes for 64/128/256 workers afterward.

Native check passes; full tests **3,732/3,732**, affected execution package
**190/190**. Some native allocation-probe C builds still emit the existing
toolchain attribute/macro warnings; the MoonBit warning-denied check passes.
All release AOT builds and binding steps completed. The compiler decisions
remain pure planning inputs; CUDA register policy stays in CUDA lowering.
No JIT, additional token-step authentication, or diagnostic round trips were
introduced. Temporary stderr instrumentation was used only in a diagnostic
copy and is absent from the tested release source.

Remote campaign root:
`/run/lunaflux-toolchain-4896771-20260913/activation-r3`.
`COMPARISON.json` contains all metrics; `TOKEN_COMPARISON.json` contains sequence
histograms; per-request JSON/SSE and both launch arrays are retained. Earlier
activation-r1/r2 failures remain separate from the successful runtime run.

Downloaded archive SHA-256:
`7b6ad744e03c074b4e92f8f9384826665061bbef3975528c7d69eeec95c1b90c`.
Local copy: `/private/tmp/lunaflux-prefill-activation-20260913.0aPSEo/activation-results.tar.gz`.
It contains source archives, runtime artifacts, request results and isolated
diagnostics; model copies and build caches are excluded.
