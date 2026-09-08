# C8 attention transfer comparison — 2026-09-08

Async transfer is not the missing universal performance switch. With actual
Qwen geometry (16 Q heads, 8 KV heads, dimension 128, page size 8), async Q64
improves synchronous Q64, but synchronous Q32 remains faster on dense C8
queries. Enabling async or always preferring a wider tile would regress those
cases.

Times below are microseconds. Q is query tokens **per request**, context is
also per request, and concurrency is 8. Timed total query tokens are 8×Q.

| Q/context | Sync Q32 (312) | Async Q32 (316) | Sync Q64 (314) | Async Q64 (317) |
| --- | ---: | ---: | ---: | ---: |
| 128/1024 | 1466.28 | 1753.63 | 2078.01 | 1670.90 |
| 128/4096 | 5975.78 | 7120.73 | 8448.33 | 6694.12 |
| 256/1024 | 2643.41 | 3205.90 | 3668.34 | 2947.56 |
| 512/1024 | 4499.07 | 5495.62 | 6202.90 | 5019.98 |
| 512/4096 | 21470.70 | 26035.20 | 29040.10 | 23152.20 |

The full matrix includes per-request query lengths 16, 64, 128, 256, 512,
contexts 512, 1024, 2048, 4096, and four small/tail/ragged cases. Four kernels
ran sequentially on the RTX 5060 Ti with CUDA 13.1.115. Each case used ten
warmups and nine samples of forty launches, measured by CUDA events.
The source uses the existing ordered attention compiler; no math changed.

All 48 paired output files matched bitwise (312/316 and 314/317), all scalar
oracle checks passed, and both async kernels passed memcheck, racecheck,
initcheck, and synccheck on the bounded sanitizer cases. This does not measure
end-to-end serving or compare fresh vLLM/SGLang executions.

## Compiler and dispatch changes

The static Pareto frontier previously discarded async transfer candidates
because its cost model does not credit overlap and async consumes more shared
memory. Static dominance now compares like transfer modes, retaining async
alternatives without selecting them by default. Tests cover actual CUDA AOT
source survival, not only abstract candidates.

Deep partition selection also now rejects a query frontier larger than the
partition count. Previously additional query parallelism reduced the context
threshold and made deep partitioning easier to select, introducing unnecessary
partials/merge work. The new pure predicate preserves sparse, long-context
eligibility. Its serving benefit requires the separate end-to-end A/B test;
the table above measures unpartitioned kernels only.

## Artifacts

- Generator: `tests/attention_tile_cuda_source_probe`, arguments
  `312|314|316|317 qwen`.
- Preparation: `/private/tmp/lunaflux-prefill-qwen-prepare-20260908.mbtx`.
- Runner: `/private/tmp/lunaflux-prefill-pipeline-20260908.mbtx`.
- Remote: `/dev/shm/lunaflux-prefill-qwen-20260908-r3`.
- Downloaded: `/private/tmp/lunaflux-prefill-qwen-results-20260908-r3`.

Earlier disposable harness runs r1/r2 failed because expanding the page-count
limit did not initially expand both host K/V allocations. Those failed outputs
were preserved; r3 fixes both allocations. No production kernel fix was needed
for those harness failures. The inherited final summary string lists the older
query vector; individual case lines record the complete actual vector.
