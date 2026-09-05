# Functional single-row dot scheduling — 2026-09-05

Kernel implementation: `2564c31`, based on `ffec385`. This is the first bounded
optimization following the vLLM/SGLang source comparison, not completion of the
attention, batching, autotuning, or asynchronous token-relay work.

## Change

The fused QKV → QKNorm → RoPE → paged KV-write kernel previously used only one
warp for one token. Each lane loaded a different weight row and serially folded
all 1,024 input elements. The new pure compiler schedule assigns output columns
to subgroups, folds contiguous lane-strided products, and combines partials with
a fixed pairwise tree. CUDA lowering uses all four warps, coalesced weight loads,
and subgroup shuffles. The multi-token WMMA path and postprocessing are unchanged.

The schedule is reusable across projection families and subgroup widths. Only
the first consumer is the Qwen-shaped fused CUDA ingress. Model builders and the
scheduler contain no new NVIDIA branches. The change adds no request-time
allocation, tuning, filesystem work, or identity checks.

Floating-point reassociation is explicitly permitted by
`AllowDeterministicDotTree`; it does not follow from purity. Ordered requests
remain the default and preserve existing canonical bytes. The new numerical
policy names the deterministic tree, retaining the existing tolerance ceilings.
The backend compiler still uses `fmad=false` and no further reassociation.

## New versus old kernel

RTX 5060 Ti, CUDA 13.1.115; BF16 Qwen3-0.6B shapes: input width 1,024,
Q/K/V output widths 2,048/1,024/1,024, head dimension 128. The test uses one
packed query row, a nonzero position, reversed page mappings, and eight-token
pages. No other GPU workload ran concurrently.

Each measurement is a GPU-event interval around 64 captured repetitions,
divided by 64. Seven trials interleave old/new and reverse their order every
trial; the table reports medians. Capture is used only to amortize benchmark
launch overhead. It does not establish production CUDA Graph behavior.

| Tokens | Old full ingress μs | New full ingress μs | Old/new |
| ---: | ---: | ---: | ---: |
| 1 | 170.231 | 27.603 | **6.167×** |
| 2 | 58.688 | 58.557 | 1.002× |
| 15 | 75.931 | 75.838 | 1.001× |
| 16 | 48.813 | 48.312 | 1.010× |
| 17 | 65.141 | 65.323 | 0.997× |
| 32 | 60.243 | 59.826 | 1.007× |
| 64 | 77.075 | 77.358 | 0.996× |
| 128 | 114.019 | 113.877 | 1.001× |
| 256 | 210.433 | 210.442 | 1.000× |

Single-token kernel time falls 83.8%. Multi-token results are effectively
unchanged at this measurement precision. This is a hot, repeated single-layer
microbenchmark with the same weight buffers, not rotating all model layers.
**It does not imply a 6.167× serving speedup**, nor a newly measured advantage
over vLLM, SGLang, or llama.cpp. End-to-end Qwen measurements remain pending.

## Correctness and scope

- All nine token sizes match the previous kernel byte-for-byte on dense dyadic
  fixtures, including output tails and untouched KV pages.
- Independently computed CPU dense dots check selected components across all
  eight V heads. Every written K/V component is checked against the output at
  its physical page address; the preceding prefix token remains untouched.
- Dyadic inputs deliberately make FP32 dot products exactly representable.
  This tests topology and addressing, **not general trained-weight tolerance**.
  Independent model-output qualification is still required before deployment.
- CUDA memcheck: zero errors and zero bytes leaked. Racecheck: zero hazards,
  errors, or warnings. Synccheck: zero errors. All four stderr logs are empty.
- Local warning-denied native check passed; the three directly affected
  compiler/fused/probe packages pass 48 tests, including a public-API test.
- The broader five-package selection passes 68/69. Its existing
  `luna_cuda_projection_aot/physical_fixture_wbtest.mbt` recipe-digest assertion
  also fails identically on untouched `ffec385` (actual `f499ebe6…`, expected
  `c00e6391…`). It was not rewritten to mask the failure. An earlier broad
  `-p` substring selection also reached an FP8 fixture panic; no claim is made
  that the entire repository test suite is green.

## Reproduction and retained results

The existing native probe gained two modes:

```sh
moon run --target native --release tests/qwen3_bf16_physical -- ingress-benchmark OLD_CUBIN NEW_CUBIN
moon run --target native --release tests/qwen3_bf16_physical -- ingress-check OLD_CUBIN NEW_CUBIN
moon run scripts/summarize-ingress-benchmark.mbtx BENCHMARK_STDOUT
```

The probe is explicitly fixed to the above shape, launch envelope, page stride,
and test GPU; it is not a generic runtime loader. `ingress-check` uses one
repetition for sanitizer runs. Do not interpret its timings as benchmarks.

- Old CUBIN SHA-256:
  `cbef54174f67c11ae99ea77464e6b1050fd8551482b21cb80343db25edea9fa0`.
- New CUBIN SHA-256:
  `4dcd31411241ed29c3f6e56cceee819cbbff16f9dedd7e333889de14255b5c2b`.
- Remote source: `/tmp/lunaflux-dot-tree-20260905-r2`.
- Generated kernel: `/tmp/lunaflux-dot-tree-candidate-20260905-r1/reusable-qwen-full-ingress`.
- Raw results: `/tmp/lunaflux-dot-tree-results-20260905-r1`.
- Downloaded archive (logs, both sources/recipes and both CUBINs):
  `/private/tmp/lunaflux-dot-tree-results-20260905-r1.tar.gz`.
- Archive SHA-256:
  `72de214f164d8bd36be8d672bbeb01170b7b29257a07ef341c7795bb670c077a`.

The old runtime artifacts and production deployment were not modified.
