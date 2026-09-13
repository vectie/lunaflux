# Functional pipeline implementation

Scope: the five measured follow-ups in
[GPU_PIPELINE_MEASUREMENTS_2026-09-13.md](GPU_PIPELINE_MEASUREMENTS_2026-09-13.md).
The target is less total request time, not an instruction-counter target.

## Full-path activation follow-up

The user has approved enabling the new ownership/read-view path across the
exported prefill family, prioritizing long inputs rather than retaining the
old schedule for short-query regressions. The implementation will carry an
explicit portable ownership requirement through single and partitioned
compilation, preserve that requirement during bucket specialization, and emit
CUDA register policy from the selected schedule. Dense current K/V remains a
read optimization only: persistent cache writes and history reads are retained.
Validation is an isolated complete Qwen runtime comparison on long input/output
vectors, plus the affected compiler tests and generated-kernel sanitizers.
This authorizes benchmark runtime preparation, not a production deployment.

## Ordered work

1. Partially evaluate concatenated GEMM operand iteration into disjoint typed
   segments. Preserve vector ownership, ordered arithmetic, zero-filled tails,
   and the producer/publication fence. Compare the generated instructions and
   all row buckets, not only the largest matrix.
2. Express attention row/fragment ownership across QK, softmax, and PV in the
   schedule. Lower compatible ownership to register values; keep an explicit
   numerical contract for changed reduction trees.
3. Make query-tail work and workgroup waves visible to scheduling. Measure
   127/128/129/130 queries independently of 4095/4096/4097 history lengths.
4. Represent current dense KV and historical paged KV as ordered read views.
   Do not change persistent KV write ownership or infer contiguity from page
   numbers. Test prefix, no-prefix, mixed rows, and fragmented pages.
5. Compare complete schedules using resource residency, rounded waves,
   padded work, and transfer volume; retain exact offline measurements as the
   final selector. Missing register/resource observations are unknown, not zero.

## Validation

Each implementation needs affected native tests and generated-source checks.
Kernel changes additionally need independent numerical comparisons,
memcheck/racecheck, selected-kernel timing, and final end-to-end Qwen tests.
Use new output directories and only one GPU workload at a time. Preserve the
old and new selected artifacts for comparison. No production deployment.

The previous C2 output disagreement is an unresolved correctness observation;
performance does not discharge it. A twofold speedup over vLLM is a research
target, not an established consequence of these changes.

## Progress

The five compiler mechanisms are implemented. This is **not** a claim that
every production bucket now selects the new kernel, or that the half-vLLM
request-time target has been reached.

| Mechanism | Implemented boundary | Selection/status |
|---|---|---|
| Segmented GEMM transfers | Pure disjoint vector-owner segments; source address and tail validity hoisted outside the ordered fold | Sibling GEMM lowering uses it; final measured latency is essentially neutral |
| Cross-operator ownership | Generic query-owner schedule; register QK/softmax/PV CUDA lowering; explicit numerical identity | Q32/K32 and Q64/K64, head64/head128; requires alternative-softmax permission |
| Query tails and waves | Inactive query owners skip arithmetic without skipping CTA fences; smaller query tiles; rounded-wave resource score | Orthogonal query/history boundary matrix passes; no universal tile winner |
| Current/history KV views | One immutable logical stream; shared sync/async address lowering with checked current-position substitution | Implemented and tested, but not enabled as a blanket production default |
| Resource feedback | Immutable device budget, register allocation granularity, per-schedule counts; single/partitioned AOT selection | Static score is a ranking proxy; exact offline latency records still win |

The strategy file was split into candidate specifications, legality/generation,
cost, and selection. These files are now below 500 lines. No model-name branch,
runtime JIT, profiling call, or new request-path filesystem/cryptographic work
was added.

## Physical results

GPU: RTX 5060 Ti, UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`,
PCI `00000000:17:00.0`; CUDA 13.1.115. `nvcc` SHA-256:
`6ce80365abc2f14ff5f9e069fe687a93d4eaacc555dbf1d1696ee899d39df52a`.
One workload used the GPU at a time; production was not changed.

Final attention campaign: `followups-r8`. Timings are the median of five
paired CUDA-event trials, alternating execution order, each with 30 timed
launches after warm-up. Baseline is the installed selected compiler kernel
from `runtime-final-r2/fused/prefill.cubin`; it was built with
`--maxrregcount=128`. Both variants use `-O3 -arch=sm_120 --fmad=false`.
The **complete schedule** comparison below deliberately includes each new
kernel's stated register policy; it must not be attributed solely to ownership.

`Q` is total packed query tokens, `R` is packed rows, `H` is base history length.
Ragged multirow fixtures additionally vary history by `row % 3 * 8`.

| Q / R / H | New schedule | Old µs | New µs | Time reduction |
|---|---|---:|---:|---:|
| 32 / 1 / 0 | 319, default register allocation | 22.56 | 16.42 | 27.2% |
| 1528 / 1 / 0 | 318, default register allocation | 558.17 | 453.95 | 18.7% |
| 1528 / 8 / 0 | 318, default register allocation | 161.79 | 137.19 | 15.2% |
| 1528 / 8 / 4096 | 318, default register allocation | 2743.43 | 2497.16 | 9.0% |
| 128 / 1 / 4096 | 318, default register allocation | 256.43 | 371.59 | **−44.9%** |
| 129 / 1 / 4096 | 318, default register allocation | 401.28 | 375.02 | 6.5% |
| 1528 / 1 / 0 | 319, default register allocation | 558.98 | 758.09 | **−35.6%** |
| 1528 / 8 / 0 | 316, dense-current, register cap 128 | 161.86 | 195.30 | **−20.7%** |

Matched-register-cap controls are important. At cap 128, schedule 318 reduces
1528/R1/H0 time by 10.2% and 1528/R8/H0 by 12.9%, but is neutral/slower with
4096 history. Schedule 313 at the same cap tracks the old kernel within about
1% around the query-tail boundary. The initial uncapped-control regression was
**not** evidence that a tail guard caused a slowdown: compile flags differed.

The new register path smooths the 128→129 discontinuity (371.59→375.02 µs),
but it does so partly by being slower at 128. That is not a solved performance
problem. Keep the old schedule for that bucket until a measured replacement
wins. Dense-current reads likewise do not automatically beat the paged cache.

### Selected-kernel counters

`followup-counters-r2`, Q1528/R8/H0, cold replay (not the warm timings above):

| Metric | Old | Query-owned, cap 128 | Query-owned, default |
|---|---:|---:|---:|
| Executed warp instructions | 25,728,044 | 10,549,878 | 10,598,584 |
| Registers/thread | 128 | 128 | 168 |
| Local spilling requests | 0 | 412,224 | 0 |
| Tensor pipe activity | 22.23% | 27.59% | 28.59% |
| Cold replay duration | 168.29 µs | 150.14 µs | 144.32 µs |

The ownership rewrite removes about 59% of executed instructions in this case.
Forcing an equal register count introduces spill traffic, so a register cap is
a tuning dimension, not a universal optimization. Residency alone remains
insufficient: measured spill and issue behavior must accompany the cost model.

### GEMM iterations

The first segmented lowering was slower; the first pointer-hoisted version
still executed 60,949,248 instructions versus 59,824,896 for the old kernel,
with no spills. The final version separates invariant validity predicates
from nullable source pointers. `projection-followup-r3` passes bitwise
workspace comparison and an independent sampled scalar reference for
`[1,8,16,32,64,128,129,256,1024,1528,2048]` tokens.

Final medians include 510.42→510.19 µs at 1528 tokens and
687.46→684.38 µs at 2048; the full vector ranges from −0.6% to +0.45% time
reduction. Treat that as **performance-neutral**, not a demonstrated GEMM
speedup. The r2 instruction counters are not counters for the final r3 code.
This transfer pass is currently connected to the sibling gate/up fold; it
does not establish new QKV/output/down improvements.

## Verification and remaining integration

- Warning-denied native check and `moon info` pass.
- Clean archive of implementation commit `bad310e`: warning-denied native
  check and **212/212 affected-package tests pass**, without the unrelated
  untracked model-family sources in the shared workspace. Regenerated attention
  sources/probe match the physically tested r8 files byte-for-byte.
- Workspace full native suite: **3,728/3,728 pass**. An earlier whole-suite run
  hit the unrelated zero-wait TCP timing test; its focused rerun and the final
  full suite pass. No service files were changed for this work.
- Project-wide `moon fmt --check` still reports existing trailing-comma
  differences in unrelated benchmark packages. The changed compiler files
  were formatted individually; those unrelated files were preserved.
- `followups-r5` and `r6`: 48 paired cases and 24 sanitizer runs each;
  fragmented physical pages, current/paged views, sampled scalar reference.
- Final `followups-r8`: **52 paired cases and 12 sanitizer runs pass**;
  includes Q `[127,128,129,130]` × H `[4095,4096,4097]` independently.
- `followup-coverage-r2`: four executables covering head64, head128, and
  partitioned partial/merge pass independent references and all twelve
  memcheck/racecheck/synccheck runs. The r1 partitioned probe exposed missing
  generated metadata constants; source emission was fixed and r2 rerun passed.
- Final sibling GEMM: eleven token cases and three sanitizer runs pass.

Remaining production work is explicit: populate compile-flag-specific offline
records for complete serving buckets, wire the winning permitted attention
variants into Qwen's exported runtime, and rerun end-to-end correctness/latency.
The existing C2 output discrepancy must be resolved there. Dense-current and
register attention are **not** silently enabled on old production contracts;
the strict numerical default and paged view remain unchanged. Further QKV,
output and down pipeline speedups are also still required. No new comparison
against live vLLM/SGLang or whole-serving speedup is claimed here.

## Reproduction and saved output

First-party automation is in `benchmarks/gpu_pipeline/*.mbtx`; C++ files are
isolated GPU diagnostic fixtures, not production runtime dependencies.
Use `prepare_views.mbtx` / `run_tail.mbtx`, `prepare_coverage.mbtx` /
`run_coverage.mbtx`, and `prepare_projection.mbtx` / `run_projection.mbtx`.
The preparers require a new directory and the runners retain commands, stdout,
stderr and exit status. Run preparers from the repository root.
`prepare_views.mbtx` uses the committed `attention_followup.cu` replay fixture
by default and accepts an optional prior fixture path. `prepare_projection.mbtx`
defaults to baseline commit `daafa52` and accepts an explicit baseline revision;
it must not silently compare the current commit against itself.
Profiler access used the approved elevated `ncu` invocation; no driver policy
was changed. The initial unprivileged permission failure is preserved.

Archive `followup-results-20260913-r8.tar.gz` was downloaded and its SHA-256
verified locally:
`5fb6a637c9e6c9b41f76e0d3197ec1c85b7cf1f7f5e88422ec10fa584b904b45`.
Local extraction: `/private/tmp/lunaflux-counters-analysis.MkHuxU`.
Remote root: `/run/lunaflux-toolchain-4896771-20260913`.
