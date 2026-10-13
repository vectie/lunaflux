# GLM native FP4: measured operand-supply bottleneck

## Scope and decision

Implementation `f0f301b2`, on the two GB10 Sparks, changes the operand supply
of the existing native block-scaled FP4 grouped experts. It does not change
model semantics, routing, quantization, MMA association, or TP publication.
Paired standalone chains take 12–27% less time. Two full-model cohorts show
only 2.0–4.3% lower completion time, with variable decode timing; the remaining
reference ratio is 3.08–3.22x. This is a useful kernel fix, not a closed engine
performance gap.
No payload hashes/integrity scans were run. Existing `SHARDS.sha256` files are
read only as previously declared bounded metadata/labels.

The current reference is vLLM's selected FlashInfer CUTLASS W4A4 path, not a
new unselected B12X implementation found in another source checkout. The
relevant architectural lesson is separating coalesced operand producers from
matrix-fragment consumers. This change implements that lesson in our private
CUDA lowering, with existing immutable physical layout/storage IR. Comparison
and profiler dependencies remain outside the compiler.

## Root cause and falsification

The old actual FP4 source loaded register operands immediately before each
64-column MMA. Gate and up each traversed the operand sequence separately.
Four-lane row fragments issued half-sector scalar loads. A large grid also
contained inactive groups, but that was not the main measured cost.

Probe geometry is capacity 64, H=4096, rank-local I=1024, eight routes, with
eight local synthetic experts and nonuniform packed weights/scales. It uses
the generated eight-stage chain and actual launch metadata. Reusing the same
eight expert identities across rows is a diagnostic distribution, not the
full model's measured route distribution.

Frozen baseline: `/tmp/lunaflux-glm-fp4-performance-20261014-v1`.
The date-like suffix is an experiment identifier; the measurements were made
on October 13 CST.

1. **Inactive-grid intervention:** reducing one-row gate/down grids from
   16,384/65,536 CTAs to 256/1,024 saved only about 2% of standalone chain time.
   At 22 rows it regressed in the initial sequential measurements. This is not
   a production host readback or selected launch change. Active-only counters
   still show long-scoreboard dominance (89.7% gate/up, 98.6% down).
2. **Two-slot register lookahead:** output/workspace bytes matched, but paired
   timings were inconsistent and 22-row performance regressed. SASS still has
   load-to-MMA dependencies. Not retained as a selectable production path.
   Root: `/tmp/lunaflux-glm-fp4-performance-lookahead-20261014-v1`.
3. **64-bit loads plus lane shuffles:** generated SASS really used `LDG.E.64`,
   but added shuffles; it did not consistently improve completion time.
   Root: `/tmp/lunaflux-glm-fp4-performance-coalesced-20261014-v1`.
4. **Bulk shared staging:** winner of the three source experiments. Producers
   copy a 256-column chunk with 16-byte asynchronous copies. Both sides bind
   the same row-preserving XOR layout. One publication barrier precedes four
   ordered MMA steps; a reader-retirement barrier precedes reuse. This is
   bulk asynchronous supply, **not** a claim of double-buffered copy/compute
   overlap. Gate/up's duplicated activation traversal is not eliminated.

The first staged export v1 was not executed after a source review caught a
consumer XOR-boundary address mistake. Corrected v2 was measured; the earlier
export remains preserved. Small non-16-byte-aligned scale rows use bounded
scalar scale publication; payload copies remain aligned. This preserves K=64,
128 and other supported tails, instead of silently dropping small shapes.

## Counters on identical one-row workload

Paired report, downloaded without overwrite:
`/tmp/lunaflux-glm-fp4-performance-staged-20261014-v2/paired.ncu-rep`.
Raw tables: `/tmp/lunaflux-glm-staging-analysis-20261014-v1/1-paired-raw.stdout`.
These are isolated kernel counters, not end-to-end service times.

| Metric | Gate/up before | Gate/up after | Down before | Down after |
|---|---:|---:|---:|---:|
| Duration, µs | 401.696 | 249.056 | 175.264 | 142.112 |
| Warp instructions | 6,279,168 | 5,814,272 | 4,767,744 | 5,459,968 |
| Long-scoreboard cycles per issued instruction | 117.04 | 61.40 | 122.76 | 55.51 |
| Barrier cycles per issued instruction | 0 | 14.51 | 0 | 12.58 |
| Registers/thread | 40 | 48 | 40 | 40 |
| Static shared bytes/block | 0 | 13,824 | 0 | 6,912 |
| Tensor-pipe active, % elapsed | 1.27 | 2.06 | 1.46 | 1.80 |
| Active warps, % peak active | 41.16 | 41.63 | 85.47 | 80.05 |

The improvement is reduced load dependency/transaction overhead, not higher
occupancy. Down executes more instructions and acquires barrier/MIO pressure
but completes faster. Tensor utilization remains low; this is not parity with
the reference implementation. The initial unprivileged capture failed with
`ERR_NVGPUCTRPERM`; scoped administrator captures then succeeded without
changing driver settings. Numerical probes were capped at 2 GiB with no swap.

## Alternating unprofiled whole-chain measurements

Five paired trials, 20 complete chains/trial, alternating old/new order. All
trials compare output, weighted intermediates, activation and packed operands
byte-for-byte. This is a schedule-equivalence check against the frozen native
baseline, not an independent full-model quality test.

| Live rows | Old median µs | New median µs | Median-time reduction |
|---|---:|---:|---:|
| 1 | 522.926 | 381.213 | 27.1% |
| 3 | 452.130 | 353.213 | 21.9% |
| 22 | 782.179 | 685.742 | 12.3% |
| 64 | 1,715.978 | 1,468.882 | 14.4% |

Raw trial files are in the staged-v2 root. Run-to-run absolute times fluctuate;
use within-cohort pairs, not cross-cohort absolute baseline medians. Trimmed
grid interventions are retained there separately and are not the selection.

## Correctness and resource checks

- Full-width paired probe: bitwise chain equality at live rows 1/3/22/64.
  One-row full-capacity memcheck/full leak and active-grid race/sync checks:
  zero errors/hazards, explicit CUDA release.
- Independent packed-operand scalar oracle: capacity 64, live
  1/3/16/17/64/0/1, K=64/128, partial F32 publication, zero mismatches;
  memcheck/full leak/race/sync all zero. Root:
  `/tmp/lunaflux-native-fp4-staged-partial-20261014-v1`.
- Source/layout tests 15/15 and physical-tile tests 17/17. Native warning-denied
  check and full suite **5140/5140** pass with the repository's unchanged
  exclusions `-20-25-29-35-79-92`. `moon info`: 3665 existing migration warnings,
  zero errors. Unrelated dirty toolchain/TLS files were not staged.
- Clean source-pinned ARM build:
  `/tmp/lunaflux-checkpoint-build-fp4-staging-20261014-v1`.
- Full runtime artifacts:
  `/tmp/lunaflux-glm-expert-tp-staged-rows64-20261014-v1`.
  Export reports 98,252,360,500 / 99,521,773,364 planned bytes, each below
  100 GiB. Both rank sources select the new expert operand supply.

## Full-model timing and remaining attribution

First full-model cohort completed at
`/tmp/lunaflux-model-bench-glm-staged-rows64-20261014-v1`: six prompt lengths
`[22,23,22,23,24,36]`, output counts `[4,4,8,8,8,3]`, all output IDs identical to
the retained native baseline. One warmup is excluded. Completion times are
`[1.177,1.189,2.083,2.117,2.091,1.099]` seconds, sum **9.756 s**, versus the
retained **10.197 s** (4.3% lower). TTFTs are
`[.502,.513,.499,.531,.544,.644]` seconds. Sum TTFT falls from 3.583 to 3.233 s;
post-prefill time only falls from 6.614 to 6.523 s. Kernel gains therefore do
not imply a similar decode or end-to-end gain. Fresh controls are below.

Fresh vLLM root `/tmp/lunaflux-reference-bench-vllm-glm-staging-20261014-v1`
completed all six literal requests with HTTP 200 after one excluded warmup.
Times are `[.385331,.396595,.656153,.670817,.673605,.381087]` seconds, sum
**3.163588 s** (prior reference 3.151630 s). Current native completion time is
therefore **3.08384x** reference. Four rendered texts agree after excluding the
native stop marker; Chinese/JSON differ. Output counts agree, but the reference
API did not return output IDs. No numerical/model-quality parity is claimed.
Native is TP2/PP1 with BF16 KV; reference is TP2/PP1 with FP8 MLA KV. Native
timing includes CLI diagnostic output/retirement, reference is buffered HTTP.
These short requests cannot establish long-context throughput.

Reference Docker limits were 96 GiB memory and 96 GiB memory-plus-swap (no
container swap allowance); host available memory stayed at least 17.463745 GiB.
After all responses, the runner deliberately stopped both containers. Rank 1
required forced shutdown and exited 137, so the watcher records a terminal
failure; this is not suppressed or described as clean reference teardown.
The logs place the shutdown after the last successful response and retain the
reference's semaphore/shared-memory cleanup warnings. Both containers and GPU
owners were verified gone before the next native experiment.

External report:
`/tmp/lunaflux-model-bench-glm-staged-rows64-20261014-v1/comparison-fresh-reference.json`.

### Repeated old and new native builds

The frozen old build and unchanged new build were then each loaded and run
again, serially, with the same six inputs and excluded warmup. All six output
ID vectors remain identical. Roots are
`/tmp/lunaflux-model-bench-glm-old-repeat-rows64-20261014-v1` and
`/tmp/lunaflux-model-bench-glm-staged-repeat-rows64-20261014-v1`.

| Prompt | Old repeat s | New repeat s | Fresh vLLM s |
|---|---:|---:|---:|
| Fact | 1.387 | 1.156 | 0.385331 |
| Arithmetic | 1.284 | 1.406 | 0.396595 |
| Chinese | 2.159 | 2.165 | 0.656153 |
| JSON | 2.181 | 2.236 | 0.670817 |
| Code | 2.181 | 2.118 | 0.673605 |
| Retrieval | 1.207 | 1.108 | 0.381087 |
| Sum | **10.399** | **10.189** | **3.163588** |

The repeat saves 2.0% total time, while three individual requests regress.
Old/new TTFT sums are 3.779/3.222 s; after-prefill sums are 6.620/6.967 s.
Together with the first cohort, mean sums are 10.298/9.9725 s (3.2% lower),
but two cohorts do not establish a confidence interval or a reliable decode
speedup. Do not discard the losing requests or present the first run as the
only result. Completion-inclusive output throughput is 3.44–3.59 token/s
versus fresh reference 11.06 token/s; these are not decode-only rates.
Repeat report: `comparison.json` in the staged-repeat root.

Both repeated native ranks exit zero with explicit context release and zero
task swap. New-repeat cgroup peaks are 57,114,755,072 / 58,460,778,496 bytes;
minimum host available is 23.090878 GiB. Old-repeat peaks are
56,697,069,568 / 58,796,142,592 bytes. Each is under the 100 GiB ceiling.
Both GPU owner lists are empty after collection. The output and loss records
remain intact; no payload hashing or reference/tolerance edits were performed.

Both ranks report explicit context release, exit zero and zero task swap.
Cgroup peaks are 69,130,846,208 / 68,139,085,824 bytes, below 100 GiB.
Historical/fresh comparison reports are produced by the external summarizer;
it now respects the declared warmup and actual TP selection instead of
mislabeling the current TP2 run as the old PP2 topology.

The attempted six-prompt Nsight Systems trace
`/tmp/lunaflux-model-bench-glm-fp4-counters-rows64-20261014-v1` reached the
16 GiB host reserve guard (11,993,436 KiB available on ingress). The watcher
stopped both ranks; both are inactive. It is **not** a valid timing result or
an OOM claim. Journals/state are preserved in the staging-analysis root. The
100 GiB limit was not increased.

The earlier completed rows16 trace has 132 collectives per step. Per-rank
collective duration is 32.1%/25.6% of summed GPU kernel time, but includes peer
wait and is not pure wire time. The new SQL gap summaries include initialization
and request boundaries; in particular the egress trace includes a 1167 ms
startup gap. They must not be added to steady-state latency or interpreted as
a measured completion-time contribution without range selection. Host
submission/collective readiness still needs a focused, low-overhead causal
intervention. This patch changes neither NCCL progress nor recurrence.

Embedding-to-head range selection removes request boundaries from the egress
trace. The last decode step contains 2,022 compute kernels and 132 collectives:
187.268 ms compute, 96.275 ms collective kernels, and 212.978 ms with no kernel
recorded, totaling 496.520 ms. Of the kernel-free intervals, 181.469 ms precede
collectives. Copies may occupy these gaps; they are not automatically GPU idle
time. This is the **older profiled rows16** run, with substantial observation
overhead, not the current unprofiled decode cost. Do not subtract these values
from the fresh benchmark or assume their whole duration is removable.
Reproducible range queries and outputs:
`scripts/collect-glm-staging-analysis.mbtx`,
`/tmp/lunaflux-glm-staging-analysis-ranges-20261014-v1/1-step-gaps.stdout`.
Ingress has no vocabulary head and consequently no corresponding head-bounded
range; its unbounded totals are not substituted.

Source inspection identifies a concrete remaining scheduling restriction:
`CollectiveKernelProgram::poll` queues consumers after submission readiness,
then waits for `TensorCollectiveOwner::poll` before submitting the next
collective. The latter reaches `lf_device_collective_lease_query`, which calls
`cuStreamQuery`, not a fence immediately following the previous collective.
Thus it also waits for those consumers/next producers to finish. This prevents
ahead-of-execution submission across collective sites. A correct change needs
separate submission progress and lifetime retirement, preserving cancellation
and buffer ownership. Its exact unprofiled latency contribution is **not yet
measured**; do not call summed NCCL time pure communication or claim this alone
explains the remaining ratio.

The reference's actual launch configuration uses full/piecewise CUDA graphs
(`Breakable CUDA graph enabled`, capture sizes 1/2). Our selected TP program
enqueues kernels separately. Chunk KDA is also **not** selected in this cohort:
the existing alternative was measured slower and remains experimental
(`GLM_CHUNK_KDA_2026-10-13.md`). Its triangular algorithm still needs tiled
matrix lowering of causal products and state/value contractions; simply
enabling its flag is not a demonstrated optimization. Neither the graph
difference nor the KDA implementation difference is an additive measured
decomposition of the remaining 3.08–3.22x gap.

The local vLLM source supports this architectural comparison:
`../vlm/vllm/distributed/device_communicators/pynccl.py` submits `ncclAllReduce`
on the current stream and returns without a per-call stream-completion wait;
its explicit synchronization is in initialization warmup. The local
`../vlm/vllm/compilation/breakable_cudagraph.py` records graph/eager segments and
replays their callables in order. These local files explain the mechanism;
they are not claimed to be byte-identical to the installed container. The
installed reference logs independently confirm graph selection, not the exact
time saved by graphs or ahead-of-execution collective submission.

The next causal experiment should separate enqueue progress from GPU lifetime
retirement on the same stream, retaining buffers/communicator until the final
completion fence. Compare that alone before adding graph capture; record
per-step queue depth and boundary gaps with bounded observation overhead. Then
measure graph replay independently. Do not remove cancellation/ownership to
make a benchmark faster, and do not introduce reference-dependent compiler
selection. GPU effect lifetimes belong to the execution plan; arithmetic and
physical operand layout remain independent immutable plans.
