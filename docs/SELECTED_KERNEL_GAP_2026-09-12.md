# Selected-kernel comparison: long prefill and ordinary serving

## Scope and method

This investigation follows `OUTPUT_DOWN_TTFT_2026-09-12.md`. The control is
LunaFlux `f2979e2`, with its 2048-token runtime profile, not an older
1024-token profile. Qwen3-0.6B BF16 runs on the same RTX 5060 Ti (36 SMs,
UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`). The actual installed
baselines are vLLM 0.24.0 and SGLang 0.5.2. This does not describe the latest
upstream release of either project.

Three distinct measurements must remain separate:

1. Nsight Systems traces identify the actual kernels, launch geometry,
   execution steps, and where the GPU time goes. Their timings are perturbed
   and are not ordinary serving results.
2. Nsight Compute replays of the selected projection kernels use the same
   matrix dimensions and each baseline's installed cuBLAS library. They
   explain instruction/memory behavior, not full framework throughput.
   Cold-cache profiler time is not interchangeable with warm CUDA-event time.
3. Unprofiled finite-burst serving uses input/output vectors `(59,256)`,
   `(128,128)`, `(512,64)`, `(1528,32)` at concurrency 1/2/4/8/16, one warm-up
   and three timed trials per cell. Identical input token IDs, greedy sampling,
   ignored EOS, and disabled prefix reuse are retained. Engines run serially.

## Is it a batching failure?

The long-input C8 trace accounts for 256 generated tokens and 248 decode
tokens. LunaFlux and vLLM both execute 38 steps/forwards: seven prefill or
mixed steps followed by 31 decode steps. The LunaFlux trace contains five
2048-token mixed steps; requests are genuinely combined.

| Profiled long-C8 quantity | LF control | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Sum of kernel durations, ms | 801.928 | 555.651 | 594.771 |
| GPU busy time, ms | 801.788 | 555.727 | 587.689 |
| First-to-last GPU kernel envelope, ms | 840.494 | 564.418 | 621.952 |

The small differences between summed durations and busy intervals include
overlap and profiler timestamp granularity. LunaFlux's between-step gaps
total 24.764 ms, much less than its 246.277 ms excess kernel duration versus
vLLM. The evidence does **not** support blaming most of this gap on absent
batching or CPU dispatch. SGLang's head-based grouping has two incomplete
groups; its group count is not treated as an exact forward count.

LunaFlux's first 1528-token prefill spends 69.783 ms in kernels. Across 28
layers, its per-call times are attention 781.67 us, gate/up 536.68 us, QKV
381.17 us, down 270.24 us, ingress numerics 243.81 us, and output 200.93 us.
The corresponding first vLLM projection calls are about 265.06, 133.76,
393.40, and 200.03 us for QKV, output, gate/up, and down. vLLM also executes
a separate activation after gate/up. Its first prefill attention averages
296.65 us. These observations locate the expensive work before changing
compiler policy.

## Same math, different instruction cost

Cold-cache selected projection replay at M=1528; duration in microseconds.
QKV is N4096/K1024, output N1024/K2048, gate/up N6144/K1024, and down
N1024/K3072. QKV's dimensions describe the concatenated output, not three
independently timed matrix calls. Gate/up's baseline time excludes its
separate activation; LunaFlux includes it.

| Projection | LF | vLLM | SGLang | LF/vLLM warp-instruction count |
| --- | ---: | ---: | ---: | ---: |
| QKV | 375.744 | 264.128 | 279.840 | 75.579M / 17.787M = 4.25x |
| Output | 200.000 | 133.440 | 147.648 | 33.510M / 8.035M = 4.17x |
| Gate/up | 527.904 | 394.464 | 396.160 | 64.986M / 26.680M = 2.44x |
| Down | 271.520 | 201.376 | 216.736 | 34.561M / 11.623M = 2.97x |

Executed HMMA counts are identical across all three implementations:
3,145,728 for QKV, 1,572,864 for output, 4,718,592 for gate/up, and 2,359,296
for down. Thus extra Tensor Core arithmetic is not the explanation for
these particular long-prefill launches. DRAM read amounts are also close:
about 11.5, 10.5, 15.7, and 15.7 MB respectively. This does not imply identical
L1/L2/shared transactions or instruction schedules.

The vLLM replay selects `64x64_32x6` CUTLASS-family kernels, 128 threads and
88 registers/thread. SGLang's replay selects `64x256_32x4` for QKV/output/down
and `256x128_32x3` for gate/up. These are observed choices, not an assertion
that those tiles are universally optimal. The sibling repositories explain
framework routing; the traces and installed libraries establish what ran.

### What the extra instructions and stalls are

| LF operation | Large executed instruction classes | Remaining measured stalls |
| --- | --- | --- |
| QKV | MOV 9.09M, IMAD 7.52M, IADD 7.09M, BRA 5.24M, WARPSYNC 3.54M | Short scoreboard 13.04%, versus vLLM 2.02% |
| Output | MOV 4.15M, IMAD 3.55M, IADD 3.26M, BRA 2.19M, WARPSYNC 1.77M | Short scoreboard 16.28%, versus 1.10% |
| Gate/up | MOV 10.83M, IMAD 8.87M, LDSM 3.54M, WARPSYNC 3.54M | Barrier 18.23% versus 6.99%; long scoreboard 19.63% versus 2.47% |
| Down | MOV 4.74M, IMAD 3.12M, LDS 2.70M, BRA 2.36M | Short scoreboard 16.24%, versus 0.74% |

Percentages are sampled shares of active-warp stall reasons; they are not
additive portions of wall time or independently recoverable speedups. The
matrix fold still includes transport-address calculation, producer loops,
row predicates, convergence management, and result-layout work. Gate/up's
hot shared store waits for a preceding global read. The correct optimization
target is the complete producer/layout/consumer schedule, not simply fewer
barriers or a prettier bank-conflict counter.

There are no register spills in these selected LF captures. Source-level
excessive shared wavefronts are zero for QKV/output/gate, but down has 896 at
the 1528-row tail. Aggregate hardware conflict-derived counters are nonzero.
The baselines themselves have substantially higher conflict counts while
running faster. These observations supersede any blanket claim that all
shapes are conflict-free or that conflict count alone predicts latency.

## Compiler changes implemented in this iteration

### Independent ingress head maps (`af7d97d`)

The functional compiler now explicitly maps independent head folds to
subgroups. Four heads share a workgroup, while each subgroup owns one entire
head and its terminal writes. The original reduction tree and BF16 rounding
are unchanged. Disjoint shared slices remove cross-head dependencies and
CTA barriers. This is not a Qwen-name dispatch rule.

The CUDA backend chooses the 32-lane implementation. Backend-neutral planning
tests also cover other subgroup widths, partial head groups, and integer
boundaries. `e4142a3` fixes the runtime consumer's stale one-head-per-workgroup
launch check, and tests that grouped partial ingress is accepted while the
full-projection ingress ABI remains unchanged. Legacy pinned launches remain
valid; their arithmetic implementation is not reintroduced into new lowering.

### Shared score maps and invariant placement (`53538ad`)

An attention score's immutable per-lane map is scalarized once and shared
by the max and exponential consumers. Query position, a loop invariant, is
read outside the KV-tile loop. Neither change reassociates the floating-point
fold. The selected compact prefill tile's CUDA lowering stages global data
directly into shared memory, removing the load-to-register/store-to-shared
dependency. The one-stage realization waits immediately; it is **not** a
claim of newly enabled multi-stage lookahead.

### Exact generated-kernel checks

Three alternating warm CUDA-event trials at 1528 tokens, one request:

| Kernel | Previous median, us | New median, us | Time reduction |
| --- | ---: | ---: | ---: |
| Selected prefill attention | 746.882 | 556.911 | 25.4% |
| QKNorm/RoPE/KV-write ingress | 221.186 | 110.666 | 50.0% |

The cold-cache counter runs separately measure attention 783.520 to 561.824
us and ingress 246.944 to 140.064 us. Attention instructions fall from
109.124M to 80.715M and long-scoreboard share from 22.90% to 9.62%.
Ingress instructions fall from 42.568M to 21.733M; CTA-barrier stall share
falls from 26.86% to zero. Both have zero source-level excessive shared
wavefronts and zero register spills. Aggregate conflict-derived counters
remain nonzero. Attention registers rise from 108 to 128, a resource cost
that must remain part of subsequent scheduling decisions.

All 48 paired shape/launch checks are bitwise equal, including whole KV
arenas, varied tails, and grid-stride cases. Shapes span 8–2048 tokens and
1–32 request rows. Eighteen memcheck/leak, racecheck, and synccheck runs pass.
The full local native suite passes 3,709/3,709. Complete-runtime validation
uses the commit-pinned `e4142a3` build; regenerated CUDA sources and recipes
match the physically checked `53538ad` artifacts exactly.

## Complete serving after attention/ingress integration (`e4142a3`)

The new worker now starts and finishes all 20 ordinary cells. The first
attempt failed before readiness because the consumer still required the old
ingress grid. That attempt contains no serving measurements and is preserved
separately. The later capacity-receipt preparation retry also occurred before
GPU serving; it did not modify any model or measured kernel.

Output tokens/s, mean of three measured trials:

| Input/output | C | LF control | LF attention/ingress | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| 59/256 | 1 | 248.46 | 246.95 | 274.53 | 270.25 |
| 59/256 | 2 | 459.19 | 458.79 | 520.15 | 504.32 |
| 59/256 | 4 | 865.11 | 864.63 | 971.85 | 928.10 |
| 59/256 | 8 | 1633.61 | 1628.86 | 1845.05 | 1757.98 |
| 59/256 | 16 | 2642.02 | 2669.58 | 3222.66 | 3144.33 |
| 128/128 | 1 | 240.60 | 239.55 | 267.23 | 256.86 |
| 128/128 | 2 | 444.96 | 444.70 | 495.81 | 477.34 |
| 128/128 | 4 | 825.81 | 825.37 | 929.79 | 901.65 |
| 128/128 | 8 | 1549.21 | 1540.78 | 1739.53 | 1659.72 |
| 128/128 | 16 | 2441.99 | 2483.43 | 2972.42 | 2867.03 |
| 512/64 | 1 | 210.30 | 210.30 | 246.16 | 242.13 |
| 512/64 | 2 | 364.67 | 363.64 | 430.04 | 420.16 |
| 512/64 | 4 | 600.00 | 610.49 | 737.05 | 719.80 |
| 512/64 | 8 | 970.38 | 1001.31 | 1178.82 | 1135.27 |
| 512/64 | 16 | 1269.95 | 1316.20 | 1651.63 | 1599.18 |
| 1528/32 | 1 | 139.94 | 146.12 | 179.50 | 176.81 |
| 1528/32 | 2 | 195.33 | 208.70 | 266.69 | 254.32 |
| 1528/32 | 4 | 248.38 | 267.78 | 362.95 | 344.09 |
| 1528/32 | 8 | 305.49 | 333.77 | 441.89 | 419.44 |
| 1528/32 | 16 | 330.82 | 363.98 | 492.78 | 454.50 |

Long C8 improves 9.26%, not the 25–50% measured for individual kernels.
Its mean TTFT decreases from 354.71 to 312.67 ms (11.85%). Long C16 improves
10.02% and TTFT decreases from 651.92 to 574.46 ms. Short inputs remain
essentially unchanged. The unmodified projection and decode work explains
why a much larger kernel-local improvement becomes a modest whole-request
improvement. This stage still needs 32.4% more throughput to match vLLM and
25.7% to match SGLang at long C8.

All requested output counts are correct. Of 372 matched old/new named
request sequences, 371 are identical; one 59/256/C2 request differs. Paired
kernel equality therefore must not be advertised as whole-serving
batch-invariant numerical equivalence, and this finite-burst throughput
comparison is not an independent model-quality evaluation.

## Finite transfer ownership (`f9df976`)

The compiler now represents a vector transfer as a finite immutable map
from `(owner, round)` to an optional vector index. The CUDA realization
specializes its known extent and owner stride, instead of rediscovering the
workgroup width inside every operand-copy loop. This enables static loop
expansion and address simplification without changing the matrix fold.
The map is backend-neutral; the CUDA source emitter supplies its lane count.
Tests cover non-divisible extents, inactive owners, and integer limits.

Actual generated primary-kernel replay at 1528 tokens/8 request rows gives:

| Operation | Previous/new warm median, us | Previous/new cold profiler time, us | Previous/new warp instructions |
| --- | ---: | ---: | ---: |
| QKV | 367.218 / 346.952 | 375.744 / 356.640 | 75.579M / 54.633M |
| Output | 191.983 / 174.643 | 200.000 / 181.216 | 33.510M / 22.259M |
| Down | 266.628 / 255.939 | 271.520 / 261.440 | 34.561M / 29.403M |

QKV and output register counts fall from 80/87 to 64/64. Down rises from
92 to 103. The new captures have zero spills and zero source-level excessive
shared wavefronts at this shape, including down's previously observed tail
excess. Aggregate conflict-derived counters are still nonzero.

This is a causal improvement, but not the whole explanation: reducing
output's instruction count by 33.6% only reduces its cold duration by 9.4%.
QKV/output still execute 3.07x/2.77x vLLM's instructions; down executes 2.53x.
Their unchanged HMMA counts coexist with remaining result-layout loads,
convergence instructions, and pipeline stalls. Output's new short-scoreboard
share is 14.49%, versus the baseline's 1.10%; down's is 16.07%, versus 0.74%.
The selected gate/up path is unchanged: 64.986M instructions, 18.16% barrier
stall share, and a hot `STS.128` waiting on the preceding global load.

The shape sweep is not uniformly positive. The primary output kernel at
128 tokens/16 rows slows from 18.839 to 20.127 us (6.8%). Primary-module
microcases must not be substituted for the runtime's selected row-bucket
variant or hidden behind an average. Sixty paired cases pass bitwise checks;
45 memcheck/leak, racecheck, and synccheck runs pass. The native suite passes
3,712/3,712.

### Convergence isolation: diagnostic only

A separate experiment removes two row guards from the matrix fold and
output-layout loops while retaining zero-filled masked producers and final
write bounds. All 15 sampled cases are bitwise equal. Beyond static transfer
specialization it improves long QKV about 3%, output about 2%, and down about
5%. However, output at eight rows becomes about 5% slower. This blanket
transformation is **not** integrated. The result argues for explicit interior
and boundary maps with a cost model, not deleting every guard or treating
every `WARPSYNC` as removable overhead.

A second isolation changes the selected gate/up's register-mediated copy to
direct asynchronous staging, then separately moves the next-slot issue ahead
of the current fold. All 22 sampled cases pass bitwise checks. Direct staging
is slightly slower; lookahead improves long rows by only about 1%, while
two/eight/sixteen-row cases become roughly 14–15% slower. Neither experiment
is integrated. A two-slot buffer and an asynchronous instruction do not by
themselves reproduce the baseline's pipelined iterator and consumer schedule.

## Latest complete serving (`f9df976`)

The final unprofiled run uses the exact commit's release worker, exported
kernels, and materialized runtime. It completes all 20 cells and drains.
Output tokens/s, mean of three measured trials:

| Input/output | C | LF control | LF latest | vLLM | SGLang | LF gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 59/256 | 1 | 248.46 | 246.47 | 274.53 | 270.25 | -0.8% |
| 59/256 | 2 | 459.19 | 461.96 | 520.15 | 504.32 | +0.6% |
| 59/256 | 4 | 865.11 | 871.74 | 971.85 | 928.10 | +0.8% |
| 59/256 | 8 | 1633.61 | 1648.95 | 1845.05 | 1757.98 | +0.9% |
| 59/256 | 16 | 2642.02 | 2704.83 | 3222.66 | 3144.33 | +2.4% |
| 128/128 | 1 | 240.60 | 240.60 | 267.23 | 256.86 | 0.0% |
| 128/128 | 2 | 444.96 | 448.87 | 495.81 | 477.34 | +0.9% |
| 128/128 | 4 | 825.81 | 831.63 | 929.79 | 901.65 | +0.7% |
| 128/128 | 8 | 1549.21 | 1562.57 | 1739.53 | 1659.72 | +0.9% |
| 128/128 | 16 | 2441.99 | 2514.95 | 2972.42 | 2867.03 | +3.0% |
| 512/64 | 1 | 210.30 | 209.39 | 246.16 | 242.13 | -0.4% |
| 512/64 | 2 | 364.67 | 369.24 | 430.04 | 420.16 | +1.3% |
| 512/64 | 4 | 600.00 | 616.87 | 737.05 | 719.80 | +2.8% |
| 512/64 | 8 | 970.38 | 1012.53 | 1178.82 | 1135.27 | +4.3% |
| 512/64 | 16 | 1269.95 | 1340.32 | 1651.63 | 1599.18 | +5.5% |
| 1528/32 | 1 | 139.94 | 146.35 | 179.50 | 176.81 | +4.6% |
| 1528/32 | 2 | 195.33 | 209.84 | 266.69 | 254.32 | +7.4% |
| 1528/32 | 4 | 248.38 | 271.38 | 362.95 | 344.09 | +9.3% |
| 1528/32 | 8 | 305.49 | 339.82 | 441.89 | 419.44 | +11.2% |
| 1528/32 | 16 | 330.82 | 370.21 | 492.78 | 454.50 | +11.9% |

Long C8 TTFT is 304.96 ms, down 14.0% from 354.71 ms; the baselines are
225.83/247.17 ms. Long C16 TTFT is 563.23 ms, down 13.6%. The final transfer
map increment adds only 1.8% throughput beyond the intermediate
attention/ingress build at long C8. Small sub-percent differences should not
be treated as established improvements without more repeats.

All 372 matched output sequences in this final run equal the original
control, with the requested lengths and no server error. This does not
explain or erase the intermediate run's one divergence, nor prove arbitrary
batch-invariant numerics or independent model quality. No production runtime
was deployed by this campaign.

The remaining long-C8 throughput increase needed is **30.0% to vLLM** and
**23.4% to SGLang**. Equivalently, current throughput is 23.1%/19.0% lower;
those are different denominators. Short-C8 needs 11.9%/6.6% more throughput.

## Latest trace: where the remaining difference actually is

A new trace uses the `f9df976` worker and generated kernels. Only the parent
has diagnostic environment forwarding and batch markers; these additions
are absent from ordinary serving. Marker coverage and token accounting pass,
with no marker crossing an attributed kernel. The long-C8 request set still
executes 38 steps, 256 outputs, and 248 decode tokens.

Kernel duration summed across the entire long-C8 request set, milliseconds:

| Family | LF control | LF latest | vLLM | SGLang | Latest minus vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention | 310.61 | 261.01 | 189.06 | 174.11 | +71.95 |
| Gate/up plus activation | 147.46 | 147.39 | 125.02 | 131.75 | +22.37 |
| QKV projection | 103.29 | 98.23 | 77.87 | 84.23 | +20.37 |
| Down | 77.08 | 74.67 | 62.37 | 66.20 | +12.30 |
| Output projection | 58.44 | 52.71 | 41.39 | 44.39 | +11.32 |
| Vocabulary head | 28.23 | 28.23 | 28.26 | 29.02 | -0.03 |
| Remaining kernels | 76.82 | 52.78 | 31.69 | 65.06 | +21.09 |
| Total | 801.93 | 715.03 | 555.65 | 594.77 | +159.38 |

Gate/up includes each baseline's separate activation for this aggregate
comparison. Remaining kernels include LF ingress numerics, normalization,
sampling, and miscellaneous work; SGLang also has 2.74 ms of projection calls
that the attribution does not confidently assign. Baseline fusion boundaries
and SGLang's forward grouping differ, so these are request-window totals,
not claims of equal per-call counts or interchangeable operator boundaries.

The latest sum is down 86.90 ms (10.8%); excess kernel time versus vLLM falls
from 246.28 to 159.38 ms. **Attention accounts for 45.1% of the remaining
excess.** LF prefill/mixed attention totals 148.30 ms versus approximately
84.21 ms for vLLM; decode-only attention is 112.71 versus 104.85 ms. About
89% of the attention excess is therefore in prefill/mixed steps, not the
steady decode tail.

The first 1528-token prefill confirms the intended paths actually execute:
attention 559.61 us/call, QKV 355.50, output 178.41, down 258.90, and ingress
124.40; gate/up remains 533.46. Mixed prefill attention still reaches
644–848 us/call on the five 2048-token steps, and the final smaller-grid
continuation takes 318.82 us/call. These measured continuation cases, rather
than only the initial empty-cache microcase, belong in subsequent schedule
experiments.

The 25 steady eight-row decode steps average 7.495 ms of kernels versus
7.552 ms before, less than 1% improvement. Between-step gaps for the whole
long-C8 set remain 24.70 ms versus 24.76 ms before. This explains why neither
the large ingress speedup nor the projection instruction reduction produces
a comparable whole-request gain. It also shows why the vocabulary head is
not the next priority in this particular workload.

## Remaining optimization work

These changes do not establish parity. Selected attention still has a large
gap to baseline prefill attention, and the projection instruction expansion
remains even after static producer maps. The next experiments should isolate
invariant address/segment selection, stage scheduling, and direct accumulator-result
ownership. They must preserve ordered numeric folds and be tested against
small rows, long rows, and masked tails before becoming defaults.

Removing a synchronization instruction without proving lane ownership and
visibility is not an optimization. Similarly, copying a baseline tile name
does not reproduce its pipelined iterators, fragment layout, or epilogue.
The functional compiler needs explicit finite maps and ownership/effect
boundaries so that these optimizations can be derived and checked centrally.

Priority from the latest profile is:

1. Prefill **and continuation** attention: represent and specialize the
   query/key/score maps, validity masks, and accumulator ownership together;
   compare the remaining shared-result traffic and fold-control instructions
   at the measured mixed-step shapes against the selected baseline kernels.
2. Gate/up: eliminate operand-segment selection and address reconstruction
   from the inner copy path, then jointly select staging depth, ownership,
   and epilogue placement. The negative copy/lookahead experiment above
   rules out treating asynchronous copy alone as a sufficient fix.
3. QKV/output/down: exploit proved uniform interior maps and explicit
   accumulator-result ownership; retain safe boundary handling and the
   existing ordered floating-point folds. The guard-removal experiment is
   not a license to remove synchronization or regress small buckets.

These are general compiler transformations and CUDA lowering decisions, not
Qwen-specific arithmetic or a justification for adding arbitrary pass counts.

## Reproduction and retained results

Latest measured source: `f9df97626b709c1518f3c79fdf97f81b694112dd`.
Its exact source archive SHA-256 is
`415abf7e37c20edd1b10a42a19ac803509769f7da80dc042e8f1c85084369cf6`.
The preceding implementation commits are `af7d97d`, `53538ad`, and `e4142a3`.
All run directories are distinct; failed attempts and rejected experiments
remain separate from successful runs.

Local retained directory:
[`benchmarks/results/selected-kernel-gap-20260912.jd5GLb`](../benchmarks/results/selected-kernel-gap-20260912.jd5GLb/).
Large result files are intentionally not committed to Git.

| Archive | SHA-256 |
| --- | --- |
| `lunaflux-selected-gap-20260912-r1.tar.gz` | `46baebb8ad8420bfd96a8c78570954016303a569665ea2cb94489e18ed27a268` |
| `lunaflux-baseline-profiles-20260912-r1.tar.gz` | `80bfc197c32688acb0450888b24614eae17df14621d652e639d1835b85d47da6` |
| `local-analysis.tar.gz` | `ebb057f6fbf28e3e115b4f8fc586c76bba599d54d39902af83f8f9d2261fe067` |

The first archive contains exact sources and release binaries, selected
artifacts, paired checks, sanitizer runs, counter reports, ordinary results,
old/new traces, and diagnosis records. The second retains both baseline
Nsight Systems reports and SQLite exports. The third contains derived
analysis databases/tables, summary JSON, and the MoonBit analyzers. Remote
and downloaded archive hashes match. Numeric model copies and rebuildable
dependency caches are excluded.

One unused 1,192,130,272-byte duplicate numeric payload was removed from the
completed grouped-ingress diagnostic deployment to recover disk space. Its
original remains available, with SHA-256
`17a8674801a38dbc8700d6b2d78fd1b160d5e4c33a3487dc69a1f33bf1838454`;
the restoration record is retained. No production model or result was removed.
