# Output/down operand scheduling and long-input TTFT

## Implemented compiler policy

The ordered matrix fold is unchanged. Its immutable operand window and its
independent result-row product are separate scheduling decisions:

- Long reductions (at least 2048 elements), at most 16 live rows, and at most
  two output tiles use up to sixteen K16 fragments per dense operand transfer,
  or eight for a materialized consumer. The transfer width must divide the
  reduction extent; other products retain four.
- A materialized down-projection consumer uses at most 32 rows per workgroup,
  instead of 64. Its sibling producer, accumulator order, output ownership,
  scalar path, and masking semantics are unchanged.
- Selected vocabulary gathers retain their separately measured window.

These are generic shape/product policies in the functional projection
compiler. CUDA transfer instructions and shared permutations remain in the
CUDA lowering. They do not add request-path allocation, validation, or model
identity branches. The profitability thresholds are conservative measured
defaults, not proof of an optimum for every device or shape.

## GPU measurements before whole-runtime integration

RTX 5060 Ti, UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, CUDA 13.1.115.
The old side is the installed `ed6f572` final-r2 runtime. Three alternating
CUDA-event graph trials compare immutable CUBINs. These are warm repeated
kernel timings, not serving throughput or cold-cache profiler timings.

| Work | Previous, us | New, us | Approximate speedup |
| --- | ---: | ---: | ---: |
| Output K2048/N1024, 8 rows, final K256 window | 15.04 | 12.33 | 1.22x |
| Down K3072/N1024, 8 rows | 17.10 | 12.11 | 1.41x |
| Down, 16 rows | 17.43 | 12.28 | 1.42x |
| Down, 33 rows | 41.23 | 25.79 | 1.60x |
| Down, 512 rows | 106.79 | 100.72 | 1.06x |
| Down, 1024 rows | 209.53 | 183.13 | 1.14x |

Paired output and varied-BF16 checks were bitwise equal with untouched tails.
Six representative launches passed memcheck (including full leak checking),
racecheck, and synccheck: 18 successful runs. They cover output-8 and
down-8/16/17/33/1024. Source-level excessive shared wavefronts and register
spills are zero in four new profiler captures; aggregate hardware shared
conflict totals are **not** universally zero.

Cold-cache new-kernel captures measured output-8 at 17.888 us, down-8 at
20.864 us, down-16 at 20.384 us, and down-1024 at 188.032 us. For context,
the previous report's vLLM output-8/down-8 captures were 13.152/20.256 us.
That comparison is not a fresh paired baseline run: down is close, output
still has a gap. New output-8 DRAM throughput was 53.81%, with long-scoreboard
stall share 17.42%; down-8 reached 69.27% and 37.45%, respectively.

A follow-up dense-only K256 window reduces output-8 further to 12.325 us
warm and 16.640 us cold. It keeps varied-BF16 bitwise equality and clean
memcheck/racecheck/synccheck results. Source-level excessive shared wavefronts
and spills remain zero; aggregate hardware counters remain nonzero. Cold DRAM
throughput is 58.06%, instructions 516,736, registers 95, and total shared
storage 34,816 bytes. The materialized down window is deliberately not doubled.

## Rejected experiments

The broad K128-window experiment regressed short-reduction QKV by roughly
47% and the wide-output C16 map by roughly 50%. Its large matrix window also
produced an unlaunchable static/shared resource combination. A 64-row output
product was slower. Narrowing large output maps to two or four groups also
regressed the 512/1024-row workloads. A two-group, 32-row, K128 down product
improved tiny shapes but substantially regressed large shapes. None of these
regressing policies is enabled in the submitted compiler change.

## Fresh old-runtime phase trace

The unprofiled old-runtime repeat reproduced 1528/32/C8 at 298.25 output
tokens/s, mean TTFT 499.29 ms, mean decode interval 11.19 ms. This confirms
the previous roughly 501 ms TTFT observation.

A separate diagnostic parent (same worker/model/kernels) records actual
submitted rows, tokens, prefill/decode counts, and outputs. Its Nsight Systems
trace accounts for every marker without crossing kernels and exactly 256
outputs / 248 decode tokens in the long-C8 cell. Profiling timings are not
used as ordinary performance results.

The eight initial, single-request, 1024-token prefill steps average 43.01 ms
GPU span, of which 42.65 ms is kernel duration. Per step, attention is 10.79 ms,
gate/up 10.13 ms, QKV 7.25 ms, down 5.76 ms, ingress numerics 4.06 ms, and
output projection 3.78 ms. Later mixed steps take about 59–62 ms. Thus
Output/down improvements alone cannot close the long-input TTFT gap; this
trace does not support CPU dispatch being the dominant prefill bottleneck.

The phase trace also exposes delayed final-row demand: eight initial prompt
chunks produce no output. Unlike vLLM's running-before-waiting selection and
SGLang's existing-chunk-before-waiting selection, LunaFlux considered waiting
prefills before continuing resident ones. Equal-step age ties also defaulted
to the waiting queue. A separate scheduler change resumes resident prompt
folds first, after the existing age-priority and decode passes, and uses request
generation for equal-step age ties. It changes inter-request work order, not
per-request KV effects or arithmetic. Its performance must be measured
separately from the kernel-only improvement.

## Whole-runtime experiments

All serving results below are unprofiled on the same GPU and Qwen3-0.6B BF16
weights. The input/output-token vectors are `(59,256)`, `(128,128)`,
`(512,64)`, and `(1528,32)`, each at concurrency 1/2/4/8/16. Each cell has
one warm-up and three measured trials. Prompts are identical token IDs;
generation is greedy with EOS ignored and prefix reuse disabled. Engines run
serially. Both baselines were rerun on 2026-09-12: vLLM 0.24.0 and SGLang
0.5.2, the latter with graph maximum batch 32. These are the installed
versions, not a claim to test the newest available upstream versions.

The stages are deliberately separate:

1. `ed6f572`: fresh original-runtime repeat, 1024-token step budget.
2. `d0a678f`: integrated K128 operand-window/down-row change, same budget.
3. `2478297`: final K256 dense window plus resident-prefill continuation,
   still the same budget.
4. `f2979e2`: bounded fused-profile support extended from 1024 to 2048,
   with a regenerated 2048-token runtime profile. This is a capacity setting,
   not an additional arithmetic optimization or model-specific compiler rule.

The third stage improves long-C8 mean TTFT from 499.29 to 360.63 ms, but its
throughput is 295.61 rather than the original 298.25 output tokens/s (and
the kernel-only stage's 306.10). Its per-request mean decode interval rises
from 11.19 to 15.36 ms: earlier first tokens coexist with the remaining mixed
prefill work. This is a latency/throughput tradeoff, **not** a free speedup or
evidence that steady-state decode kernels became slower.

The first attempt to build a 2048 profile on `2478297` aborted at candidate
export, before GPU execution. The cause was the fused-profile validation
ceiling, not a numerical kernel failure. `f2979e2` raises that bounded ceiling
and tests acceptance through 2048 and rejection at 2049. The failed export
directory is retained separately from the rebuilt runtime.

The final 2048-budget profile recovers that throughput loss: long C8 reaches
305.49 output tokens/s and 354.71 ms mean TTFT. Relative to the fresh original,
TTFT is 28.96% lower, but whole-batch throughput is only 2.43% higher (about
858 to 838 ms to finish the 256 output tokens). Earlier first output must not
be represented as a 29% reduction in total computation or completion time.
Against the fresh baselines, long-C8 TTFT remains 1.57x vLLM / 1.44x SGLang;
their output throughput remains 1.45x / 1.37x LunaFlux.

For deployment reproduction, the 2048 setting requires an exported and
compiled `max_query_tokens=2048` release. The existing launch materializer
derives `step_token_budget` and `prefill_chunk_tokens` from that release bound.
This report does not recommend editing a 1024-bound runtime's policy in place,
nor does this commit globally change deployment capacity defaults.

### Full ordinary throughput matrix

Output tokens/s; higher is better. Numbers are means of the three trials.

| Input/output | C | Original LF | LF, 2048 budget | vLLM | SGLang | LF change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 59/256 | 1 | 246.71 | 248.46 | 274.53 | 270.25 | +0.71% |
| 59/256 | 2 | 414.69 | 459.19 | 520.15 | 504.32 | +10.73% |
| 59/256 | 4 | 789.31 | 865.11 | 971.85 | 928.10 | +9.60% |
| 59/256 | 8 | 1499.27 | 1633.61 | 1845.05 | 1757.98 | +8.96% |
| 59/256 | 16 | 2523.77 | 2642.02 | 3222.66 | 3144.33 | +4.69% |
| 128/128 | 1 | 240.75 | 240.60 | 267.23 | 256.86 | -0.06% |
| 128/128 | 2 | 401.90 | 444.96 | 495.81 | 477.34 | +10.71% |
| 128/128 | 4 | 754.79 | 825.81 | 929.79 | 901.65 | +9.41% |
| 128/128 | 8 | 1419.62 | 1549.21 | 1739.53 | 1659.72 | +9.13% |
| 128/128 | 16 | 2326.40 | 2441.99 | 2972.42 | 2867.03 | +4.97% |
| 512/64 | 1 | 209.85 | 210.30 | 246.16 | 242.13 | +0.22% |
| 512/64 | 2 | 332.51 | 364.67 | 430.04 | 420.16 | +9.67% |
| 512/64 | 4 | 558.55 | 600.00 | 737.05 | 719.80 | +7.42% |
| 512/64 | 8 | 902.47 | 970.38 | 1178.82 | 1135.27 | +7.52% |
| 512/64 | 16 | 1202.82 | 1269.95 | 1651.63 | 1599.18 | +5.58% |
| 1528/32 | 1 | 135.79 | 139.94 | 179.50 | 176.81 | +3.06% |
| 1528/32 | 2 | 187.69 | 195.33 | 266.69 | 254.32 | +4.07% |
| 1528/32 | 4 | 242.12 | 248.38 | 362.95 | 344.09 | +2.59% |
| 1528/32 | 8 | 298.25 | 305.49 | 441.89 | 419.44 | +2.43% |
| 1528/32 | 16 | 323.64 | 330.82 | 492.78 | 454.50 | +2.22% |

### Long-input latency

Mean TTFT in milliseconds; lower is better. Input/output is 1528/32.

| C | Original LF | LF, 2048 budget | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: |
| 1 | 87.00 | 80.00 | 55.67 | 55.00 |
| 2 | 151.67 | 118.50 | 77.17 | 82.00 |
| 4 | 267.00 | 203.50 | 130.17 | 144.33 |
| 8 | 499.29 | 354.71 | 225.83 | 247.17 |
| 16 | 942.54 | 651.92 | 411.88 | 460.10 |

Mean client-observed milliseconds per decode token, **including interference
from remaining prefill work**, not isolated steady-state device-step time:

| C | Original LF | LF, 2048 budget | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.72 | 4.73 | 3.88 | 3.98 |
| 2 | 6.02 | 6.60 | 5.11 | 5.33 |
| 4 | 8.28 | 9.71 | 6.92 | 7.20 |
| 8 | 11.19 | 14.91 | 10.84 | 11.60 |
| 16 | 19.49 | 27.31 | 18.96 | 21.19 |

## Validation and scope

The working-tree native suite passes 3703/3703, including unrelated local work
that was not included in the benchmark source archives. Exact committed
source is archived separately for every runtime build; no dirty working-tree
changes enter those GPU builds. Focused compiler/CUDA-generator tests and the
scheduler allocation boundary pass. The scheduler change adds no heap
allocation or storage to the token-step selection path.

The exact `f2979e2` source passes Linux `moon info`, `moon fmt --check`, native
warning-denied check, and **2982/2982** native tests after repairing the single
zero-byte generated cache file described below. The different local/remote
test totals reflect the excluded unrelated working-tree packages, not missing
tests within the committed source.

For the installed `2478297` runtime, 55 packaged-kernel paired checks and
55 diverse-BF16 checks pass. Ten additional 1528/2048-token comparisons for
`f2979e2` match a reference constructed from separate old, bounded row-map
launches bitwise, with finite results and untouched output tails.

The 1024-budget final runtime matches 367/372 named measured output sequences
from the old runtime, including all 93 long-input sequences. All 496 sequences
including warm-ups occur in the old same-prompt output pool. That is not a
claim of batch-invariant generation or independent model-quality validation.
An interim `d0a678f` short-input sequence did not occur in that pool and is
preserved in its comparison report rather than silently discarded.

The final 2048-budget runtime matches 369/372 named measured sequences,
including every long-input sequence; all 496 outputs including warm-ups are
in the old same-prompt pool. The three named mismatches are at 512/64/C4 and
are reported explicitly in `token-comparison.json`.

Remote test infrastructure exhausted `/dev/shm` during the full native test
build. This also interrupted sanitizer log writing. The failed records remain
separate. Two completed-campaign copies of the numeric weights were removed
only after matching the preserved original's SHA-256 and checking that they
were not in use; they are recoverable from that original. The current runtime,
CUBINs, sources, and measured logs were not removed. An incremental retry then
exposed a zero-byte generated `model_startup.blackbox_test.core` left by the
out-of-space build, rather than a source change in that package.

The first seven enlarged-shape cases have all three clean sanitizer results in
the original directory; the final three are rerun in a new tail directory.
Together these cover QKV/output/gate-up/down/head at both 1528 and 2048 tokens,
with memcheck (full leak checking), racecheck, and synccheck. The interrupted
directory is not relabeled as a completed campaign.

No production deployment is performed or claimed by this report. Source-level
shared-wavefront checks and hardware aggregate conflict counters are different
measurements; the latter are still nonzero in some kernels. Remaining long
prefill compute includes attention, gate/up, QKV, and ingress numerics; fixing
Output/down alone cannot erase the full baseline gap.

## Reproduction records

Code commits, in order: `d0a678f`, `058b521`, `2478297`, `f2979e2`.
The final tested source archive SHA-256 is
`91686ce3b4b7d7c0157cc4a6530ed54a4fdf11481fe566402933d16ec7e37cbb`.

The combined archive `lunaflux-output-down-20260912-r1.tar.gz` has SHA-256
`76b12f06b6e1436977978f19ca6d7f12841067599f19d3b45881a2f4eb187628`.
It contains exact source archives, release binaries, compiled kernels,
runtime configurations, ordinary runs, kernel experiments (including rejected
ones), counters, sanitizer logs, validation failures and successful retries,
and the offline MoonBit drivers. Numeric weight copies and build caches are
excluded; their source/digests remain recorded.

Fresh baseline profiling is a separate archive with SHA-256
`80bfc197c32688acb0450888b24614eae17df14621d652e639d1835b85d47da6`.
Its profiler-perturbed timings are not used in the ordinary tables above.
Local downloaded records and derived tables are under
`benchmarks/results/output-down-20260912.z4qGmB/` (git-ignored).
