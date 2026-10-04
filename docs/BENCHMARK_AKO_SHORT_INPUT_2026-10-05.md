# AKO short-input experiment — 2026-10-05

## Decision

No new serving schedule was promoted. Three predeclared decoder alternatives
failed to beat the selected c468 schedule robustly. This is a completed bounded
experiment, **not a performance improvement or proof of optimality**.

The `lunaflux-ako` skill guided the sequence: fresh serving control → selected
kernel trace → fixed-budget paired schedule trials → counters and correctness
→ retain regressions. Production compiler/runtime sources were not changed.
All new automation is offline MoonBit `.mbtx`; no model-name policy, JIT,
qualification scan, or measurement dependency was added to the request path.

## Fresh serving control

GB10 DGX Spark (`spark-368c`), Qwen3-0.6B BF16, GPU UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Serving package:
`/home/wlc004s/lunaflux-ako-pilot-v2-20261004.bfWwShhe/long`.
Its prepared worker identity is
`d3b4d60d1eacb6698209fc14f88061b3a47c1fa9ac0db3c306eee454f9cc77b2`.

Three fresh engine starts, one warmup plus three measured waves per cell per
start. Identical varied token-ID prompts, greedy sampling, fixed output counts,
no EOS shortening. Each measured request retained full token IDs and arrival
times. Wall time is concurrent-wave completion; TTFT and token-gap columns are
medians across individual requests/tokens, not additive wall-time components.

| Input / output | Concurrency | Wall median | Output tok/s | Request TTFT median | Token-gap median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 32 | 1 | 241 ms | 132.78 | 14 ms | 7 ms |
| 128 / 32 | 8 | 302 ms | 847.68 | 42 ms | 8 ms |
| 128 / 32 | 16 | 356 ms | 1,438.20 | 54 ms | 9 ms |
| 512 / 64 | 1 | 480 ms | 133.33 | 22 ms | 7 ms |
| 512 / 64 | 8 | 716 ms | 715.08 | 88 ms | 9 ms |
| 512 / 64 | 16 | 972 ms | 1,053.50 | 143.5 ms | 12 ms |

The C16 128/32 waves ranged from 353 to 359 ms. Compared with the **older
October 4** matched vLLM/SGLang 323/326-ms measurements, 356 ms is approximately
10.2%/9.2% longer. References were **not rerun** in this round; these percentages
are historical comparisons, not a fresh three-engine benchmark.

Accuracy remains separate from timing: only 6/9 C16 128/32 output-vector sets
matched the first measured wave, and 8/9 C8 sets matched. The other four cells
matched 9/9. Fixed token counts permit fixed-work timing, but these observations
do not establish deterministic serving or reference quality parity. Do not
silently widen tolerance or change argmax tie handling to hide the variation.

## Selected route and falsifiable experiment

The ordinary serving worker was retained in the trace. An existing diagnostic
launcher propagated the profiler environment; no marker worker replaced it.
The C16 trace contains the actual c468 owned-eight blockwise F32 decode symbol,
grid `(16,8,1)`, block 64. Short-context routes were already measured; adding a
missing long-context route is not a short-input solution.

The warmup and measured trace together show these major selected paths:

| Selected symbol / geometry | Calls | Profiled total |
| --- | ---: | ---: |
| Gate/up `rows16`, grid 96, block 64 | 1,680 | 101.000 ms |
| Full QKV ingress, grid `(1,32)`, block 128 | 1,736 | 97.491 ms |
| Segmented vocabulary head, grid 1,187, block 256 | 68 | 95.292 ms |
| Owned-eight decode, grid `(16,8)`, block 64 | 1,680 | 78.897 ms |
| MLP down `rows16`, grid 64, block 32 | 1,680 | 58.414 ms |
| Output projection `rows16`, grid 32, block 64 | 1,680 | 51.628 ms |

These are launch-group totals, not mutually exclusive full-chain percentages.
Other prefill/mixed geometries also executed. The profiled measured wave was
360 ms versus the unprofiled 356-ms median; use unprofiled runs for performance
decisions. Gate/up, ingress, head and decode deserve attention before further
short-prefill tuning.

Hypothesis: smaller score ownership or a simpler producer schedule might reduce
short-context overhead. Test exactly three existing compiler-generated AOT
alternatives: c400 serial, c450 single-stage blockwise, c464 four-owner
double-buffered. Keep `--fmad=false`, no reassociation and the existing 0.003
probe tolerance. Numerical laws remain explicitly recorded; changing ownership
or recurrence is not automatically bitwise-equivalent for all inputs.

Each candidate covered rows `{1,8,16}` × history `{128,256,512,1024,4096}` with
five alternating baseline/candidate pairs, actual bucket launch geometry and
independent sampled FP64 attention checks: 45 cells, 225 paired samples. The
selected baseline module and candidate inputs were hashed before/after trials.
No candidate met the predeclared minimum **3% improvement in every pair**.

| C16, history 128 | Selected c468 | Alternative | Decision |
| --- | ---: | ---: | --- |
| Serial c400 | 19.207 µs | 189.415 µs | Regression |
| Single-stage c450 | 19.718 µs | 39.477 µs | Regression |
| Four-owner c464 | 19.159 µs | 20.534 µs | Regression, about 7.2% slower |

Baselines differ between independently paired experiments; do not compare a
candidate against the baseline from another experiment. The complete losing
shape table and all five samples are in `summary.json` and `decode-v2/`.
c464 C16/history512 was inconclusive, not a robust win.

## Why fewer registers did not help

Full Nsight Compute capture compared selected c468 and c464 at C16/history128.
Counter replay uses two launches, 38 passes each, uncontrolled caches. These
counter replay times are **not** the unprofiled timing decision above.

| Metric | c468 | c464 |
| --- | ---: | ---: |
| Registers/thread | 148 | 124 |
| Probe resident blocks/SM | 2 | 2 |
| Waves/SM | 1.33 | 1.33 |
| Warp instructions | 1,664,256 | 1,661,440 |
| Eligible warps/active cycle | 0.3085 | 0.2837 |
| Issue activity | 30.85% | 28.37% |
| Short-scoreboard stall / active issue ratio | 0.8186 | 1.0749 |
| Long-scoreboard stall / active issue ratio | 0.2930 | 0.2937 |
| Barrier stall / active issue ratio | 0.0330 | 0.0337 |
| Local spilling requests | 0 | 0 |
| Shared-load hardware conflicts | 0 | 0 |

The lower-register candidate did not buy another resident block or reduce
instructions materially; short-scoreboard dependency pressure rose about 31%.
This supports rejecting the ownership change. It does not identify a precise
source instruction as the sole cause, nor attribute the whole cross-framework
gap to decode. In particular, neither register count nor bank-count reduction
alone is a useful optimization target here.

## Validation, failed setup and next hypothesis

The first probe setup incorrectly reused the ingress input width 1024 instead
of concatenated QKV width 4096. The independent oracle rejected **all three**
initial trials before timing. Those failures remain under `decode/`; corrected
trials live separately under `decode-v2/`. A regression test now checks this
width conversion. Failed setup output is never used as performance evidence.

Corrected paired probes passed their unchanged oracle/error checks. The c468 /
c464 C16/history128 pair also passed memcheck (zero errors) and racecheck (zero
hazards), with bitwise equality on that input and oracle max error 0.000243713.
Four new automation files passed warning-denied native checks and their four
unit tests. No production kernel/native ABI changed, so no release-wide rebuild
or serving promotion was performed.

GPU jobs were serialized and externally memory/time-bounded. Engine memory
limit was 64 GiB with no swap; host monitoring retained the 32-GiB available
reserve. Probe/sanitizer/counter units were limited to 8 GiB. GPU was idle after
completion; no global counter permission setting was changed.

Next hypothesis should address selected small-row **gate/up and QKV** operand
reuse/dependency chains, with head weight bandwidth separately measured. First
collect matched short-row SASS/source-correlated counters and exact-chain
timings, then change the pure ownership/transport plan or terminal lowering.
The selected decode alternative search above is closed for this budget; do not
repeat it with a success-only stopping condition.

## Reproduction and preserved evidence

- Remote root: `/home/wlc004s/lunaflux-ako-short-20261005.UIQHY3zG`.
- Local archive: `/tmp/lunaflux-ako-short-results-20261005.gnJaluwO/measurement.tar.gz`.
- Archive SHA-256 (remote/local agree):
  `45e5fc0c032d668ef0aeecb1254bc267d58f71c195a66a5035e1c1b28be1c19b`.
- Serving vectors: `control/r{0,1,2}-luna-full/requests/`.
- Trace: `trace/r0-luna-full/trace.nsys-rep`, `trace/kernels.csv`.
- Paired samples: `decode-v2/trial-c{400,450,464}/`.
- Hardware capture: `decode-v2/c464-short-counters.ncu-rep`.
- Compact results: `summary.json`, `counters-summary.json`.
- Drivers: `benchmarks/gpu_pipeline/ako_short_{serving,decode,report,counters}.mbtx`;
  use their argument guards for non-overwriting replay directories. The seal
  extension is in the repository driver; archived counter driver is the
  original measured version, without that later archive-only extension.
