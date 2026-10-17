# Bulk expert transport and persistent consumers

Continuation of `GLM_EXECUTION_REPAIR_2026-10-17.md`. Gate/up remains about
74 ms in the selected warmed prefill; the previous buffering/read-ahead
campaign did not materially change it. All other open families remain in that
report's ledger; this document does not close them by implication.

First hypothesis: issue granularity and work reuse must change together.
Test at most three coupled schedules against the frozen actual source:
packed byte-preserving rows, byte-counted bulk row transfers, explicit
producer/consumer epochs, wider independent output ownership and bounded
persistent workers. Start with 64/128 columns and one/two consumer teams.
This is bulk **row** transfer, not descriptor-based multidimensional TMA.
Its unswizzled consumer layout is a measurable tradeoff, not presumed faster.
If this loses, inspect transaction/instruction/dependency evidence before
adding another buffering variant. Descriptor-based tensor transfer remains a
distinct alternative with startup-owned descriptor lifetime requirements.

The numerical program is unchanged: existing quantized bytes/scales, ascending
K64 matrix folds, and declared BF16 publication boundaries. Physical plans own
the worker/consumer cardinalities; CUDA lowering owns instructions and barriers.
Comparison, counters and selection evidence stay outside the compiler.

Require exact complete-chain outputs, tails, invalid/padded rows, memory/race/
synchronization/leak checks, five alternating timing pairs, and actual serving
selection before claiming a model gain. Include packing and quantization time.
Keep all failures and losing shapes. The retained vLLM component uses different
physical padding; a component comparison alone does not explain the full-model
completion gap.

Original GLM remains running during source work. GPU tests temporarily stop
only its head/worker, leave NFS intact, and restore the original containers.
Every task owner stays within 100 GiB, no swap, independently observed host
MemAvailable at least 16 GiB. No payload hashing or integrity scans.

## First three candidates: rejected for selection

All retain exact full-chain output and packed payload/scales on 1/3/22/64 rows,
both capacity and diagnostic trimmed grids. Roots are
`/tmp/lunaflux-glm-fp4-performance-bulk-20261017-v{1,2,3}`. At 22 rows the
capacity chain changes approximately 6.76→10.38 ms (64 columns/one consumer
team), 6.78→10.25 ms (128/two teams), and 6.76→10.01 ms (padded rows).
These are regressions, not production winners. No full-model gain is claimed.

V3 instruction-level SourceCounters demonstrate the mechanism: varying row
addresses within the producer warp lower to repeated ELECT, R2UR.BROADCAST,
UBLKCP and BRA.U.ANY. Issuing a bulk operation per lane is serialized. Consumer
readiness polling dominates its extra executed instructions (three polling
instructions each execute 120,147,483 times in that instrumented gate/up
capture). SourceCounters counts differ from the shorter metrics capture;
polling depends on profiling perturbation. Do not combine those captures.
In the metrics capture, gate/up is 6.253 versus 4.194 ms, 251.0 versus 96.5M
instructions; L2 read sectors fall 36.49→27.85M and shared-load conflicts
1.270M→0.317M. Layout alone is not the answer.

New hypothesis, bounded to two executable schedules: represent each operand
as contiguous physical tiles, preserving every encoded byte and scale, so one
elected producer issues a whole-tile transfer. Transform immutable weight
storage once at startup using one-expert scratch, never each token; activation
packing remains an explicit effect included in timing. This is an alternative
to tensor-descriptor TMA, not a claim to have implemented that API. Its added
startup cost, scratch ceiling, exact permutation, launch propagation and whole
chain must all be checked. Persistent workers and numerical K order remain
independent immutable plans. Retain the failed row-transfer experiments.

## Contiguous tiles: demonstrated component benefit, serving test pending

The first whole-tile schedule succeeds. Weight payload and scales are permuted
bijectively once at startup through one-expert scratch; activation packing uses
the same physical-domain plan. One elected producer issues six whole-plane
copies per paired gate/up K epoch. Persistent 48 workers, two consumer teams,
two slots and the existing ordered K64 arithmetic are explicit schedule choices,
not new model semantics. Ordinary source/default selection is unchanged.

Root `/tmp/lunaflux-glm-fp4-performance-tiles-20261017-v2` compares the frozen
pre-lookahead source with this candidate, using all 288 distinct expert banks,
five alternating pairs of 20 iterations, and including activation packing and
quantization in the chain:

| Capacity workload | Retained median | Whole-tile median | Time reduction |
| --- | ---: | ---: | ---: |
| 1 row | 361.33 us | 301.43 us | 16.58% |
| 22 rows | 6.7586 ms | 5.2136 ms | 22.86% |
| 64 rows | 13.1398 ms | 10.1793 ms | 22.53% |

All eight 1/3/22/64-row capacity/trimmed cases match complete outputs and the
independent packed payload/scale oracle bitwise, with no nonfinite values.
V2 passes memcheck/full leak, racecheck and synccheck. V3 repeats those gates
using the actual one-expert scratch plus copy-back startup helper, rather than
a separate whole-bank destination; all three pass without swap. These probes
exercise three live rows with all distinct banks, not every production shape.

Separate matched metrics capture at 22 rows:

| Metric | Retained gate/up | Tile gate/up | Retained down | Tile down |
| --- | ---: | ---: | ---: | ---: |
| GPU time (ms) | 4.188 | 3.162 | 2.161 | 1.632 |
| Warp instructions (M) | 96.530 | 113.701 | 49.418 | 66.889 |
| L2 read sectors (M) | 36.488 | 24.384 | 18.230 | 12.948 |
| Active warps (%) | 45.89 | 18.75 | 81.10 | 18.75 |
| Long-scoreboard contribution (%) | 77.74 | 37.18 | 73.34 | 55.73 |
| Barrier contribution (%) | 19.67 | 1.62 | 13.06 | 6.14 |

The faster candidate issues more total instructions and has lower occupancy.
The causal change is coarser operand delivery with reusable physical tiles and
separate producer/consumer ownership, reducing traffic and dependency waits.
Neither occupancy nor instruction count alone explains completion time.
Shared-load bank counters remain nonzero; zero conflicts is not this objective.

This does not yet explain the complete 42.12% model gap or establish a model
speedup. The retained production trace puts gate/up, down and shared experts
ahead of the sparse-attention leaf; collective duration also includes peer
waiting. The installed CUTLASS component uses different physical padding and
a separate capture. Its 3.247/1.745 ms cannot be used to claim a vLLM model win.
Next gate: commit-pinned startup selection, bounded full-model run, matched
output lengths and original-service restoration. Other ledger items remain open.

## Full-model propagation and phase accounting

Implementation `74529f10` builds cleanly on ARM (558.9 MiB reported process
peak, zero swap). Both exported rank sources contain all 42 routed-bank
preparation kernels. Actual execution logs select `expert_operand_transport=tiles`
with 91 replay graphs and 90 F32 collective sites per step. The immutable
physical tile plan reaches startup permutation, activation packing, producer
copies, consumer reads and launch geometry. Shared experts stay unchanged.

Unprofiled roots `/tmp/lunaflux-model-bench-glm-bulk-tiles-20261017-v{1,2}`
reuse the previous six literal prompts and exclude one warmup. Every generated
ID sequence is unchanged versus the retained lookahead run.

| Prompt / input tokens | Retained completion ms | Tiles run 1 ms | Tiles run 2 ms |
| --- | ---: | ---: | ---: |
| Fact / 22 | 563 | 542 | 565 |
| Arithmetic / 23 | 496 | 468 | 483 |
| Chinese / 22 | 906 | 866 | 885 |
| JSON / 23 | 933 | 884 | 902 |
| Code / 24 | 936 | 885 | 902 |
| Retrieval / 36 | 585 | 677 | 545 |
| Sum | 4419 | 4322 | 4282 |

These are **2.20% and 3.10% less completion time**, not a 23% model win.
The first retrieval slowdown does not reproduce. Neither discard it nor
invent a thermal/network explanation without evidence. Two cohorts do not
establish a confidence interval. First-token sums change 1959→1853/1826 ms;
the remainder changes 2460→2469/2456 ms (1 ms rounding differences between
reported phase totals are possible). The measured benefit is mostly prefill.

Selected trace root
`/tmp/lunaflux-model-bench-glm-bulk-tiles-trace-20261017-v1`, compared with
`lunaflux-model-bench-glm-repair-lookahead-trace-20261017-v1`, uses the same
short diagnostic prompt, two output tokens and one excluded warmup. Egress:

| Region | Retained prefill ms | Tiles prefill ms | Retained decode ms | Tiles decode ms |
| --- | ---: | ---: | ---: | ---: |
| Routed gate/up | 74.188384 | 57.095264 | 9.536704 | 7.230496 |
| Routed down | 34.279296 | 27.178464 | 4.333472 | 3.530144 |
| Shared experts | 29.110880 | 30.185088 | 6.427552 | 6.482240 |
| Hyper-connections | 17.314336 | 17.455168 | 12.883360 | 12.982912 |

Ingress confirms warmed prefill gate/up 57.105120 ms and down 26.992320 ms.
The main repair is selected and its matrix saving survives production routing,
not just the synthetic benchmark. Prefill matrix savings are about 24.19 ms;
decode matrix savings are only 3.11 ms per step. The latter does not imply a
23% decrease in an entire decode step. Recurrent projections alone still cost
16.187+5.615 ms in the selected decode; recurrent update itself is 0.683 ms.
Calling all of that time the recurrent algorithm or FlashAttention would
misidentify the work. Sparse leaf is 0.220 ms, while its surrounding projection
and indexing chain is much larger.

The trace still has 91 graphs/90 collective launches and no stream query before
the last collective enqueue. Collective sums and kernel-free intervals vary
under profiling: egress warmed prefill is 32.999 ms collective/17.702 ms empty,
decode 36.543/43.572 ms. These are **not** pure link cost or unprofiled CPU cost.
Do not subtract them from a separate vLLM run to manufacture a gap allocation.
The dominant remaining work is spread across unchanged projection/control/
shared-expert regions and distributed submission, not a missing tile IR level.

Offline analysis now materializes the resolved kernel and step-boundary CTEs.
This avoids repeated SQLite range/name expansion over startup permutation
launches; per-rank family and leaf tables match the original query exactly.
Slow v1 analyses were stopped after those two tables; v2 completes all tables.
This is analysis-only, outside compiler/runtime, and is not a model speedup.

Validation: warning-denied native check with unchanged migration exclusions;
full native suite 5178/5178; affected packages 77/77, subsequent tile-domain
tests 2/2, benchmark runner 3/3, source and trace analyzer regressions 1/1 each.
Targeted info has 1059 existing warnings and zero errors. Native task memory
plans are 99,738,719,500/101,008,132,364 bytes, below 100 GiB per rank; the
observed run-1 host floor is 21.813 GiB. The declared additional storage is not
free: tiled activation capacity and conservatively counted startup scratch add
about 1.09 GB to each prior plan. Startup permutation duration is not isolated.
All completed native ranks close their contexts, with zero task swap.

The fresh vLLM comparison follows below. No ordinary default is promoted from
these measurements, and the other ledger gaps are not declared complete.

## What is major, and what is only a plausible next explanation?

The objective is completed requests, separated into first-token and subsequent
token time. The repaired expert matrices were the largest demonstrated short
prefill target. They are not the largest remaining decode region. The repeat's
first-token sum is 1826 ms, while its remaining time is 2456 ms. Optimizing only
prefill cannot remove most of this corpus's completion time. The selected trace
is useful for ranking and verifying propagation, not for turning instrumented
milliseconds into an additive explanation of the unprofiled vLLM gap.

The remaining source differences are concrete, but their recoverable gains are
not yet measured:

* **Recurrent projection scheduling, not recurrence itself.** Decode input and
  output projections sum to 21.802 ms; recurrent update is 0.683 ms. Our
  `SegmentedMatrixPlan` joins six launch domains but retains separate weight
  pointers and independent per-region CTAs. The single-row lowering performs
  scalar loads and one dependent FP32 accumulator per lane. Buffered prefill
  staging is bypassed by this single-row branch. The installed GLM reference
  physically concatenates six projection banks in `MergedColumnParallelLinear`.
  Both use BF16 here. Physical concatenation alone does not prove fewer DRAM
  bytes or better input reuse: that requires a consumer schedule and matched
  counters. Compare vectorized/prefetched GEMV and physical region packing
  separately, preserving the declared fold unless an explicit numerical
  alternative is independently validated.
* **Hyper-connections still have an ordered serial fold.** Cooperative loads
  and products do not make the reduction parallel: in
  `parallel_source.mbt`, worker zero consumes every 256-product tile in order,
  with two workgroup barriers per tile. This is required by the currently
  selected ordered law, not by functional programming. Our separate pre/post,
  normalization and control publications also differ from the installed
  reference's inter-layer and attention-to-FFN `MHCFusedPostPreOp`. The whole
  hyper region is 12.983 ms in the decode trace. A cross-boundary lifetime plan
  can remove materialization without silently changing arithmetic; a tree
  reduction needs its own numerical law and output-quality validation. Neither
  is accomplished by renaming the existing schedule parallel.
* **Shared experts are unchanged.** Their 30.185 ms prefill / 6.482 ms decode
  does not benefit from the new routed FP4 tile transport. They have a different
  operand and activation contract; silently activating the routed quantizer
  would be a numerical change, not a layout optimization. First compare the
  same representation with shared-input matrix scheduling, then evaluate
  fork/join overlap under explicit scratch lifetimes.
* **Distributed execution is not yet one replayable dependency graph.** There
  are still 91 graph intervals separated by 90 F32 collective submissions.
  The whole-stream wait was removed, but collective-inclusive capture was not
  implemented by that removal. Profiled collective duration includes peer
  waiting; the 36.543 ms decode sum is not pure network transfer. The observed
  kernel-free span is not automatically recoverable CPU overhead. Rank-aligned
  ready/enqueue/complete timestamps and a matched capture ablation are needed
  before assigning this region a share of the vLLM completion gap.

The sparse-attention leaf is only 0.220 ms in this decode. Even eliminating it
cannot explain a large end-to-end gain on this workload. Larger histories may
reverse the ranking, so these priorities must not be generalized to long
context or higher concurrency without measurements. Keep comparison and
causal interventions in the external harness, and lower any verified change
through the existing immutable numerical, schedule, layout and lifetime plans.

## Fresh matched vLLM result

`/tmp/lunaflux-reference-bench-vllm-glm-bulk-tiles-20261017-v1` uses the
retained installed vLLM 0.1.dev20051+g487ecf187 image and the same NVFP4
checkpoint, TP2/PP1, concurrency one, no prefix cache/speculation, maximum eight
output tokens and one excluded request warmup. Its context capacity is 256
and KV allocation 512 MiB. It is not the EXL3 production deployment. All six
literal input ID arrays agree with both native cohorts. Native reports internal
generation duration; reference reports local HTTP request wall time. Thus this
is not an identical API-boundary timer or a matched per-kernel gap allocation.

| Case | Native run 1 ms | Native run 2 ms | Fresh vLLM ms | Native / vLLM output counts |
| --- | ---: | ---: | ---: | ---: |
| Fact | 542 | 565 | 387.121 | 4 / 4 |
| Arithmetic | 468 | 483 | 392.841 | 3 / 4 |
| Chinese | 866 | 885 | 657.336 | 8 / 8 |
| JSON | 884 | 902 | 670.561 | 8 / 8 |
| Code | 885 | 902 | 680.855 | 8 / 8 |
| Retrieval | 677 | 545 | 379.882 | 3 / 3 |

The five equal-output-count requests sum to **3854 / 3799 ms native versus
2775.755 ms reference**: native takes **38.85% / 36.86% longer**. Their decoded
text agrees after removing the native stop marker. Arithmetic is excluded from
this equal-count ratio: native emits `</think>42<|user|>`, while the reference
emits `42</think>42`. Both answer 42, but their token work differs. Across all
six requests native sums 4322/4282 ms versus 3168.596 ms; do not use that ratio
as fixed-output-work evidence. The fresh reference sum is about 0.56% above
the retained reference; this single new cohort is not a confidence interval.
No new full-model correctness or long-context claim follows from six short
prompts. The complete reports are each native root's
`comparison-fresh-reference.json`; previous reports are retained unchanged.

Native run 2's observed host floor is 21.828 GiB; the fresh reference floor is
16.903 GiB. Reference containers retain 96 GiB/no-swap limits. All requests
complete before paired shutdown. Rank zero exits zero; rank one exits 137
after the harness's explicit five-second stop grace. Recorded Docker events
show SIGTERM at epoch 1792234152 followed by SIGKILL at 1792234157, with no OOM
event in that stop interval. The watcher consequently reports failure on the
terminal rank, **not a measured reserve violation**. Preserve this non-graceful
shutdown result; do not call the entire reference lifecycle clean. Both
containers disappear and both GPUs have no compute owners before restoration.
`worker-stop-events.stdout`, both terminal journals and `memory-watch.txt`
retain the distinction. No payload hashing or integrity scan was performed.

Original worker `785b9b6a03b9` and head `2bd8297b9569` restart, in that order,
at approximately 18:49 CST. Their original configurations are unchanged;
readiness confirmation is recorded separately below when complete. The
original production footprint is not a permitted concurrent benchmark state.
