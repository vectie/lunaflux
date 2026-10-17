# Parallel HC and collective-inclusive replay

Continue from `GLM_BULK_EXPERT_2026-10-17.md`. The measured decode HC region
is 12.983 ms; it contains 16,384-add serial control folds. The executor still
submits 91 compute graphs separated by 90 collectives. These profiled durations
are hypotheses for removable dependencies, not an additive attribution of the
unprofiled 36.86% completion gap.

## Implementation contract

* Reuse precision IR's explicit tree square/dot laws. Reassociation defines a
  different finite-precision program; it is not an exact rewrite of ordered
  addition and agreement with vLLM is not a correctness proof.
* Physical HC planning owns reduction partition geometry and legal fusion.
  Fuse collapse/normalization only while preserving the BF16 intermediate
  publication, normalization law and each scalar rounding boundary. Hardware
  instructions belong exclusively in private CUDA lowering.
* Parallel IR owns a fixed-shape collective-inclusive execution graph. Capture
  is startup-only, with stable addresses, ordered rank agreement, explicit
  memory reserve and retained communicator/buffer lifetimes. No live-request
  capture, variable NCCL count inside a fixed graph, or removal of cancellation
  and terminal retirement is permitted. Unsupported shapes retain an explicit
  prepared path, not silently mis-sized collectives.
* Comparison, error measurements, profiler interpretation and quality decisions
  remain outside compiler/runtime. No payload hashing or integrity scans.

## Bounded experiment

At most three HC candidates and two capture variants, followed by separate
HC-only, graph-only and combined full-model ablations against the retained
whole-tile baseline. Use identical literal inputs, output-token vectors,
unprofiled alternating trials, selected-symbol/graph evidence, affected
sanitizer/leak/cancellation tests and a separate vLLM comparison. Retain failures
and regressions. Do not equate microkernel benefit with request benefit.

Keep the original GLM service running during source work. Stop only its exact
head/worker containers for physical tests, leave NFS untouched, and restore the
same containers afterwards through successful health/model responses. Each
task owner stays within 100 GiB and zero swap; independently maintain at least
16 GiB host MemAvailable on each Spark. No concurrent benchmark GPU owners.

## Validation before full-model ablations

Implementation `1251a565` propagates both alternatives through the plans,
lowering, frame binding and explicit checkpoint CLI. Defaults are unchanged.
The clean ARM build peaks at 797.4 MiB with zero swap. The HC experiment
`/tmp/lunaflux-glm-hyper-tree-20261017-v1` passes all four geometry cases and
live-row/tail/identity controls bitwise against an independent staged renderer
of the **same tree arithmetic**. Full GPU memcheck/leak, racecheck and synccheck
report zero errors. This is not bitwise equivalence to ordered reduction or a
vLLM quality claim. Its serial tree oracle is not a production speed baseline.

The first full native suite was 5,182/5,183: the new cancellation fixture lacked
capture API redirects and incorrectly passed fake handles to the real ABI.
The repaired fixture's collective and cancellation tests pass 6/6. Native
ordered-executor and NCCL graph lease/sequence/cancellation ASan/UBSan gates
also pass. Graph teardown now discards executable/raw graph references before
NCCL abort, retaining operand leases until stream drain and executor close;
otherwise NCCL abort can wait for references owned by the graph itself.

After the fixture repair, the complete native suite passes **5,184/5,184**.
Full-model ablations are in progress. The first capture attempt
`/tmp/lunaflux-model-bench-glm-collective-graph-20261017-v1` fails before requests
with `Busy` while NCCL's lazy connection jobs are outstanding. Both ranks close
their CUDA contexts, zero swap. Preserve this failure, not a timing sample.
The startup poll-count cutoff has been replaced by an explicit caller-supplied
monotonic timeout (default 60 seconds); replay performs no clock reads.
vLLM additionally warms up NCCL before capture; whether warmup is necessary
here must be distinguished from an artificially short polling budget.

Fusion is
collapse plus normalization, preserving the BF16 intermediate; it is not yet
vLLM's cross-layer post/pre fusion.

## Separate full-model ablations and capture repair

HC-only runs `glm-parallel-hc-20261017-v1/v2` are repeatable, but change the
generated token IDs for three prompts. The three unchanged-output requests
improve from 885/902/545 ms to 779/812/505 ms in repeat two (7–12%). Aggregate
time is misleading: the first answer now emits eight tokens rather than four.
The tree arithmetic remains an explicit alternative, not a quality-approved
default. Both ranks close cleanly, zero swap, with the host reserve respected.

The longer startup timeout permits capture without adding a warmup. The
capacity-only graph run `glm-collective-graph-20261017-v2` completes inference
but regresses: an eight-token case rises from 902 to 1,307 ms, and a three-token
case from 545 to 672 ms. It communicates 64 rows at every decode site instead
of one. These are provisional request timings from a lifecycle-failed run,
not a successful benchmark gate.

The egress rank then stalls in NCCL teardown. A scoped debugger backtrace
finds `commDestroySync -> ncclCommPollCallbacks -> ncclIntruQueueMpscDequeueAll`:
the checkpoint caller's reverse-order defer aborts its communicator before
closing its graph-owning stage. The exact task unit was terminated after TERM
failed; original service containers and NFS were not modified. Failure logs
are retained in that experiment root. The caller now closes rank/stage before
aborting the communicator, including error paths.

The second schedule is a pure parallel-IR row-bucket plan: one-row decode and
capacity-bounded prefill. Both graphs are prepared at startup. NCCL retains
each graph's operand leases separately from transient submissions; each replay
advances only the logical site count, never the total captured lease count.
Graph references retire before communicator abort. No runtime allocation,
clock, capture or change to kernel arithmetic is added to replay. Alternating
bucket replay/cancellation and native reference-balance/ASan/UBSan tests cover
both graph lifetimes. The source-pinned `fc4c10ad` ARM build passes (749.7 MiB,
zero swap), affected tests pass 154/154, and the complete native suite passes
5,185/5,185 with the existing warning exclusions unchanged.

The repaired graph-only run `glm-collective-graph-20261017-v3` completes normally
on both ranks, records two prepared graphs and explicit closed CUDA contexts,
and retains all six baseline output-token sequences. Timings (ms) are
576/498/911/920/921/562 versus 565/483/885/902/902/545: **2.48% slower** in total.
Prefill/first-token time sums to 1,958 ms versus 1,826 ms (+132); after-prefill
time is 2,430 versus 2,456 ms (−26). This localizes the aggregate regression to
prefill without assigning every millisecond to NCCL. Minimum observed host
availability is 21.803 GiB; process peaks are 63.86/62.15 GiB and swap is zero.

The inspected vLLM source explicitly distinguishes these regimes:
`../vlm/vllm/config/compilation.py:630` describes full decode graphs and piecewise
prefill/mixed graphs; `v1/cudagraph_dispatcher.py:167` constructs separate legal
keys. This is source evidence for an alternative scheduling policy, not proof
of which policy the frozen baseline selected. Our capacity prefill graph still
reduces 64 rows for 22–36 live rows. Keep this non-winning graph schedule
explicit; fewer host launches alone are not an end-to-end optimization.

The retained **actual baseline startup log** adds stronger evidence: compilation
mode NONE, graph capture sizes `[1, 2]`, maximum graph size 2, and breakable CUDA
graph enabled. Local dispatcher source returns NONE above its maximum. Thus
the matched long-enough prefills do not use that engine's full decode graph;
do not describe the default policy as proof of piecewise capture in this run.

Both full-prefill-capture schedule variants have now been measured. Rather
than tuning their kernels, a separate, bounded follow-up tests **one** regime
policy: `CaptureDecodeCollectives`. Parallel IR selects collective-inclusive
one-row replay, but retains existing interval graphs and exact live collective
counts for prefill. All graphs and leases remain startup-bound; alternating
ordinary and captured submissions use the same monotonically ordered effect
owner. This changes schedule, not HC or any other numerical operation. Keep
the two losing policies available explicitly for reproducibility; only the
diagnostic `-collective-graph` option chooses the revised policy. Validation
and the new paired measurement are required before a performance claim.

## Selected-kernel and timeline evidence

The separate two-token Nsight run
`/tmp/lunaflux-model-bench-glm-trace-hc-collective-graph-20261017-v3`
selects both experimental paths. Its decode has one graph launch containing
90 collective kernel nodes, versus 91 compute graph launches in the retained
baseline. There are no whole-stream queries before the last collective enqueue.

Including sparse-layer HC names (not only the `glm_hyper_*` family), the warmed
egress traces show:

| Region | Previous | Parallel HC + bucketed graph |
|---|---:|---:|
| Prefill HC, all 90 sites | 19.897 ms / 540 calls | 5.836 ms / 450 calls |
| Decode HC, all 90 sites | 14.765 ms / 540 calls | 4.775 ms / 450 calls |
| Prefill collective kernel time | 32.999 ms | 57.371 ms |
| Decode collective kernel time | 36.543 ms | 3.980 ms |
| Decode kernel-free intervals, instrumented | 43.572 ms | 0.687 ms |

The removed 90 launches are the legally fused normalization publications. The
remaining HC kernels use the selected 256-partition tree. In contrast, routed
gate/up and down decode time is essentially unchanged: 7.230/3.530 ms versus
7.349/3.498 ms. KDA input/output projection remains 16.412/5.804 ms in the new
trace. This work did not optimize those other regions.

Do **not** convert the profiled gap/collective reductions into a promised
unprofiled speedup. The profile records every CUDA API and graph node; changing
91 host graph submissions into one changes its instrumentation burden. The
unprofiled graph-only after-prefill total improves by just 26 ms across all six
requests, not the tens of milliseconds per decode step suggested by naive
subtraction of these traces. The trace proves selected structure and the HC
kernel gain; the unprofiled ablations decide request performance.

Raw analysis is in
`/tmp/lunaflux-glm-hc-graph-trace-egress-analysis-20261017-v4` and
`/tmp/lunaflux-glm-bulk-tiles-trace-egress-analysis-20261017-v3`.
The existing external analyzer now includes all `*hyper*` names to avoid
silently excluding sparse-layer HC from the comparison; its regression passes.

## Final unprofiled comparison

Clean source-pinned `b2a1538b` builds on ARM and selects the revised policy in
`glm-collective-graph-20261017-v5` and `glm-hc-collective-graph-20261017-v5`.
Both ranks report 92 prepared graphs: one collective-inclusive decode graph
plus the existing 91 compute intervals used for prefill. All benchmark ranks
exit successfully and explicitly close their CUDA contexts. Graph-only
preserves all baseline token IDs; combined preserves all HC-only token IDs.

The fixed workload is TP2/PP1, C1, input lengths 22/23/22/23/24/36, output cap
8, one excluded warmup, native capacity/history 64, no prefix reuse or
speculation. Reuse the same-day frozen vLLM run
`/tmp/lunaflux-reference-bench-vllm-glm-bulk-tiles-20261017-v1`; no reference
rebuild or payload integrity work was performed. vLLM is
`0.1.dev20051+g487ecf187`, modelopt NVFP4 on the same checkpoint, TP2/PP1,
context cap 256 and 512 MiB KV allocation. Timing boundaries remain native
resident CLI versus buffered reference HTTP; native KV is BF16 versus the
reference's FP8 MLA path. This is a short-request comparison, not long-context,
concurrency, accuracy, or throughput-parity qualification.

| Prompt | Native output tokens: old → HC (vLLM) | Old ms | HC-only repeat 2 ms | HC + revised graph ms | vLLM ms |
|---|---:|---:|---:|---:|---:|
| Fact, 22 input | 4 → 8 (4) | 565 | 791 | 823 | 387.121 |
| Arithmetic, 23 input | 3 → 4 (4) | 483 | 512 | 517 | 392.841 |
| Chinese, 22 input | 8 → 8 (8) | 885 | 779 | 792 | 657.336 |
| JSON, 23 input | 8 → 8 (8) | 902 | 803 | 818 | 670.561 |
| Code, 24 input | 8 → 8 (8) | 902 | 812 | 823 | 680.855 |
| Retrieval, 36 input | 3 → 3 (3) | 545 | 505 | 510 | 379.882 |

Only Chinese/code/retrieval retain identical **native output IDs** across all
numerical variants. On that common cohort, total completion time is 2,332 ms
old, 2,096 ms HC-only (**10.12% less time**), 2,125 ms combined (**8.88% less**),
and 1,718.073 ms vLLM. HC-only remains **22.00% longer than vLLM**; combined
remains **23.69% longer**. The graph-only revised policy totals 4,247 ms across
all six unchanged-output requests versus 4,282 ms old (−0.82%), so no robust
additional graph speedup is established. The best measured variant is HC-only
with existing interval graphs. Do not add the profiled HC and gap reductions.

On all five latest output-count-matched requests, HC-only is 3.411 s versus
vLLM 2.781475 s (22.63% longer), and combined is 3.460 s (24.39% longer).
This five-request cohort differs from the old baseline's five-request cohort;
do not subtract those headline percentages as a paired improvement. The raw
six-request combined total is almost unchanged, but it now emits five more
tokens; reporting that as zero optimization would also be misleading.

The two prior combined capacity-prefill runs, v3/v4, are retained: totals
4.436/4.396 s with identical HC token IDs. They are slower than the revised
4.283 s schedule. Correctness of the explicit tree program is established by
its independent arithmetic oracle and GPU sanitizer controls; **model quality
is not established** by these short truncated outputs. In particular, fact,
arithmetic and JSON change token sequences. Neither the tree nor the replay
experiments are promoted to ordinary serving defaults.

Final revised runs retain at least 21.797/21.845 GiB host MemAvailable,
respectively, with zero swap and 100 GiB process/device caps. Combined process
peaks are 62,563,627,008 / 61,945,716,736 bytes. Comparisons and trace analysis
are offline `.mbtx`; runtime replay does not hash, capture, allocate MoonBit
objects, read clocks, or render diagnostic evidence. The final mixed-path
native tests also exercise both graph and prefill cancellation; native
ASan/UBSan and explicit reference-balance checks pass.

Final warning-denied check with the unchanged repository warning exclusions
passes; the local working-tree native suite passes **5,186/5,186**. The final
mixed-path packed-execution package passes 124/124, affected packages 155/155,
and the external trace analyzer regression passes 1/1. A name-filter invocation
mistakenly selected zero tests and was not counted; the complete package and
full suite were subsequently executed. The clean commit-pinned ARM checkpoint
build is separate evidence from the dirty local full suite. Unrelated
toolchain/TLS edits are preserved and are not included in these commits.
