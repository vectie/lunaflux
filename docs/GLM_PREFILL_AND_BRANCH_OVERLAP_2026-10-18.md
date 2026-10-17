# Prefill ownership and independent MoE branches

The subsequent HC control-tail fusion and vocabulary TP ablations are in
[the HC/vocabulary report](GLM_HC_AND_VOCABULARY_TP_2026-10-18.md).
The results below remain the retained pre-HC/vocabulary baseline.

## Experiment contract

This follows the actual serving traces in
[the execution-work report](GLM_EXECUTION_WORK_AND_WAITS_2026-10-17.md), not a
new assumption that all routed expert GEMMs are slower than vLLM.

Two independent changes are evaluated, followed by their combination:

1. Expose narrower independent output-column ownership for grouped BF16
   prefill. A 16-row shared group currently produces only 16 gate/up CTAs on
   48 SMs. Test 32 and 16 columns against the retained 64-column schedule.
   Keep the ascending K16 MMA sequence, BF16 boundaries, decode fold and
   collective placement unchanged. Column partitioning belongs to immutable
   grouped-matrix IR; CUDA owns consumer subgroup mapping and staging.
2. Fork the independent shared-expert branch alongside routing/routed experts,
   then join before their existing coalesced partial publication. Use explicit
   fork/join intervals in parallel IR, startup-owned queues/events and distinct
   branch scratch. Preserve numerical combine order, cancellation and resource
   release. Graph replay must retain these dependencies rather than reverting
   to an ordered kernel list.

Budget: at most three schedules per target before revisiting the hypothesis.
First use boundary/deterministic/sanitizer component checks, then frozen
baseline/candidate full-model runs and the matched vLLM prompt corpus. Report
per-request input/output counts, TTFT and completion time; separately report
selected kernel geometry, overlap and stage time. Different output lengths and
different timer boundaries are not silently normalized into a speedup claim.
Comparison and profiling remain offline, outside compiler legality.

Both Sparks are exclusively reserved during GPU tests, at most 100 GiB per
Spark with a separate 16 GiB host MemAvailable reserve and no swap. Stop only
the previously identified GLM head/worker containers, leave NFS running, and
restore those original containers after testing. No payload hashing or
integrity scans. Unrelated toolchain/TLS changes remain outside task commits.

## Results

The first component experiment is complete. H4096/I1024, cap64, one shared
expert; paired AOT kernels on Spark .179, five alternating-order trials with
20 launches per timing. All BF16 activation and F32 partial outputs are bitwise
identical at live rows 0/1/3/16/17/22/32/63/64/65, including inactive tails.
Both 32- and 16-column alternatives pass memcheck/full leak, racecheck and
synccheck. No payload hashes were computed.

| Live rows | 32-column gate/up vs 64 | 32-column down vs 64 |
|---|---:|---:|
| 1 | 0.025552 vs 0.031798 ms | 0.015008 vs 0.013157 ms |
| 16 | 0.134261 vs 0.203419 ms | 0.027701 vs 0.037944 ms |
| 22 | 0.144536 vs 0.199733 ms | 0.031602 vs 0.041733 ms |
| 32 | 0.148229 vs 0.202557 ms | 0.031701 vs 0.042037 ms |
| 64 | 0.165211 vs 0.242888 ms | 0.043371 vs 0.053922 ms |

These are per-kernel medians, not full-model savings. Decode arithmetic is
unchanged, but shared resource declarations and padded launch counts change,
so decode timing is NOT unchanged: down regresses while gate/up improves.
The 16-column route also regresses down at64 rows (0.058322 vs0.053502ms).
Select32 for the next complete-model experiment; retain16 as a nondefault
candidate, not a universal winner.

NCU R22 confirms32 gate/up changes static shared memory17,440→9,248bytes,
registers40→46, active-warps8.87→12.04%; total instructions8.136M→7.890M
and L2 read sectors1.147M→1.241M. More effective parallelism wins despite
more traffic and a higher barrier-stall percentage. Down instructions increase
4.563M→5.022M while time decreases. Counter runs are replay-perturbed and not
substituted for the unprofiled timings above.

Raw roots:
`/tmp/lunaflux-glm-shared-vector-columns32-20261018-v1` and
`/tmp/lunaflux-glm-shared-vector-columns16-20261018-v1`.

The fork/join executor uses existing device events and ordered-launch owners;
no new CUDA ABI is needed. Shared scratch and partial outputs are separately
owned; input/counts are shared reads. Shared producers issue first on the
auxiliary lane, while the primary lane retains routing/routed order. Both join
before the existing F32 TP publication. Scalar index loops preserve the
allocation-free submission contract; iterator allocation caught by the
regression test was removed.

The actual MoonBit GPU queue owner now passes 32 eager and 32 captured replays
with changing input, exact outputs and explicit release, plus partial-submission
cancellation/drain. The same test passes CUDA memcheck/full leak (zero leaked
bytes), racecheck (zero errors/warnings), and synccheck. Evidence:
`/tmp/lunaflux-glm-fork-owner-20261018-v2`; clean source pin `19d4883c`.
The retained v1 failure was a test symbol containing a trailing NUL, rejected
by the real function-loader ABI; the corrected fixture uses plain symbols.
This is a real queue-owner gate, not a replacement CUDA-only harness.

Complete-model AOT variants at columns64/32 are built from `0d614c5f` on ARM;
the owner implementation is unchanged by the fixture correction. Full-model
ablations, paired control/candidate repetitions and a separate fresh vLLM run
are complete. The historical reference remains separately identified.

Local phase validation: `moon info --target native` reports 3,669 existing
migration warnings and zero errors; `moon fmt` and native check pass.
The full native suite passes **5,193/5,193** with the existing warning
exclusions `-20-25-29-35-79-92`; no new exclusions or tolerance changes.
This is the current working-tree suite, which includes unrelated ongoing
migration files. Clean pinned ARM builds independently validate the task's
executable and actual fork-owner test; unrelated files are not staged.

## Full-model ablations and repetitions

The frozen six literal input lengths are `[22,23,22,23,24,36]`, C1,
one excluded warmup and at most eight output tokens. Initial native outputs
remain `[8,4,8,8,8,3]` with identical IDs. Historical vLLM emits
`[4,4,8,8,8,3]`; exclude prompt0 from equal-output-count speed ratios, not
from the raw evidence. Fresh vLLM retains these output counts. Native internal resident completion and buffered
reference HTTP time have different timer boundaries.

| Schedule | Six completion sum, ms | TTFT sum, ms | After-first sum, ms | Minimum available host, GiB |
|---|---:|---:|---:|---:|
| 64 columns, serial (fresh control) | 4132 | 1630 | 2502 | 21.78 |
| 32 columns, serial | 4130 | 1605 | 2525 | 21.73 |
| 64 columns, overlap | 4283 | 1562 | 2721 | 21.79 |
| 32 columns, overlap | 3932 | 1568 | 2364 | 21.84 |
| 32 columns, overlap, repeat | 4036 | 1584 | 2452 | 21.84 |
| 64 columns, serial, repeat control | 4166 | 1639 | 2527 | 21.82 |

The tile-only full-model change is **not a demonstrated win**: 25ms saved
before first token is offset by 23ms later. These are single unprofiled runs,
not confidence intervals or an attribution of all post-first movement to one
kernel. Component timings cannot be summed into an end-to-end promise.
Both ranks exit0 with explicit context release and zero swap. Source/export
and execution logs confirm the actual 64/32-column selection.

The first combined run takes 4.84% less time than the first control; output
IDs remain unchanged in every ablation and repetition. The wide-tile overlap run regresses 3.65% overall,
concentrated in its first three requests' post-first-token durations. This
negative run is retained; neither a warmup nor a resource-contention cause is
proven by that total. The two control runs average **4149ms**, versus
**3984ms** for the two combined runs: **3.98% less completion time**. Their
TTFT sums average 1634.5→1576ms (3.58% less); after-first sums average
2514.5→2408ms (4.24% less). These are two repetitions, not a confidence
interval or a long-context/concurrency result.

On all five prompts with equal output counts against the retained vLLM
reference (indices1–5, 31 outputs), mean native completion is
3370→3218.5ms (4.50% less). Equal counts are not identical generated content
or a correctness proof. Do not compare all-six aggregate cross-framework
times as equal output work: prompt0 generates eight native versus four
reference tokens. A fresh vLLM comparison is collected separately below.

Why the component gain is larger than the end-to-end gain: the shared kernels
are only part of each request; the component loop repeatedly reuses one weight
bank, unlike full-model execution. Narrow tiles alone do not demonstrate a
full-model gain. Concurrent branches lengthen individual kernel lifetimes
while shortening their combined execution region. Thus neither a kernel
median nor raw overlap is an additive prediction of request savings.

The repeat's coarse 5-second telemetry observes SM clocks 2398–2431MHz and
temperatures 44–48°C during work, without an observed thermal collapse. It
cannot attribute per-kernel variation or prove that clocks/cache caused the
wide-tile negative result. Memory-clock telemetry is unavailable on this
device and is not fabricated.

Roots: `/tmp/lunaflux-model-bench-glm-columns64-serial-20261018-v1` and
`/tmp/lunaflux-model-bench-glm-columns32-serial-20261018-v1`; each contains
`comparison.json`, per-request tokens, selection, terminal and memory logs.
The overlap roots have the same pattern with `columns64-overlap` and
`columns32-overlap` respectively. Repetitions use the same serial64/overlap32
root names ending in `-v2`. Both ranks in all six runs exit0, release their
CUDA contexts and report zero swap; all observed host reserves exceed21.7GiB.
Per-rank native planned allocations are about 99.74/101.01GB, below 100GiB;
cgroup peaks alone are not treated as total unified-memory device usage.

## Fresh vLLM comparison

Root `/tmp/lunaflux-reference-bench-vllm-glm-columns-overlap-20261018-v1`.
Installed vLLM is `0.1.dev20051+g487ecf187`, not the newer local `../vlm`
checkout. It mounts the same GLM-5.3-Flash-NVFP4 checkpoint as the native
benchmark, not the EXL3 checkpoint used by the original service to be restored.
TP2/PP1, C1, context256, batched-token cap256, 512MiB KV budget, one excluded
warmup, no prefix caching/speculation, and literal input IDs are explicit.
Both containers have 96GiB memory and equal memory+swap limits (no benchmark
swap allowance); the independent watcher enforces 16GiB host availability.

Actual selected backends are `FLASHINFER_MLA_SPARSE_SM120` with `fp8_ds_mla`
KV and `FLASHINFER_CUTLASS` NVFP4 experts. Native uses BF16 KV and its own
numerical laws. vLLM's retained warning that `w1_weight_scale_2` must match
`w3_weight_scale_2` remains visible. Neither timing nor agreeing token counts
establishes reference or native numerical correctness. No checkpoint, input,
tolerance or reference arithmetic is edited for this benchmark.

All six HTTP requests complete200 after the excluded warmup. Raw completion
times, including the unequal-output fact request, are:

| Prompt | Native control mean, ms | Combined native repeat1 / repeat2, ms | Fresh vLLM, ms | Native / reference outputs |
|---|---:|---:|---:|---:|
| Fact | 779 | 748 / 783 | 386.577 | 8 / 4 |
| Arithmetic | 502.5 | 486 / 485 | 393.741 | 4 / 4 |
| Chinese | 783 | 726 / 747 | 655.168 | 8 / 8 |
| JSON | 794 | 751 / 766 | 675.038 | 8 / 8 |
| Code | 799.5 | 756 / 773 | 673.430 | 8 / 8 |
| Retrieval | 491 | 465 / 482 | 377.356 | 3 / 3 |

Using **all five** equal-output-count prompts (31 output tokens):

| Variant | Completion sum, ms | Longer than fresh vLLM |
|---|---:|---:|
| Native64 serial, two-run mean | 3370 | 21.45% |
| Native32 serial, single run | 3351 | 20.77% |
| Native64 overlap, single run | 3441 | 24.01% |
| Native32 overlap, two-run mean | 3218.5 | 15.99% |
| Fresh vLLM, single run | 2774.733 | — |

The combined equal-count total improves4.50% over the repeated native control;
the remaining mean completion gap is **15.99%**, not eliminated. Individual
combined repeats are14.75%/17.24% longer than this reference. Output tokens
per request-second, including prefill, are9.63 native versus11.17 reference;
these are not decode-only or saturated serving throughput. The historical
three-prompt cohort is also retained: combined1947/2002ms against1705.954ms
fresh reference (14.13%/17.35% longer). No favorable prompt is substituted.

Four decoded texts agree after removing the native stop marker; fact and JSON
do not. This is visible comparison evidence, not a quality acceptance test.
All native IDs remain unchanged across the six native runs. Every native root
now has a separate `comparison-fresh.json`; historical `comparison.json` files
are not overwritten. Reproduce the cohort accounting with
`scripts/summarize-glm-replay-ablation.mbtx` over those fresh reports.

Fresh reference host availability never falls below **17.116GiB** in the
5-second observations, above the16GiB floor. Both reference containers have
96GiB memory/no-swap limits; observed model processes are about91,800MiB
during loading. The benchmark client completes all six requests, then requests
paired stop. Rank0 exits0; rank1 exits137 after that explicit5-second Docker
stop, with peer-shutdown messages. The watcher reports nonzero because it
classifies that terminal worker as failed. Its full logs are retained: this
is not relabeled a clean process exit, an OOM, or a reserve violation. Both
GPU owner lists are empty before starting the original service restoration.

## Actual concurrent execution

Combined selected trace:
`/tmp/lunaflux-model-bench-glm-columns32-overlap-node-20261018-v1`, analysis
`/tmp/lunaflux-glm-columns32-overlap-trace-20261018-v3`.
It contains one 16-input/2-output request after an excluded warmup, captured
with node **and all-API** tracing. Steps3/4 are the measured request.
Both ranks explicitly select independent branches; 91 interval graphs are
prepared. It is not the older collective-inclusive decode-graph trace.

Egress shared/routed intersection **unions** are 67.278784ms prefill and
11.672064ms decode, across 43 observed streams. This proves real overlap,
not just the presence of a stream or a new IR field. Shared kernel duration
sums are 67.579520/12.807776ms; routed/router/glue sums 103.925088/17.151872ms.
These sums double-count concurrency and must not be called completion times.
Whole compute unions are 175.257472/64.992160ms. All-API tracing inflates host submission/NCCL timing severely
(decode span182.609312ms); do not use that span as unprofiled latency or a
removable production budget. No observed compute/NCCL overlap is claimed.

NCU resource fields additionally record 102400 bytes shared memory per SM,
including allocated/driver overhead: shared gate/up64 uses 18560 bytes/CTA,
32 uses 10368; down 10240→6144. This increases the possible co-residency
budget, but theoretical capacity alone is not proof of achieved co-residency
or the cause of the full-model wide-tile regression.

Fresh serial trace uses identical profiling flags, workload and interval
capture: `/tmp/lunaflux-model-bench-glm-columns64-serial-node-20261018-v1`,
analysis `/tmp/lunaflux-glm-columns64-serial-trace-20261018-v2`. It has zero
shared/routed overlap. The selected measured-step GPU accounting is:

| Quantity | Serial prefill | Combined prefill | Serial decode | Combined decode |
|---|---:|---:|---:|---:|
| Expert compute union, ms | 120.497055 | 104.200864 | 21.182751 | 18.196448 |
| All compute union, ms | 192.180063 | 175.257472 | 68.109375 | 64.992160 |
| Shared/routed overlap union, ms | 0 | 67.278784 | 0 | 11.672064 |
| Shared duration sum, ms | 20.587776 | 67.579520 | 6.353856 | 12.807776 |
| Router/routed/glue duration sum, ms | 99.964127 | 103.925088 | 14.834687 | 17.151872 |

The expert compute union shrinks by 16.296191ms (13.52%) / 2.986303ms
(14.10%), not by the raw overlap duration. Both branches' kernel lifetimes
grow under concurrent execution, absorbing much of the theoretical overlap
benefit. This is the important execution arrangement difference: parallelism
must be evaluated jointly with resource footprint. It is not a claim that
instruction work disappears. Device residency, shared memory and duration
observations do not by themselves isolate the precise cache/issue cause of
the wide-tile run's decode regression.

Profiled complete step spans are 301.58→278.57ms prefill but
175.77→182.61ms decode. All-API host gaps and rank-arrival/NCCL residency vary
strongly; they are not unprofiled speedup evidence. The unprofiled cohort and
repeat remain the performance gate. Analysis unions also avoid tiny overlapping
graph timestamps within a nominally serial stream being counted twice.

## Compiler boundary and remaining scope

The two independent mechanisms are implemented and exercised, not inferred
from source presence:

| Layer | Responsibility in this change |
|---|---|
| Model/numerical semantics | Existing activation, accumulation, quantization and combine order remain unchanged. |
| Grouped-matrix physical IR | Immutable 16/32/64-column ownership; unsupported schedule combinations reject. |
| Parallel/effect IR | Immutable independent producer spans and a common join; no collective may bisect the producers. The binding owner supplies disjoint writable branch storage. |
| Execution owner | Startup-owned queues/events/leases, allocation-free submission, graph-preserved dependencies, cancellation and deterministic release. |
| Private CUDA lowering | Concrete subgroup work, staging and matrix fragments; no NVIDIA concepts added to the model or scheduler. |
| Offline evaluation | Counter interpretation, vLLM comparison and winner selection remain outside compiler legality/numerical correctness. |

The 32-column+fork profile is explicit, not a universal default. This phase
does not claim to finish the separate hierarchical quantization-max reduction,
HC boundary fusion, vocabulary TP sharding, collective payload/placement, or
expert publication-wait investigations in the preceding report. It also does
not establish long-context, larger-batch, or quality parity. The measured
short-prompt benefit must not be extrapolated to those workloads.

## Original service restored

Restoration completed at about00:59 CST October18. The exact original worker
`785b9b6a03b9e985e82133a5f3ede582f90f4816bd0382afdf0a0e53e39d1572`
and head `2bd8297b956955c35146cbf657c6e0f44c532990712c56d7e969edc1e5ac352c`
were restarted without changing their configuration. `/health` returns200;
`/v1/models` reports `GLM-5.3-Flash-EXL3`, context850000. NFS remained running
throughout. Evidence:
`/tmp/lunaflux-glm-columns-overlap-restore-20261018-v1`.

The100GiB/Spark and16GiB host-reserve constraints above describe the exclusive
benchmark runs, not the restored production service's original configuration.
After restoration the original head/worker report102660/102666MiB GPU usage
(about100.3GiB each) and8969552/11161616KiB MemAvailable (8.55/10.64GiB).
Thus the restored service does **not** meet that benchmark envelope. It was
restored unchanged as explicitly requested, not silently resized or used for
further tests. No benchmark GPU owner remains. Another physical campaign must
first release that service or obtain a separately agreed smaller configuration.

Compiler contract documentation is updated in `docs/PARALLEL_IR.md`,
`docs/ARCHITECTURE.md` and the parallel-IR package README; its25 native tests
pass again after the documentation update. The complete fresh comparison is
committed in `508be88a`; implementation and actual-source validation pins are
listed above. All commits are scoped to task paths on `main`.
