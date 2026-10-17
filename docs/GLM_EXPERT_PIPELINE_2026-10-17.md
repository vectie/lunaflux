# GLM expert pipeline experiment

The user authorized temporarily stopping `glm53-exl3-head` on Spark .178 and
`glm53-exl3-worker` on .179, then restoring those same containers. No production
engine cutover is authorized. Preserve both containers and the separate NFS
container. Benchmark limits remain 100 GiB per Spark, zero task swap and an
independent 16 GiB host available-memory reserve. No payload hashing.

The retained clean native baseline is coalesced-20261014-v1. Its selected
prefill trace attributes 74.721536 ms to routed gate/up and 36.826272 ms to down.
Prior same-source synthetic counters show substantial load-dependency and
barrier waiting, but are not counters from the full checkpoint. The previous
shared-input-staging optimization did not demonstrate a full-model benefit.

Current lowering issues `cp.async`, immediately waits for all groups, computes,
then retires shared readers. Thus it has asynchronous instructions but no
inter-chunk copy/compute overlap. The local vLLM Marlin implementation separates
shared-memory fetch, completion, register fetch and matrix work, maintaining
multiple live stages. This motivates a measured experiment, not a claim that
Marlin is the exact selected vLLM kernel for this checkpoint.

At most two candidate schedules: two-slot independent staging, and two-slot
shared-left staging. An immutable grouped-matrix plan selects stage count;
physical storage/lifetime planning defines disjoint slots; private CUDA lowering
publishes the next chunk before consuming the current chunk. Ascending K order,
MMA arithmetic, scales, activation quantization and output publication remain
unchanged. Test boundary chunks, stage retirement, exact outputs, sanitizers,
hardware counters and clean full-model timing. Keep comparison outside the
compiler and preserve losing results. Do not select an unmeasured default.

## Loading schedules rejected

Full-width synthetic probes now allocate 288 distinct weight banks, rather
than mapping 64 logical experts onto eight repeated banks. They still are not
captured checkpoint operands. The two-stage independent candidate preserved
all output/workspace bits, but R22 chain median only fell from 7.791 to 7.611 ms.
Its NCU gate/up capture shows 96.53M to 125.83M warp instructions, shared memory
14,848 to 28,672 bytes, active warps 45.66% to 23.94%, load-wait percentage
75.46% to 64.67%, and barrier percentage 19.52% to 25.00%. Counter percentages
are not additive wall-time contributions. Requested DRAM-byte metrics were
not emitted by this device/report; no bandwidth claim is inferred.

Reusing the sequential gate/up scratch restored the intended footprint, but
R22 chain median was 7.749 versus 7.783 ms (less than 0.5%). Two-stage paired
staging likewise had no useful benefit. All retained probe roots start with
`/tmp/lunaflux-glm-fp4-performance-two-stage-20261017-` or
`/tmp/lunaflux-glm-fp4-performance-two-stage-paired-20261017-v1`.
These experimental source changes were removed from the repository; their
generated sources, negative measurements and counters remain in those roots.

## Stable grouping fix

The same expert chain's grouping kernel used one thread to perform repeated
route searches (15.896 ms summed over the retained profiled prefill). The new
pure `StableAssignmentPlan` factors first-occurrence order into independent
leader/rank calculations, one ordered integer prefix, and uniquely owned
publication. CUDA lowering alone chooses bounded shared storage and thread
mapping. Arithmetic, expert order, within-expert order and padded metadata are
unchanged; large capacities retain the previous implementation.

`stable-ranks-20261017-v1` R22 chain median is 6.771 versus 7.793 ms (13.1%
less component time), with bitwise outputs/workspaces. The v2 harness also
compares the entire padded metadata area. Its separate integer oracle covers
35 combinations of empty, invalid, singleton, boundary and capacity rows,
with identical, distinct and mixed expert patterns. All 35 pass plain,
memcheck/full leak, racecheck and synccheck runs under 2 GiB/no swap.
The component result is not a 13.1% engine claim; full-model measurements
below show the smaller end-to-end benefit.

The fresh retained-engine cohort `/tmp/lunaflux-model-bench-glm-baseline-20261017-v1`
completes the six measured prompts in 4.685 s, all prior native output IDs
unchanged. Its selected source and memory observations are retained. The
temporary original GLM containers remain stopped during the bounded retest.

## Hardware attribution and propagation

The selected-source grouping capture is retained at
`/tmp/lunaflux-glm-fp4-performance-stable-ranks-20261017-v2/paired.ncu-rep`.
Same 22-row operands, 155 live groups, grid1/block128:

| Counter | Serial search | Parallel stable ranks |
|---|---:|---:|
| Replayed kernel time, ms | 1.044448 | 0.013888 |
| Warp instructions | 301,818 | 11,499 |
| L2 read sectors | 5,069 | 1,306 |
| Registers/thread | 32 | 30 |
| Allocated shared bytes/block | 1,024 | 7,168 |
| Active-warps percentage | 2.113 | 8.312 |
| Long-scoreboard percentage | 36.567 | 28.511 |

The extra shared storage enables concurrent rank calculation. A larger
barrier-stall percentage (0.007% to 11.308%) accompanies a much shorter kernel:
percentages alone are not optimization goals. These are component counters,
not full-model timings or counters from vLLM.

The local vLLM `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu` similarly
distributes route counting and performs an integer prefix before publication.
We preserve our existing first-occurrence/job order rather than copying its
expert-sorted or atomic placement contract. The compiler contains no reference
engine dependencies or measured-output comparisons.

Implementation `149aa071` builds cleanly on ARM in
`/tmp/lunaflux-checkpoint-build-glm-stable-ranks-20261017-v1`. The new runtime
`/tmp/lunaflux-glm-expert-tp-stable-ranks-20261017-v1` has the parallel assignment
in all 42 routed layers in each rank's generated AOT source. Existing plan
memory remains 98,648,722,228 / 99,918,135,092 bytes. Warning-denied native check
passes with existing migration exclusions unchanged; affected native tests
pass 170/170. Unrelated dirty migration/TLS changes remain unstaged.

The full working-tree native suite ran 5,159 tests: 5,158 passed, and the
existing `service/online_tcp/control_server_wbtest.mbt:53` zero-wait socket test
timed out. This was not a clean full-suite pass. No timeout, tolerance or
warning policy was changed in response; an isolated rerun is recorded below.
The unchanged `service/online_tcp` package then passed 57/57 in isolation.
This supports timing sensitivity under concurrent local load, not a claim that
the original full-suite failure did not occur.
The subsequent complete native rerun passed 5,159/5,159 with the unchanged
warning exclusions and no source/test modifications between runs.

## Unprofiled full-model retest

All three runs use the same six literal prompt-ID arrays (22, 23, 22, 23, 24,
36 input tokens), one warmup request, C1, TP2, maximum eight generated tokens,
64-row native capacity, BF16 KV, queued collectives and graph replay. Every
native output token ID is unchanged. These short prompts do not establish
long-context, concurrent-serving or independent model correctness.

| Measured cohort | Six completion times summed | TTFT summed | Completion reduction |
|---|---:|---:|---:|
| Retained engine | 4.685 s | 2.181 s | — |
| Stable ranks | 4.503 s | 1.986 s | 3.88% |
| Stable ranks repeat | 4.488 s | 1.978 s | 4.20% |

The first-token improvement is about 9%; post-prefill time is approximately
unchanged. The fix is useful but does not close the remaining expert-matrix
gap. Roots are `/tmp/lunaflux-model-bench-glm-baseline-20261017-v1`,
`/tmp/lunaflux-model-bench-glm-stable-ranks-20261017-v1` and
`/tmp/lunaflux-model-bench-glm-stable-ranks-repeat-20261017-v1`.
Both ranks exit successfully and close CUDA. The first two observed minimum
host reserves are 22.32 and 22.43 GiB; task swap remains zero.

The fresh reference root is
`/tmp/lunaflux-reference-bench-vllm-glm-20261017-v1`. Its startup log explicitly
selects `FLASHINFER_CUTLASS` NVFP4 MoE, not Marlin. The local vLLM adapter
`model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py` passes
quantized weights/scales, routing IDs/weights and TP geometry into FlashInfer's
fused MoE entry point. Marlin was a pipeline-design comparison, not evidence
for the selected reference kernel. New matched reference results and the
post-fix selected trace must be reported separately from older ratios stored
in the initial comparison reports.

The current native expert path already uses packed E2M1 operands and the
`mxf4nvf4.block_scale` matrix instruction; the earlier decoded-BF16 limitation
in the October 13 report is historical, not the current explanation. Its
remaining source-level scheduling restriction is one published K chunk at a
time (`commit_group; wait_group 0`, compute, reader barrier), with 16-row
groups and separate gate/up reductions in the retained schedule. The tested
two-stage replacements did not demonstrate a useful gain. FlashInfer's
adapter alone does not prove which lower-level schedule accounts for the
remaining end-to-end difference; that requires selected reference-kernel
tracing, not another unsupported inference from library names.

## Fresh vLLM comparison

The fresh reference completes all six literal requests with HTTP 200 in
3.150044 s (one warmup excluded). Per-request times are
0.384370 / 0.388344 / 0.654044 / 0.665396 / 0.675554 / 0.382336 s.
Native outputs contain 4 / 3 / 8 / 8 / 8 / 3 tokens; vLLM contains
4 / 4 / 8 / 8 / 8 / 3, so arithmetic is excluded from the equal-output-count
subset. Equal counts are not proof of equivalent outputs or numerical behavior.

| Five equal-output-count requests | Completion sum | Time above fresh vLLM |
|---|---:|---:|
| vLLM | 2.761700 s | — |
| Retained LunaFlux | 4.164 s | 50.78% |
| Stable ranks | 3.996 s | 44.69% |
| Stable ranks repeat | 3.989 s | 44.44% |

This is a 4.2% native improvement on that subset, not performance parity.
The older 43% gap used an older vLLM cohort (2.912043 s for that subset);
do not compare that percentage directly with the fresh 44.44% and call it
a native regression. Both current native versions use the same fresh
reference here. Reports are `comparison-fresh-reference.json` in each native
root. Native CLI completion includes diagnostic output/retirement; reference
timing is buffered HTTP completion. The reference selects FP8 MLA KV while
native uses BF16 KV. This is matched prompts, not an identical numerical policy.

The reference's observed host-memory minimum is 17.488 GiB; its inspected
GPU allocations are 92,736 MiB on each Spark, below 100 GiB. Docker task caps
are 96 GiB with no task swap. Native repeat's observed minimum is 22.585 GiB.
The reference emits its existing gate/up global-scale mismatch warning.
After successful responses the requested stop produces head exit 0 and worker
exit 137, which makes the generic watcher report failure. The retained log
shows shutdown after responses and ample released memory, not a reserve breach
or an inference failure; both GPU owners are gone before the native trace starts.

## Selected full-model trace

`/tmp/lunaflux-model-bench-glm-stable-ranks-trace-20261017-v1` uses the same
literal `2+2?` prompt, two output tokens and one warmup as the retained
coalesced trace. Source selection and runtime select the new AOT on both ranks.
Egress step 3 is the warmed measured prefill, not the cold first step:

| Sum across 42 routed layers, ms | Retained trace | Stable ranks trace |
|---|---:|---:|
| Group assignment | 15.895968 | 0.173216 |
| Gate/up | 74.721536 | 74.271552 |
| Down | 36.826272 | 36.279680 |
| Activation global quantization | 8.178176 | 8.150592 |
| Entire routed family | 138.060672 | 121.275200 |

Ingress grouping is independently 0.168192 ms. This establishes actual
full-model propagation of the grouping fix; gate/up and down remain essentially
unchanged. Egress prefill span falls 305.013 to 274.785 ms, while kernel-free
intervals change 25.238 to 9.112 ms. These profiled intervals cannot be
substituted for unprofiled request latency or summed across TP ranks. Egress
decode span is 215.549 versus 207.969 ms in the historical trace; its collective
and host-gap distribution changes under profiling, while the unprofiled cohort
does not demonstrate a material decode improvement.

The next largest selected prefill families after routed experts are recurrent
34.731, shared experts 29.179 and sparse attention 25.283 ms. Removing grouping
does not remove those costs or establish their precise excess versus reference.
Both ranks retain 91 graph launches and 90 collective launches per step,
with zero `cuStreamQuery` calls before the last collective enqueue. This is
not a fallback to a missing graph path or a recurrence of whole-stream polling.
Both ranks exit zero and close CUDA; trace memory floor is 19.494 GiB, task swap
zero. Profilers/converters retire before the original containers are restarted.

## Restoration and replacement boundary

At 12:52:55–56 CST on October 17, the original `glm53-exl3-worker` and
`glm53-exl3-head` containers were restarted, not recreated. Their Docker IDs
remain `785b9b6a03b9` and `2bd8297b9569`; `glm53-nfs` remained running throughout.
At 13:01:04, the head's `/health` returns HTTP 200 and `/v1/models` lists
`GLM-5.3-Flash-EXL3`. Both original containers are running and report no OOM.
No task benchmark, profiler or GPU workload remains active.

The restored original deployment is EXL3 with its existing draft model,
multimodal configuration, 850,000-token maximum and 0.85 utilization setting;
it is not the bounded NVFP4 reference used above. Its settings were preserved.
Immediately after restoration, GPU allocation is approximately 100.4 GiB per
Spark and the head has about 8.34 GiB available host RAM. **Those original
production settings do not meet the benchmark's 100 GiB/16 GiB envelope.**
Do not describe them as memory-limit compliant merely because the task's
benchmark runs were. No further benchmark is launched alongside that service.

The new engine is not substituted into production. This phase fixes and
proves the stable-grouping bottleneck, but the much larger expert-matrix
cost remains, the matched completion gap is 44.4%, and the short NVFP4 corpus
does not validate replacement of the restored EXL3/draft/media configuration.
Performance comparison and numerical/output validation remain external to the
compiler. No payload hashes, tolerance changes or reference-dependent runtime
logic were introduced.

## Selected CUTLASS follow-up: output ownership experiment

The original service is temporarily stopped again with explicit permission.
The first bounded vLLM trace (`cutlass-trace-20261017-v1`) was stopped by the
independent memory guard during graph capture: 16,268,312 KiB available is
below the 16 GiB floor. No OOM or performance result is inferred. The separate
v2 trace disables graph capture only for selected-kernel identification;
ordinary graph-enabled timing remains unchanged.

External installed-FlashInfer component root
`/tmp/lunaflux-glm-cutlass-component-r22-20261017-v3` uses synthetic NVFP4
operands, 288 full banks, R22/H4096/I1024/top8/TP2. Its CUTLASS matrix symbols
select 128x128x256, two-stage TMA, block384 and 48 persistent CTAs. Counter
capture uses 12 GiB/no swap after confirming at least 28 GiB available.
Gate/up and down respectively: 3.247/1.745 ms replay time, 79.03M/64.37M warp
instructions, 3.99%/10.34% long-scoreboard and 2.49%/9.86% barrier stalls.
Both have about 22.9% active warps, 168 registers/thread and 89,088 dynamic
shared bytes. These are synthetic component measurements, not serving times
or captured checkpoint arithmetic. Pending server-symbol comparison must not
be silently assumed. The CLI prints "outputs matched" even with no supplied
output oracle; this is explicitly NOT a numerical correctness result.

At most two new schedules: 64-column and 128-column output ownership with the
same 16 logical rows, K ordering, quantization and publication. An immutable
grouped-matrix column plan drives physical operand regions and terminal CUDA
fragment ownership. Fewer independent output tiles should amortize producer
addressing and barriers; resource growth may negate the benefit. Compare
against frozen stable-grouping sources with alternating complete-chain trials,
bitwise workspace/output checks, sanitizer boundaries, and new counters.
Neither wider tiles nor the earlier rejected two-stage pipeline become a
default merely because CUTLASS uses them. No payload hashing.

The eager v2 serving trace also stopped at the host reserve (16,693,680 KiB
available); it produced no valid serving trace. Both reference pairs were
retired without relaxing the floor. Consequently the installed CUTLASS
component symbols are measured, but their identity with the live serving
invocation remains unverified.

Both 64/128-column candidates preserve the complete synthetic chain and
workspace bytes at R1/R3/R22/R64 with 288 distinct banks. A separate H192/I320
tail case also agrees at every row count. The 64-column R3 full-bank run has
zero memcheck/leak/race/synchronization errors. This is R3 sanitizer coverage,
not an all-shape sanitizer claim. Roots are
`/tmp/lunaflux-glm-fp4-performance-columns64-20261017-v2`,
`/tmp/lunaflux-glm-fp4-performance-columns128-20261017-v1`, and
`/tmp/lunaflux-glm-fp4-performance-tails128-20261017-v1`.

At R22, median capacity-grid chain time moves from 6780.067 to 6673.101 us
with 64 columns (1.58% less). Trimmed diagnostic grids do not improve; 128
columns regress R3. The 64-column actual-source counter replay reduces
gate/up warp instructions from 96.53M to 73.42M, but long-scoreboard remains
81.07% to 81.62%. Down instructions fall 49.42M to 33.75M and long-scoreboard
70.34% to 57.71%. These counters do not establish a full-model gain.
CUTLASS's 128 logical rows also perform more padded arithmetic than our 16;
its higher tensor-activity percentage is not directly useful-work efficiency.

The next source-pinned full-model trial uses 64 columns as the CUDA backend's
grouped FP4 choice. Its default status is provisional until that test; no
original GLM serving container or configuration is changed. Pure grouped
matrix ownership drives physical storage and terminal fragment lowering;
comparison data and reference dependencies remain outside the compiler.

Validation after this change: full native suite 5161/5161, warning-denied
native check with the existing `-20-25-29-35-79-92` migration exclusions,
and the external runner regressions pass. Tail H192/I320 at R3 also has zero
memcheck/leak/race/synchronization errors. Pinned `15adf788` ARM build passes;
both AOT rank sources contain all 42 widened routed gate/up and down paths.
Planned bytes remain 98,648,722,228 / 99,918,135,092.

Trace v3 was a distinct setup failure, not a reserve breach or kernel result:
the compact 64-position configuration supplied 128 MiB KV/state, while the
installed hybrid model requires 0.14 GiB. No trace was captured and both
owners retired. The runner now reserves 256 MiB for compact diagnostic traces,
with a regression preserving the ordinary 256-position/512-MiB baseline.
No live-serving kernel identity is claimed from the three unsuccessful traces.

## Wider ownership: full-model decision

The `15adf788` 64-column candidate was actually exported, compiled and selected
on both Sparks, with the retained six literal prompt ID vectors, row/history
capacity 64, TP2, queued graph replay and one excluded warmup. Both runs exit
zero, release CUDA and preserve all six native output-token sequences. The
retained 32-column binary was rechecked between the two candidate runs.

| Cohort | Six completion times, seconds | Sum | TTFT sum |
|---|---|---:|---:|
| 64-column first | .569 / .501 / .913 / .955 / 1.256 / .586 | 4.780 | 2.154 |
| Retained 32-column recheck | .577 / .507 / .935 / .949 / .954 / .585 | 4.507 | 1.992 |
| 64-column repeat | .572 / .539 / .911 / .929 / .937 / .586 | 4.474 | 1.993 |

The repeat is only 0.73% faster overall than the fresh retained run and its
arithmetic prompt is slower; the first run regresses, dominated by one code
prompt. Its cause is not established by these unprofiled measurements. Do not
discard that result or declare a reliable improvement from the best run.
The five equal-output-count sums are 4.279 / 4.000 / 3.935 s, respectively,
against the same-day vLLM 2.761700 s. Retained completion remains 44.84% above
vLLM; this campaign has not closed the gap. Both engines' different timing
boundaries and KV policies, and the unequal arithmetic completion length,
remain as documented above.

Decision: **restore 32-column default**. The typed 64/128-column alternatives
remain explicit offline experiments, not a production winner. Instruction
reduction did not establish a dependable full-model latency reduction, and
gate/up's long-scoreboard frontier remains. This is not a completed repair of
the major matrix pipeline bottleneck. Comparison stays outside the compiler.

Roots: `/tmp/lunaflux-model-bench-glm-columns64-20261017-v1`,
`/tmp/lunaflux-model-bench-glm-retained32-recheck-20261017-v1`, and
`/tmp/lunaflux-model-bench-glm-columns64-repeat-20261017-v1`; each contains
literal IDs, source/runtime selection, per-request outputs, terminal/release
logs, independent host-memory observations and `comparison.json`.

The installed component's exact schedule is
`MainloopSm120ArrayTmaWarpSpecializedBlockScaled<2,3,...>` with
`KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<3>`.
In the image's `flashinfer/data/cutlass/include/cutlass/gemm/dispatch_policy.hpp`
1494–1505, those parameters mean two mainloop stages and three scheduler
pipeline stages, **not** three MMA stages. The cooperative kernel header
`kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp` 456–460 defines
one producer warp group and two cooperating consumer groups; 626–902 and
904–1074 separate producer and consumer execution. Its collective
`collective/sm120_blockscaled_mma_array_tma.hpp` uses `PipelineTmaAsync` at 168,
producer acquire at 753, TMA copies of A/B/SFA/SFB at 763–767, and consumer
wait/release at 934–977.

Our selected lowering still emits each chunk's `cp.async` requests followed
immediately by `wait_group 0`, a CTA publication barrier, MMA, and a reader
barrier before the next chunk. Column ownership does not change that
sequence. This is a concrete transport/role-scheduling difference, not proof
that copying the reference tile size or adding IR layers alone will accelerate
the full model. A future independent producer/consumer transport plan must
preserve operand/scaling arithmetic and reader lifetimes; TMA descriptor/warp
details belong only in CUDA lowering. Do not infer a quantitative full-model
gain from this source difference before an executable paired measurement.

## Final capture limitation and default-path preservation

Compact eager trace v4 supplies the required 256 MiB state, but still reaches
the reserve guard during startup/warmup: rank 0 has 16,655,500 KiB available,
below 16,777,216 KiB. The watcher requests paired shutdown; rank exits 1/137
after that request are not evidence of an OOM. Both GPUs and profilers are
subsequently released. The retained logs are in
`/tmp/lunaflux-reference-bench-vllm-glm-cutlass-trace-20261017-v4`.
No successful live-serving trace or exact component/serving symbol match was
obtained. Do not relabel the installed CUTLASS component counters as counters
from the actual matched serving request. The completed unprofiled reference
benchmark remains valid. Further profiled loading attempts are stopped rather
than relaxing the reserve or replacing this missing observation with inference.

Returning to width 32 initially left a subtle code-generation regression:
gate/up instructions matched the retained binary, but down instruction order
and register assignment differed even though resource totals were unchanged.
The single-fragment plan now partially evaluates fragment offsets/publication
coordinates and preserves its established declaration and loop policy. Wider
ownership still shares its left fragment through the same lowering; no copied
legacy implementation or model-specific branch is introduced.

The paired default export at
`/tmp/lunaflux-glm-fp4-performance-default32-20261017-v3` has **identical gate/up
and down SASS and resource reports**, after normalizing only the paired symbol
prefix, against the frozen retained stable-grouping source. The earlier v1/v2
nonidentical down reports are retained. This comparison is an external
compiled-code preservation check, not an accuracy oracle or performance gain.

Final default32 verification also preserves all synthetic chain/workspace bytes
at R1/R3/R22/R64 on both capacity and trimmed grids, with 288 distinct banks,
no nonfinite values and no device errors. The R3 capacity full-bank memcheck,
full leak check, racecheck and synccheck all report zero issues. This does not
expand sanitizer coverage to every shape. Full native tests after the final
partial-evaluation correction pass 5161/5161; warning-denied native check uses
the unchanged migration exclusions above. Targeted `moon info` completes with
561 existing migration warnings and zero errors. The external instruction
comparison runner passes its own regression.

The two-schedule experiment budget is exhausted without a reliable full-model
winner. AKO's measured decision is to preserve the retained production schedule
and the unsuccessful evidence, not to promote the best noisy sample. The next
performance change would need an independent producer/consumer transport
schedule and a memory-feasible actual-serving capture; it is not implemented
or benchmark-proven in this phase. No integrity scans or payload hashes were
performed.
