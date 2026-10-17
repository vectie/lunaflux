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
