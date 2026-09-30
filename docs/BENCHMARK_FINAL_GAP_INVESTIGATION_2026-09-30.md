# The remaining concurrent-serving gap

Follow-up: the [priority repair retest](BENCHMARK_POSITIVE_PRIORITY_REPAIRS_2026-09-30.md)
implements the selected ownership/product changes, verifies their actual
serving launches and records positive end-to-end gains. The investigation
below remains the prior-runtime diagnostic, not the latest throughput result.

Fresh GB10 traces place most of the remaining difference in decode attention
and small-batch output/down projection. They do not support another blanket
prefill rewrite, removing split decode, or chasing shared-memory conflicts.
There are concrete remaining schedule and arithmetic-lowering experiments;
there is not yet evidence that any one of them guarantees parity.

The unprofiled [matched serving benchmark](BENCHMARK_FRAMEWORK_COMPARISON_2026-09-30.md)
remains the throughput result: Qwen3-0.6B BF16, 4096 input / 256 output tokens,
concurrency 16, LunaFlux **298.8 tok/s**, vLLM **345.5**, SGLang **340.9**.
LunaFlux completion time is 15.6% / 14.1% longer. The investigation below
explains where time is spent; profiled replay time is not a replacement
throughput measurement.

## What was measured

Three new serialized Nsight Systems traces used that exact workload and the
same current serving bundle and pinned reference images. Request epoch
timestamps bound each trace window. Kernel, memcpy and memset interval unions
were computed separately from summed kernel time so overlap is not counted
as extra wall time. No new production kernel or service behavior was changed.

Nsight Compute collected bounded reference-server invocations and bounded
LunaFlux replays of the **actual serving cubins**, with synthetic operands and
the actual launch geometry. LunaFlux's full serving launcher aborted under
NCU injection, both in a container and a privileged host diagnostic; it
succeeded under Nsight Systems. The injection failure is preserved and is
not diagnosed here as a production-runtime defect. LunaFlux NCU data is
therefore a selected-cubin diagnostic, **not a counter capture of the live
serving invocation**. The reference counter workload was 4096/64 C16 with
KV allocation fraction 0.15 rather than the timed benchmark's 0.5. Geometry
and code comparisons are useful; replay durations and raw instruction ratios
are not controlled cross-framework performance or work-equivalence claims.

## Fresh service-time attribution

Times below are summed GPU kernel milliseconds inside the 4096/256 C16
request epoch. Output/down are combined because the references reuse one
CUTLASS symbol for both operations in decode. Ingress combines QKV and its
normalization/RoPE/KV-write suffix in the reference mappings. Gate/up includes
activation. Family classifications remain in the raw summaries.

| Kernel family | LunaFlux ms | vLLM ms | SGLang ms | LunaFlux minus vLLM ms |
| --- | ---: | ---: | ---: | ---: |
| Decode attention, including merge | 9171.149 | 7876.494 | 8637.069 | +1294.655 |
| Prefill attention | 660.618 | 998.551 | 365.349 | -337.933 |
| All attention | 9831.767 | 8875.045 | 9002.418 | +956.722 |
| Output and down, all shapes | 947.755 | 663.814 | 655.949 | +283.941 |
| Gate/up and activation | 1010.799 | 832.447 | 851.672 | +178.352 |
| QKV ingress chain | 870.310 | 723.699 | 888.775 | +146.611 |
| Norm and embedding | 175.438 | 110.342 | 150.335 | +65.096 |
| Head and sampling | 393.453 | 395.107 | 360.504 | -1.654 |
| All kernels, including reference other work | 13229.522 | 11680.252 | 12052.346 | +1549.270 |

All attention plus output/down explain approximately **80% of the summed GPU
time difference to vLLM** in this trace. That is a prioritization budget, not
an additive prediction of end-to-end gains. Prefill attention is better than
vLLM here but worse than SGLang. This does not mean the entire prefill chain
or client-observed TTFT is faster.

| Activity accounting | LunaFlux ms | vLLM ms | SGLang ms |
| --- | ---: | ---: | ---: |
| Kernel + memcpy + memset span | 13663.283 | 11874.895 | 11995.394 |
| Activity interval union | 13227.529 | 11679.865 | 11945.264 |
| Internal time without those GPU activities | 435.755 | 195.031 | 50.130 |

LunaFlux has about 241 ms more internal no-activity time than vLLM and 386 ms
more than SGLang. It is secondary to kernel execution but not zero. Every
one of LunaFlux's **65,456** observed kernel launches belongs to a graph;
graph fallback does not explain this trace. A host/API correlation is still
needed before assigning those gaps to scheduler work, graph submission,
stream synchronization or transport. They are not automatically fixable
CPU bubbles, and activity sums cannot substitute for interval unions.

## Decode: the current bottleneck, not the old diagnosis

The selected split partial entry is
`lunaflux_paged_attention_bf16_decode_split_partial_v3_ep_3903` from c452.
Its grid is `(32,8,8)`, block `(64,1,1)`, registers 118, dynamic shared memory
33,040 bytes. At C16 the 32-row grid includes inactive row slots. It accounts
for **8619.393 ms over 7140 launches**. The merge adds only **44.693 ms**;
merge duration itself is not the main target.

The current implementation already has blockwise state updates, asynchronous
copy, GQA reuse and consumed page-lookup hoisting. Old claims about per-key
state recurrence, selected synchronous c318, missing hoisting consumption,
or all Qwen ingress being restricted to 16 token rows do not describe this
bundle. Ordinary c452 also executes on tail steps; c454 is not the selected
decoder.

### What the selected counter captures show

| Decode resource | LunaFlux replay | vLLM server | SGLang server |
| --- | --- | --- | --- |
| Grid | 32 × 8 × 8 | 1 × 16 × 8 | 54 × 8 × 1 |
| Threads per block | 64 | 128 | 128: 16 × 2 × 4 |
| Registers/thread | 118 | 236 | 56 |
| Dynamic shared bytes/block | 33040 | 81920 | 9216 |
| Shared-memory residency limit | 2 blocks/SM | 1 block/SM | 10 blocks/SM |
| Register residency limit | 8 blocks/SM | 2 blocks/SM | 9 blocks/SM |
| Active occupancy | 8.60% | 8.33% | 67.71% |
| Tensor activity, elapsed | 0% | 12.20% | 0% |
| Total executed warp instructions | 74251392 | 17788544 | 42206504 |

None of these captures reports local spilling. LunaFlux's source-correlated
excessive shared wavefront count is zero. The reference counts are nonzero,
yet both references are faster in the ordinary serving benchmark. This
again rules out conflict count or occupancy alone as the optimization goal.

LunaFlux has 0.14 eligible warps per active scheduler cycle. Its short
scoreboard, long scoreboard and fixed-latency wait ratios are 2.73, 2.16
and 1.79 stalled warps per issued instruction respectively. These are ratios,
**not percentages of elapsed time**. The corresponding reference ratios
have different issue denominators and are not directly comparable latency
percentages. The hot sampled LunaFlux PC is a `BAR.SYNC` with 13,370
long-scoreboard samples: the instruction at the sampled PC is waiting on
earlier dependencies, not proof that the barrier instruction caused all of
the latency.

### Concrete instruction and source gaps

The selected LunaFlux SASS executes approximately:

- **8.39M `LDS.U16`** shared loads;
- **5.57M `SHFL.DOWN`** reductions;
- **14.19M `FADD` + 9.87M `FMUL`**;
- 8.40M `IMAD.U32` and 1.06M 64-bit asynchronous shared transfers.

`kernels/luna_cuda_attention_tile_source/source_blockwise_fold.mbt` walks
keys serially within each score-owning subgroup, converts scalar BF16
components, publishes scores in shared storage, and reads them again for
value accumulation. The decoder's 32-key double-buffered K/V tile consumes
most of its shared allocation. Its 64-thread block has only two query-head
warps, so this footprint restricts active work without making per-key
arithmetic especially wide. `source_decode_pipeline.mbt` retains addresses
and issues **8-byte** asynchronous copies; it does not lack async support.

The pinned SGLang FlashInfer `decode.cuh` instead selects two shared stages,
vector width eight, 16-lane dot groups, two GQA heads and four independent
Z groups. It loads vectors, distributes key work among those groups, and
merges local states at the end. The selected SASS uses `LDS.128` and 8.72M
`FFMA.FTZ`, with 56 registers and a much smaller shared footprint. This is
**also SIMT**, so a Tensor Core-only rewrite is not the only credible route.

The pinned vLLM FlashAttention `flash_fwd_kernel.h` uses matrix QK and PV,
KV block width 128, and converts probabilities to BF16 for PV. Its selected
SASS contains 4.33M tensor MMA instructions. Importantly, the captured
specialization has **`Split=false`**, despite the name `splitkv`:
`compute_attn_splitkv` maps grid Y to batch and Z to head in that case.
The grid's final eight is not eight KV partitions.

Our artifact's numeric law is `blockwise-f32-probability-v1`, compiled with
`--fmad=false`. Separate multiply/add instructions are partly a consequence
of that explicit arithmetic contract. FMA contraction, fast base-two
softmax, and BF16 probability fragments must be offered under explicit
numeric contracts and tested against an independent oracle; silently
changing compiler flags is not a semantics-preserving optimization.

### Causal check: removing split is not the answer

Five alternating unprofiled exact-cubin replay trials at 16 real rows,
4096 history and the actual 32-row launch bucket measured:

| Same c452 module | Median chain time us |
| --- | ---: |
| Ordinary unsplit | 1648.059 |
| Current eight partitions plus merge | 1240.526 |

The current split chain is **24.7% faster** in this synthetic workload. Its
output is bitwise equal to the ordinary route for these operands; oracle
maximum absolute error is 0.000244121. This is not a broad accuracy proof
or a serving-route replacement experiment. Copying vLLM's unsplit choice
without improving our own within-block schedule would regress this case.
The exporter currently emits eight partitions explicitly in
`cmd/lunaflux_qwen3_bf16_candidate_export/decode_module.mbt`; a partition
frontier should only be selected after the improved ownership alternatives
exist and are measured with the actual runtime bucket.

## Output/down and gate/up

The common rows16 output and down launches have only **16 CTAs on 48 SMs**:
at most 16 SMs can work on either invocation, regardless of register
occupancy potential. The reference decode projection launches 64 CTAs,
grid `(8,8,1)`, with 32 threads each. Its two-dimensional grid is a launch
swizzle, not proof that it computes 128 padded token rows.

LunaFlux rows16 output has 256 threads, 64 registers and 28,672 combined
static/dynamic shared bytes. Rows16 down has 128 threads, 72 registers and
20,480 static shared bytes. The exact resource replays show zero
source-correlated excessive shared wavefronts. Output's hot sampled point
is a warp synchronization waiting on loads; down's is pipeline/control
arithmetic waiting on earlier loads. The primary first experiment is a
finer column product with smaller cooperative groups, not more occupancy
inside the same 16 blocks.

Rows16 gate/up launches 192 blocks of 512 threads, 88 registers and 34,816
shared bytes; both register and shared resource limits permit only one
resident block/SM. Its hot PCs are a barrier-stalled loop branch and shared
stores waiting on global loads. The instruction mix is dominated by branch,
move, address and predicate operations. Merely widening the CTA is not a
solution; narrower producer/consumer ownership and fewer transfer/control
instructions need whole-MLP measurement.

`kernels/luna_projection_tile_compiler/work_domain.mbt` already expresses
workgroup cardinality as the pure row/column product. The required extension
is executable small-row column/cooperation alternatives, evaluated using
actual exported resources. Dynamic storage must be included: the first
diagnostic replays omitted output's 8192 and gate/up's 16384 dynamic bytes.
Those replays are retained but **excluded**; the `output-exact` and
`mlp-exact` reports use the correct allocations.

## Next experiments, within the functional multi-layer compiler

| Priority | Pure compiler alternative | Required comparison |
| --- | --- | --- |
| 1 | Vector load ownership, 16/32-lane score groups, independently distributed KV subgroups, compact live shared stages | Exact-bucket c452 chain, short/long histories and C1/8/16; instructions, dependencies, resources and duration |
| 2 | Explicit contraction/softmax/probability numeric laws | Independent numerical and model-output tests; matched strict vs approved fast arithmetic, not a hidden global flag |
| 3 | Small-row projection product tiling and smaller cooperative groups | Output/down and whole-MLP chains; CTA coverage and actual static + dynamic storage |
| 4 | Partition/bucket frontier after new ownership schedules exist | 1/2/4/8 partitions including merge, real row utilization and mixed steps; do not assume less splitting wins |
| 5 | Host-correlated graph submission and streaming intervals | Correlate each GPU gap with host/API events before adding a runtime optimization |

These alternatives belong in semantic numeric laws, immutable ownership /
schedule plans, liveness/resource estimation, and device lowering. The model
builder and scheduler need no Qwen-specific CUDA tuning rule. More IR layers
by themselves will not remove serial shared loads or generate more useful
CTAs. Resource reduction is valuable only when it preserves useful work and
improves completion time.

There is room to close the measured gap, but no measured gain is assigned to
these unimplemented alternatives. A broad workload vector, including
diverse prompts and longer histories, is still required before generalizing
this small-model result.

## Retained artifacts and diagnostic limits

Remote root:
`/home/wlc004s/lunaflux-final-gap-investigation-20260930.tiAzXMDJ`.
It contains all three Nsight Systems reports/SQLite traces and exact-epoch
activity summaries, NCU reports, raw source counters, sampled PC summaries,
synthetic correctness results, memory samples, pinned reference source
headers, and the MoonBit automation used to create them.

The selected decode cubin extracted from serving module eight hashes to
`e35c853b1cd8abdb95de35c0bafd704f85e707fc398103b225b5d0f844c23994`.
The ordinary and partitioned replay entries use those same bytes. Source
and image identities are the ones in the matched benchmark report.

An initial reference `per-launch-config` filter captured many graph nodes;
that exact diagnostic container was stopped, and its partial readable
report was preserved. Subsequent captures use a global one-launch bound.
Two early projection-name matches captured the vocabulary head, not
output/down; neither is used as output/down evidence. The final steady
projection capture verifies grid `(8,8,1)`. SASS exports can contain aliased
function entries: they are not summed as additional launches.

No numerical DRAM-byte equality or bandwidth-ceiling conclusion is made
from these GB10 counter captures. Missing byte metrics are unavailable, not
zero traffic. Source-correlated excessive wavefronts are not an assertion
that every hardware aggregate conflict metric is zero.

GPU campaigns were serialized. Serving/trace runs used 64/80 GiB ceilings,
no additional cgroup swap and a monitored 32 GiB available-memory reserve;
bounded cubin replays had an 8 GiB ceiling. Profiling was shut down and the
GPU released. This report changes documentation only and does not admit or
deploy a new runtime.

Trace minimum available memory was 98.9 GiB for LunaFlux, 53.3 GiB for vLLM,
and 50.9 GiB for SGLang. The retained archive contains 1430 files; model
weights, source/build caches and unreadable failed-capture files are listed
as exclusions, never modified. It was downloaded without overwriting to
`/private/tmp/lunaflux-final-gap-investigation-20260930.gLcCHdhf/final-gap-investigation.tar.gz`.
The remote and local SHA-256 match:
`4291b7d91b7bc705a3a53c166525973566b446f763f0e360cccfcb9836b61213`.
