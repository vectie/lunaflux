# GLM scales + bulk: execution graph and local-MoE TP correction

The initial sections retain the qualified **combined** artifact's trace, not
the older latent-cache path. That initial investigation changed no production
code. The follow-up below implements and retraces **local MoE combination
before TP**, with complete-answer checks. Neither phase ran an overall
benchmark, payload hash or integrity scan. Trace collection and comparison
remain outside the compiler.

## What was actually captured

- Native artifact: `/tmp/lunaflux-glm-http-policy-combined-20261019-v1`, binary
  pinned to `72ce7bd1`. Expanded BF16 cache, row-scaled E4M3 matrix index,
  32 scale partitions, `bulk-tiles-128x2x48` experts. Ordered-token-F32 KDA
  and synchronous projection staging remain selected.
- Actual dispatch verifies scale grid 32 and bulk gate/up grid 48, block 288,
  static shared memory 78,368 bytes. Its source uses 32 producer and 256
  consumer threads with two slots. These are not the old expert launches.
- Same literal 7,973-input-token request, C1, physical prefill rows 64,
  context capacity 8,192, eight slots. One requested output token bounds the
  diagnostic; this is not an answer-quality or throughput test.
- Primary new capture: rank 1, software CUDA/node tracing, device-event
  tracing disabled, 500 ms requested collection timer. Collection command
  overhead extends the observed capture to about 693 ms.
- **One complete first 64-token chunk** is between consecutive embedding
  anchors. It includes all 45 layers, 42 gate/up and 42 down launches, 90
  reductions and one vocabulary gather. The final incomplete chunk is excluded.
- vLLM is the retained installed `487ecf187` trace of the matching request,
  checkpoint, TP2, chunk size and capacity, with the actual selected
  FlashInfer/CUTLASS kernels. Only its first chunk is used below. No new vLLM
  server or full benchmark was launched in this session.

The native long request was stopped by diagnostic memory headroom protection;
**late-history execution was not captured**. The profiler warns that some
events might be missing and reports eight incomplete events at collection
stop. Independent counts support the selected complete first-chunk region,
not completeness of the interrupted trace. There is no usable native rank-0
GPU trace, so peer-arrival delay cannot be separated from communication.

These are single observed execution windows, with different warmup/capture
states. Their stage allocations are evidence, not a stable speedup estimate.

## Whole-chunk allocation on the execution graph

Each pair below is **LunaFlux / vLLM GPU kernel-duration sum in ms** for the
first chunk, not additive wall time. Attention types are alternatives at each
layer, not parallel branches. HC spans the repeated boundaries of the model.

```mermaid
flowchart TD
    H["HC / norms across boundaries<br/>9.796 / 3.974 ms"] --> A{"Attention type"}
    A --> K["34 KDA layers<br/>Projections: 39.693 / 26.782<br/>Update + other: 9.174 / 4.432"]
    A --> S["11 sparse layers<br/>Projections: 20.892 / 9.410<br/>Index: 0.260 / 0 bypass<br/>Attention: 1.361 / 0.964<br/>Other: 0.257 / 1.156"]
    K --> T["Attention TP → HC"]
    S --> T
    T --> F{"FFN type"}
    F --> D["3 dense FFNs<br/>6.085 / 2.282 ms"]
    F --> M["42 MoE FFNs<br/>Router: 5.966 / 3.718<br/>Gate/up: 118.793 / 112.606<br/>Down: 54.681 / 61.314<br/>Routed glue: 10.643 / 3.905"]
    D --> O["FFN TP → HC → next layer"]
    M --> O
```

The latest native scale scans total **1.641984 ms**: input 0.285504 and
activation 1.356480, 42 calls each. They are already included in routed glue.
The old single-partition scan costs must not be used to describe this path.
Native down is not uniformly slower in this capture: its 42-call sum is below
the reference, although the first-layer example below is slower.

Early reference index scoring is bypassed while history is within top-K;
zero here does not mean no reference index work at long history. This first
chunk cannot establish which sparse-history component dominates near 8K.

### Overlap and collective accounting

| Measured first-chunk property | LunaFlux | vLLM |
|---|---:|---:|
| Collective kernel union, ms | 59.750560 | 17.670304 |
| Recorded reductions / gathers | 90 / 1 | 91 / 1 |
| **MoE-only collective union**, 42 calls, ms | **35.066048** | **8.260864** |
| MoE TP pack + unpack, 84 calls, ms | 4.394656 | no separate counterpart |
| Compute–collective intersection, ms | 0 | 0 |
| Shared-expert compute union, ms | 108.399360 | 7.273535 |
| Shared intersection with other compute/collective kernels, ms | 107.993247 | 3.212895 |
| Shared-only kernel coverage, ms | 0.406113 | 4.060640 |
| Uncovered by recorded kernels/copies, ms | 6.395712 | 5.330233 |

Shared work is almost entirely overlapped in LunaFlux. Adding 108.4 ms to its
serial path would be wrong. Overlapped work can still compete for resources;
this trace does not measure the counterfactual without that competition.
Neither engine overlaps compute with its recorded collectives in this window.

Native anchors are embedding-to-embedding, 148577104–492766928 ns
(344.189824 ms); reference anchors are HC-pre-to-next-chunk-HC-pre,
11675257888–11935337792 ns (260.079904 ms). These outer boundaries differ,
including one additional reference reduction. **Do not subtract these windows
as an exactly matched end-to-end gap.** MoE-only comparison uses 42 matching
FFN regions and excludes subsequent head collectives from the final region.

Native kernel union is 337.759200 ms despite summed kernel durations of
445.853791 ms. Including copies gives 337.794112 ms of recorded coverage.
Uncovered time is not automatically GPU idle, CPU overhead or avoidable delay.

## Actual first-MoE graph, with timestamps and fork/join placement

This is layer 3 in the first chunk, physical rows 64. Native timestamps are ms
from its router/producer start; the reference origin is its preceding FFN
HC-pre kernel, not the same logical router instruction. Clocks are not aligned
between engines. These envelopes are not exactly matched end-to-end timings.
Arrows denote producer/consumer dependencies, not drawn-to-scale widths.

```mermaid
flowchart TB
    subgraph L["LunaFlux — CURRENT scales + bulk"]
        LX["FFN input"] --> LR["Router + input preparation<br/>0.000–0.219"]
        LX --> LS["Shared expert branch<br/>gate/up 0.106–2.103<br/>down 2.102–2.315"]
        LR --> LG["Routed gate/up<br/>0.219–2.638"]
        LG --> LA["Activation scale / quantize / pack<br/>2.638–2.715"]
        LA --> LD["Routed down<br/>2.715–4.107"]
        LD --> LJ["Join partial producers → TP pack<br/>4.107–4.170"]
        LS --> LJ
        LJ --> LT["9 MiB FP32 all-reduce<br/>4.174–4.977<br/>0.803 ms, includes peer wait"]
        LT --> LP["Unpack → publish → MoE combine<br/>4.982–5.070"]
    end
    subgraph V["vLLM — selected CUTLASS path"]
        VX["FFN input"] --> VR["Router + input preparation<br/>0.000–0.243"]
        VX --> VS["Shared branch starts later<br/>gate/up 2.385–2.497<br/>activation/down to 2.560"]
        VR --> VG["Routed gate/up<br/>0.260–2.445"]
        VG --> VA["Activation / requantization<br/>2.485–2.543"]
        VA --> VD["Routed down<br/>2.509–3.681"]
        VD --> VJ["Finalize / combine routed + shared<br/>3.681–3.707"]
        VS --> VJ
        VJ --> VT["0.5 MiB BF16 all-reduce<br/>3.709–3.869<br/>0.161 ms, includes peer wait"]
    end
```

The reference activation/down timestamp intervals overlap; preserve this
observation rather than inventing a fully serial schedule. The graph is a
logical dependency graph, not a claim that complete kernel retirement must
precede every downstream launch. Native graph-node stream IDs are 13/126;
do not infer branch identity from the old trace's main/auxiliary stream labels.

| First MoE measured operation | LunaFlux ms | vLLM ms |
|---|---:|---:|
| Routed gate/up | 2.419360 | 2.185888 |
| Routed down | 1.392096 | 1.172736 |
| TP packing + unpacking | 0.119168 | no separate counterpart |
| TP collective | 0.803072 | 0.160640 |
| Router-start to post-TP result ready | 5.069632 | 3.869472 |

The observed first-MoE envelope difference is 1.200160 ms. Its TP-kernel
difference is 0.642432 ms, plus 0.119168 ms of explicit native packing and
unpacking. These locate a substantial part of this particular interval's
excess, but are not a causal intervention or extrapolation to the full prompt.

## Why TP differs: actual source and payload, not a guessed bubble

Native `integration/packed_expert_mlp/program.mbt` constructs
`CoalescedSumPlan(left_width=hidden*selected, right_width=hidden)`.
`program_prepare.mbt` joins both producers **before packing and reduction**;
then unpacks and publishes routed/shared results before combining them.
Current emitted `rank1.cu:8528` uses widths 32768 and 36864:

```
64 rows × (8 routed partials + 1 shared partial) × 4096 × 4 bytes = 9 MiB
```

The installed vLLM runner's selected late-reduction path combines routed and
shared results before `_maybe_reduce_final_output`. The recorded collective
is `AllReduce_Sum_bf16_RING_LL`, on the combined hidden output:

```
64 rows × 4096 × 2 bytes = 0.5 MiB
```

Thus this native MoE boundary moves **18× the payload**, not 18× the measured
time. It also has separate pack/unpack passes and a different selected NCCL
protocol. The source agreement is checked against installed snapshots, with
`../vlm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py` as the local
implementation counterpart (late reduction around lines 455–487, ordering
around 715–770). This is not an allegation that vLLM's arithmetic is the only
correct arithmetic: native explicitly preserves per-expert FP32 reduction
before rounding. Moving the sum/rounding boundary changes numerical order and
requires an explicit numerical/execution plan plus output-correctness testing.
This trace request did not make that compiler change.

## Submission and wait boundaries

The pinned HTTP service selects `CaptureDecodeCollectives` in
`cmd/glm_service/checkpoint.mbt`. `compiler/parallel_ir/submission.mbt:128–140`
enables collective-inclusive replay only for `live == 1`. The observed 64-row
prefill therefore uses compute-interval graphs and separate NCCL submissions,
not one collective-inclusive graph for the entire step.

In the current native window the runtime trace contains 92 `cuGraphLaunch`
calls and 91 `cuLaunchKernelEx` calls, consistent with those boundaries.
It also records 273 `cudaStreamWaitEvent` calls with 162.65 ms of summed CPU
API duration, and 132,498 `cuCtxSetCurrent` calls with 5.66 ms summed duration.
These are **profiled CPU API spans**, not another 168 ms of GPU idle time;
they can overlap GPU work, and instrumentation can perturb them. They must not
be added to kernel or collective time. The 6.40 ms uncovered GPU region is a
different measurement, not proof that all submissions are free either.

The collective GPU durations above include peer-arrival waiting and transport.
The rank-0 process launcher has an explicit environment allowlist in
`internal/process/process_approved_spawn.c`; Nsight's injection environment is
not forwarded to its exec child. Its parent trace consequently has no CUDA
events. **Wire time versus rank skew remains unallocated.** No host API duration
or blank trace has been substituted for that missing measurement.

## Failures, memory limits and restoration

All owned GPU units used `MemoryMax=100G`, `MemorySwapMax=0`; planned native
allocation was 103127137948 bytes/rank. This does not by itself reserve host
headroom for an external profiler.

1. `combined-hwtrace-v1`: profiler temp-directory permission failure; resolved
   with a task-owned TMPDIR, without modifying shared-directory permissions.
2. `combined-hwtrace-v2`: diagnostic client's warmup omitted Content-Length;
   HTTP 400, corrected in the external client. Not an inference-kernel failure.
3. `combined-hwtrace-v3`: platform reports CUDA hardware tracing unsupported
   and falls back to software; device-event tracing was enabled in that
   attempt. **Host reserve briefly fell to 15.56 GiB on .179 during shutdown**
   despite the early-stop guard. This violated the requested 16 GiB floor;
   this attempt is not the primary comparison and the violation is retained.
4. `combined-bounded-v4`: explicit software CUDA/node capture with device-event
   tracing off and a 500 ms timer. Lowest sampled host availability across both
   watcher files/ranks was **17796992 KiB = 16.9725 GiB**. Maximum sampled GPU
   usage was **96916 MiB = 94.6445 GiB**; owned units had zero swap. The 17 GiB
   diagnostic guard then retired the remaining long request. No late capture
   or other GPU job was started afterward.

The Nsight collection was stopped before report conversion. Reports were
exported only after diagnostic GPU owners retired; large raw reports remain
remote and only small SQLite exports were collected. Failed captures remain
available; no limits were raised or missing data invented to complete a graph.

**Production GLM was restored.** Both original user units are active; health
is healthy, and a fresh chat request for 17×19 returned `323`. Post-restoration
sampled host availability was 18480204 KiB on .178 and 20194184 KiB on .179,
with zero unit swap. No payload hashing was performed.

External launcher regression tests pass 4/4; the bounded controller and trace
collector pass MoonBit checks, and task-file formatting/diff checks pass. An
existing launcher `StringView.to_string` deprecation warning remains visible;
it was not suppressed. No new full native-suite or GPU-sanitizer result is
claimed: production compiler/kernel/ABI code was not changed in this task.

## Evidence and next measurement boundary

- Current capture: `/tmp/lunaflux-glm-http-combined-bounded-20261019-v4/rank1.sqlite`.
- Current categories: `/tmp/lunaflux-glm-combined-window-analysis-20261019-v2/{summary,passes}.json`.
- Current coverage/API spans: `/tmp/lunaflux-glm-combined-overlap-20261019-v1/`.
- Reference capture: `/tmp/lunaflux-reference-bench-glm-long-egress-trace-20261019-v2/rank1.sqlite`.
- Reference categories: `/tmp/lunaflux-glm-vllm-long-pass-analysis-20261019-v4/regions.json`, **filter pass == 0**.
- Reference coverage: `/tmp/lunaflux-glm-vllm-first-overlap-20261019-v1/`.
- Installed source: `/tmp/lunaflux-glm-selected-vllm-source-20261019-v1/`.
- External tools: `trace-glm-http-window.mbtx`, `trace-glm-long-http.mbtx`,
  `analyze-glm-native-long-passes.mbtx` in `partial` mode, and
  `analyze-gpu-overlap.mbtx`. No comparison enters numerical or compiler IR.

The proven current differences are payload/order at the MoE TP boundary,
substantial projection/HC/KDA work, and observable routed-matrix costs—not
the old long scale scans and not an entirely serial shared branch. The next
missing evidence is a **late-history bounded window plus both-rank collective
arrival/completion**, using a diagnostic child launcher that preserves the
existing wire descriptors and profiler injection, with measured profiler
headroom reserved before resident weights. Do not rerun an unrestricted
whole-request node trace or raise the 100/16 GiB limits. Until that evidence
exists, this report does not claim complete 8K critical-path attribution.

## Follow-up: local combination before TP (implementation plan)

The measured MoE boundary transports eight routed F32 partials and one shared
F32 partial per token. The alternative publishes each rank's expert results
locally, combines them into one BF16 hidden vector, then performs one BF16
sum-allreduce before HC. For 64 rows and 4096 channels this changes the
collective buffer from 9 MiB to 0.5 MiB and removes concatenation pack/unpack.

This is a different finite-precision program, **not** a bitwise-equivalent
reassociation. The numerical IR names the choice; the parallel plan joins both
producers before reduction; the device boundary carries the element type.
Both expert branches must be sharded, never replicated shared contributions.
Existing serving defaults remain unchanged during testing.

Qualification covers source/plan propagation, typed collective/fork boundaries,
bounded GPU correctness, complete answers, and retracing the same scales+bulk
workload against the retained matched vLLM trace. No overall throughput
campaign, payload hashing, or automatic promotion.

### Implemented numerical and effect boundaries

`MoeReductionPlacement::LocalCombinedBFloat16` is an explicit alternative to
the unchanged `ExpertPartialsFloat32` default. Both shared and routed branches
must produce TP-sharded partials. Their existing local publications execute on
the two producer streams, the fork joins, and the final local BF16 combination
precedes one BF16 sum-allreduce. A replicated shared branch is rejected.

The immutable numerical law propagates through the MoE program, recurrent and
sparse GLM FFN, decoder stage, typed reduction boundary, and existing private
NCCL binding. HTTP export and execution consume the same stage plan. No
comparison result, reference output, or vLLM-specific decision enters the
compiler. The existing router/expert precision programs remain separate.

This is not a claim of identical finite-precision arithmetic to either the old
path or vLLM. In particular, the current local path retains per-expert BF16
publication, route weighting, routed-output rounding, and the final local BF16
addition before the inter-rank BF16 sum. Independent complete-answer checks
are required in addition to checking that the declared program executes.

The new typed boundary exposed a startup graph-capture bug: initialization
still multiplied the element count by four. Commit `07d0ca98` makes validation
and initialization use the same element-size query. An exact-size BF16 buffer
regression covers ordinary, padded, and bucketed collective replay; allocating
an oversized F32 buffer would have hidden this bug.

### Qualification evidence before full-model replay

- Full local native suite: **5,327/5,327** passed before the capture-size repair;
  the exact-size capture regression passed afterward. The workspace contains
  unrelated user changes, so this is not described as a clean-worktree suite.
  Existing migration-warning exclusions are unchanged.
  The final full rerun passed **5,324/5,327**; three large DeepSeek fixture
  writes failed with `No space left on device`. Only the regenerable
  `_build/native/debug/test` cache (5.6 GiB) was removed; source, models and
  evidence were retained. A fresh **135/135** `tests/packed_moe_execution`
  rerun then passed, including all three failed fixtures and BF16 capture.
  The final native check also passed with the existing warning exclusions.
- Clean pinned ARM service build at `07d0ca98` succeeded. The new host artifact
  `/tmp/lunaflux-glm-http-local-moe-20261019-v2` reuses the preceding AOT only
  after direct generated-source comparison, without computing payload hashes.
- The actual three exported local publication/combine kernels were extracted
  from `rank1.cu`, not independently reimplemented in the GPU probe. Signed
  TP partials, nontrivial routing weights, permuted expert IDs, two producer
  streams, and live-row sequence **1, 3, 64, 2** are checked against an
  independent CPU execution of the declared rounding program.
- The uninstrumented **two-Spark NCCL** probe matched all active values on both
  ranks (4,096 / 12,288 / 262,144 / 8,192), preserved inactive sentinels, and
  released resources. Its historical root ends in `local-race-20261019-v1`,
  but it was run in `run` mode: **it is not a racecheck result**.
- Actual local kernels, with the NCCL call excluded, passed both-rank
  memcheck/full-leak, racecheck and synccheck under **2 GiB/no owned swap**.
  Evidence roots are `local-mem-20261019-v3`, `local-race-20261019-v2` and
  `local-sync-20261019-v1` beneath `/tmp/lunaflux-expert-tp-`.
- NCCL-inclusive memcheck attempts v1/v2 failed during communicator
  initialization with `cudaErrorNoKernelImageForDevice`, 40 reported errors
  and zero leaks. v2 used the same installed NCCL library as serving. These
  failures are retained; the paired collective is **not claimed sanitizer
  clean**, and no error was suppressed. This is separate from the passing
  exported-kernel checks and the passing uninstrumented paired collective.

Planned device allocation is **102,730,776,220 bytes/rank**, 396,361,728 bytes
(378 MiB) below the previous scales+bulk plan. This is an allocation reduction,
not a measured speedup. Resident qualification retains the 100 GiB device/unit
limit and independent 16 GiB host-availability watcher. Pre-existing global
host swap is not attributed to these no-swap task units.

### Complete-answer qualification of the local-combine candidate

`/tmp/lunaflux-glm-http-qualification-local-moe-20261019-v2` completed using
the source-pinned `07d0ca98` host executable and exact reused AOT. The external
controller used `--correctness-only`: no short/long latency sweeps or overall
performance benchmark ran.

- The **7,973-token** retrieval returned **`orchid-70`**, `finish_reason=stop`,
  with a 128-token output budget.
- All **11 automated** checks in `glm_complete_answers_v1.json` passed. All
  **15 answers terminated normally** with a 512-token budget, rather than
  judging truncated one-token outputs.
- Manual inspection: translation accurately preserves the cat-under-table
  sentence; `return sorted(set(values))` implements the requested integer
  function; the summary preserves all four supplied facts without invention.
- SQL aggregation and tie-breaking are semantically correct, but its Markdown
  fence violates “Return only SQL.” The retained scales+bulk baseline emits
  the identical fenced answer. This remains a **formatting failure**, not a
  new numeric regression or a waived check. Do not call the corpus 15/15 strict
  instruction passes.
- JSON chat, SSE, Responses API, four concurrent routed requests, cancellation
  and slot reuse passed. These are finite service checks, not a broad context/
  concurrency certification or a claim that two arithmetic programs are equal.

The qualification watcher sampled minimum host availability **18,854,556 KiB
(17.9811 GiB)** and maximum GPU allocation **96,528 MiB (94.2656 GiB)** across
the two Sparks. Both owned units had zero swap and retired successfully before
the profiler service started. Full answers and raw responses remain in
`/tmp/lunaflux-glm-broad-policy-local-moe-20261019-v2` and
`/tmp/lunaflux-glm-service-benchmark-policy-local-moe-20261019-v2-answers`.

External `analyze-glm-moe-reduction.mbtx` follows each observed local combine
(or old pack/reference routing finalization) to its next same-stream NCCL
kernel. It rejects intervening unrelated kernels, requires all 42 MoE sites,
and keeps peer-arrival and transport inseparable rather than inventing wire
time. Its dependency regression passes; the retained captures independently
reproduce **35.066048 ms + 4.394656 ms pack/unpack** for the old path and
**8.260864 ms** for vLLM. New-candidate attribution follows the fresh capture,
not these historical values.

### Fresh local-combine trace: the targeted gap is removed

Capture: `/tmp/lunaflux-glm-http-local-moe-trace-20261019-v2/rank1.sqlite`.
It references the qualified `07d0ca98` host/AOT artifact described above, the
same literal request, scales32/bulk transport, row64, context8192/slots8, and
software node tracing. The bounded recording contains exactly one complete
chunk, **185591520–497164480 ns**, followed by a partial chunk which is
excluded. All 42 routed gate/up, 42 down, 42 local combine and 42 BF16 MoE
collective sites are observed. The one-output request completed after capture
stopped; its overall duration is deliberately not used or reported.
The actual scale launch still has grid32/block256; bulk gate/up still has
grid48/block288/static shared78,368 bytes. Thus this is the latest scales+bulk
execution with the new reduction boundary, not an older expert fallback.

| First complete chunk, rank 1 | Previous scales+bulk | Local-combine candidate | Retained vLLM |
|---|---:|---:|---:|
| MoE payload per 64-row site | 9 MiB F32 | **0.5 MiB BF16** | 0.5 MiB BF16 |
| MoE collective kernel time, 42 sites | 35.066048 ms | **6.758272 ms** | 8.260864 ms |
| MoE pack/unpack time | 4.394656 ms / 84 calls | **0 / 0** | no separate counterpart |
| All collective kernel time | 59.750560 ms | 31.469024 ms | 17.670304 ms |
| Non-MoE collective time | 24.684512 ms | 24.710752 ms | 9.409440 ms |
| Routed gate/up sum | 118.793440 ms | 118.573280 ms | 112.605934 ms |
| Routed down sum | 54.680512 ms | 54.560640 ms | 61.313952 ms |
| Shared-only kernel coverage | 0.406113 ms | 0.102432 ms | 4.060640 ms |
| Compute–collective overlap | 0 | 0 | 0 |

The observed MoE collective time falls **80.7%**, or **82.9% including removed
pack/unpack**. In this selected window it is below the retained reference;
this is not a throughput claim or an assertion that all TP overhead is solved.
The native kernel is actually `ncclDevKernel_AllReduce_bf16_RING`, **not** the
reference's `AllReduce_Sum_bf16_RING_LL`: the improvement does not require
assuming an LL protocol switch.

The native embedding-to-embedding interval changes 344.189824 → 311.572960 ms
(-32.616864 ms). Removed MoE TP/packing accounts for 32.702432 ms of measured
kernel spans, while non-MoE collective and routed GEMM times remain close.
That accounting supports the intended mechanism instead of attributing the
gain to an unmeasured general overlap improvement. These are single profiled
windows; their difference is not a repeated latency estimate. Reference outer
anchors still differ, so **do not infer a full-request or model-level gap**
from these intervals.

First MoE, native router origin **206914560 ns**, times in ms:

```mermaid
flowchart LR
    X["FFN input"] --> R["Router + row preparation<br/>0.000–0.211"]
    X --> S["Shared gate/up<br/>0.107–2.216"]
    R --> G["Routed gate/up<br/>0.211–2.627"]
    G --> D["Activation/packing → down<br/>down 2.724–4.051"]
    D --> RP["Local routed publication<br/>4.050–4.078"]
    S --> SD["Shared down/publication<br/>2.214–2.445"]
    RP --> C["Join → local BF16 combine<br/>4.078–4.084"]
    SD --> C
    C --> T["BF16 TP incl. peer wait<br/>4.089–4.231 (0.141 ms)"]
    T --> H["HC / next layer"]
```

Small overlapping endpoints in same-stream trace records (sub-microsecond to
a few microseconds) are profiler timestamp artifacts, not permission to omit
producer joins. The graph's dependency arrows come from the actual effect
plan; the observed launches verify that selection. Shared publication is now
before the join and remains hidden under routed work. No transport pack/unpack
symbols execute. The corresponding first old/reference MoE collective spans
were **0.803072 / 0.160640 ms**, compared with **0.141472 ms** now.

What remains visible, not fixed by this change:

- **Other collectives** remain about 24.71 ms in the native region versus
  9.41 ms in the reference region. Their counts/outer coverage differ slightly;
  this is a next attribution target, not proof that transport alone explains
  that difference. Rank-1 NCCL durations include peer-arrival waits.
- KDA projections 38.312224 ms, sparse projections 21.786400 ms, HC 9.979424 ms
  and dense FFN 6.500672 ms remain in the native graph. This implementation
  does not optimize those kernels.
- GPU uncovered time is 6.314784 ms; shared/other overlap is 108.326624 ms.
  Neither means that every blank interval is avoidable CPU time or that shared
  work has no resource contention. No usable paired-rank arrival measurement
  was added, and no late-history conclusion is drawn from this early chunk.

Trace minimum sampled host availability: **17,948,892 KiB = 17.1174 GiB**;
maximum GPU use: **96,538 MiB = 94.2754 GiB**; owned swap zero. Both diagnostic
units exited successfully and released GPUs before SQLite conversion. The
8.8 MiB SQLite was collected; raw profiler output remained remote. Evidence:
`/tmp/lunaflux-glm-local-moe-window-20261019-v1`,
`/tmp/lunaflux-glm-local-moe-overlap-20261019-v1`, and
`/tmp/lunaflux-glm-moe-{old,local,vllm}-20261019-v1.json`.

### Restoration and deployment status

The original `lunanexa-glm-lunaflux-rank0/rank1` service configuration was
restored after trace conversion. Both units are active; authenticated backend
health returns healthy, and a fresh complete chat answer for 17×19 returns
**`323`**, `finish_reason=stop`. Post-restoration samples: .178 **96,881 MiB
GPU / 18,496,068 KiB host available**; .179 **96,909 MiB GPU / 20,103,072 KiB
host available**; owned swap zero. The original deployed artifact is unchanged.

The new local-combine law is implemented, wired to HTTP, source-pinned and
physically tested, but remains **opt-in** (`moe_reduction=local-combined-bf16`)
rather than silently replacing the service restored at the user's request.
No overall benchmark, payload hashing, new tolerance, reference substitution,
or unrelated workspace migration was introduced by this task.
