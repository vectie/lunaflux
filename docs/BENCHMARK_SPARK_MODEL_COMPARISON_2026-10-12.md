# Two-Spark exact-checkpoint comparison

## Fixed experiment scope

Compare LunaFlux, vLLM and SGLang on the installed
GLM-5.3-Flash-NVFP4 and DeepSeek-V4-Flash-DSpark checkpoints, using literal
model-specific chat templates and six prompt categories: facts, arithmetic,
Chinese, constrained JSON, code and retrieval. Greedy generation, concurrency
one, no speculative decoding, no prefix reuse between requests. Retain input
and output token vectors. Eight output tokens is a bounded short-answer probe,
not a general quality evaluation; report truncation and EOS explicitly.

Comparison and observation live in offline scripts/docs, never the compiler.
No payload hashing or integrity scans. Existing declared shard-label metadata
is not recomputed. Preserve unrelated workspace changes and prior results.

Before loading an external engine, inspect its exact installed version,
architecture/quantization support and SM121 backend. An unsupported combination
has no throughput number. Do not replace either checkpoint with a smaller or
differently quantized model. A modified numerical oracle is not a stock-engine
performance baseline.

The hardware is two GB10 Sparks (.178/.179), one GPU per host. Serialize full
model runs across both hosts; cap each process at 96 GiB with no process swap,
monitor host available memory, and keep a 16 GiB host reserve. Do not overlap
the retired MiniMax experiment. Fixed initial budget: one six-category corpus
per engine/model pairing, plus one representative GLM prefill/decode timeline.
Profiled runs are not unprofiled performance measurements.

## Timing boundaries and diagnosis

LunaFlux's current checkpoint CLI is a resident two-stage pipeline, with
single-row prefill. Measure request time after weights are loaded; report
prefill length, actual output length and finish condition alongside elapsed
time. Its full-request time is not decode-only token throughput. External HTTP
request time includes client/server overhead and must be labeled separately.
Do not infer TTFT or individual decode times from buffered journal timestamps.

Use Nsight Systems on both LunaFlux ranks to attribute selected kernel families,
launch geometry, GPU idle intervals, CUDA calls and copies. Inspect the actual
AOT source corresponding to those symbols. Follow up the largest measured
family with counters where available. Distinguish serial pipeline waiting,
single-row prefill, scalar arithmetic, repeated work and device memory stalls;
code inspection alone cannot assign elapsed-time percentages.

## Starting observations (not new benchmark results)

Previous GLM resident two-request smoke took 671,411 / 643,002 ms for 22 / 21
input tokens and eight output tokens each. DeepSeek's chat smoke took 38,807 ms
for 14 input and three output tokens, ending at EOS. These are different
workloads, not a model-to-model speed comparison. The GLM smoke deliberately
disabled EOS; this experiment enables checkpoint EOS 154827 (DeepSeek: 1).

The preinstalled NGC SGLang image is 0.5.7+nv26.1. The separately installed
GLM image contains vLLM 0.1.dev20051+g487ecf187 and registers both exact model
architectures. Registration is not proof that quantization/SM121 execution works.
Upstream also records a [GLM Spark DSA backend issue](https://github.com/sgl-project/sglang/issues/40286).
Inspect a current supported build before drawing conclusions from the old NGC
image. Fresh vLLM GLM measurements are recorded below; the matched native and
SGLang corpus measurements are not yet complete.

## Public recipes found from the user's template table

- [SparkGLM / Atlas](https://github.com/Enntity/sparkglm) uses the NVIDIA NVFP4
  checkpoint, but Atlas is now its engine. It is not a vLLM or SGLang result.
  Its [archived vLLM branch](https://github.com/Enntity/sparkglm/tree/archive/vllm)
  documents an NVFP4 route and a sparse-candidate approximation for long
  contexts; do not silently adopt that approximation in LunaFlux.
- [MiaAI GLM TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  is an EXL3/TensorFold recipe, not the exact NVFP4 engine comparison requested.
- [MiaAI DeepSeek two-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  now defaults to Vision-Exp. Its historical 0731 recipes are useful deployment
  references, but neither checkpoint may replace the installed DSpark package
  without explicitly changing the comparison. Disable drafting for the initial
  matched target-only experiment; speculative results are a separate axis.
- The [GLM SGLang Spark recipe](https://github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark)
  names `lmsysorg/sglang:glm-5.3-flash-arm64`. Only its target-engine build is
  relevant here: no drafter, replacement checkpoint or global host changes.
- The [official DeepSeek Spark cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx)
  names `lmsysorg/sglang:dev-v4f-2dgx-v2`, with SM12x b12x kernels. Its listed
  checkpoint variants do not prove compatibility with our installed DSpark
  package; actual startup and generation must establish that separately.

Docker Hub connections from the Sparks were refused. Per the user's suggestion,
the exact recipe tags are now being pulled directly on both Sparks through
`docker.1ms.run`. DaoCloud returned HTTP 502 on the Spark, though its manifest
was accessible on the Mac. 1ms returned the ARM64 manifests and downloaded
layers. A partial download is not yet an installed/working engine. No system
registry configuration was changed and no extra payload hashing was added.

## Fresh GLM vLLM six-category smoke

Completed root: `/tmp/lunaflux-reference-bench-vllm-glm-20261012-v2`.
Actual vLLM version is `0.1.dev20051+g487ecf187`. Exact installed NVFP4
checkpoint, TP2/PP1 over ConnectX-7, concurrency one, no speculation or prefix
cache. One warm-up was excluded; each following prompt was measured once.
Times below are buffered HTTP completion time, including prefill and generation,
not TTFT or decode-only throughput.

| Category | Input tokens | Output tokens | Seconds | Finish |
| --- | ---: | ---: | ---: | --- |
| Fact | 22 | 4 | 0.385179 | EOS |
| Arithmetic | 23 | 4 | 0.393374 | EOS |
| Chinese | 22 | 8 | 0.656677 | Length |
| JSON | 23 | 8 | 0.674772 | Length |
| Code | 24 | 8 | 0.672520 | Length |
| Retrieval | 36 | 3 | 0.388957 | EOS |

Outputs include Tokyo, 42 and Jon. Chinese, JSON and code are truncated by the
eight-token budget; JSON is not valid completed JSON. This is broader subject
coverage, not broad quality acceptance. Literal input IDs, output text, counts
and finish reasons are retained; the API returned null output-token IDs, so
raw output-vector agreement cannot be claimed.

The engine selected `FLASHINFER_MLA_SPARSE_SM120`, `FLASHINFER_CUTLASS` MoE,
`fp8_ds_mla` KV and breakable CUDA graphs. Native GLM uses BF16 KV, PP2/TP1 and
one-row prefill; these strategy/precision differences must accompany any later
comparison. Loading took about 571 seconds and 88.47 GiB per rank, excluded
from request timing. Its warning that `w1_weight_scale_2` and
`w3_weight_scale_2` differ is retained: this engine is not a correctness oracle.
Both containers stopped and released their GPUs. No ratio against an unmatched
native prompt or historical smoke is reported.

## Measurement setup failures retained

GLM profile v1 failed before ingress execution because Nsight could not create
its shared `/tmp/nvidia/nsight_systems` directory. Its waiting egress was stopped.
The harness now gives each run a private `TMPDIR`; no global permissions changed.
Profile v2 was stopped by the 32 GiB host-reserve monitor during model loading.
This was not an OOM or an inference failure. The small-model reserve was
incompatible with the existing full-GLM 85–95 GiB per-rank arena. Profile v3
uses a 16 GiB host floor with the same 96 GiB process caps and no process swap;
the external monitor stops only the experiment's ranks if that floor is crossed.
GPU allocations on GB10 must be checked through host available memory as well
as cgroups: the cgroup peak alone does not account for the entire GPU arena.

The Docker Hub SGLang v0.5.20 pull timed out. An NGC 26.09 pull was started, then
stopped while establishing the GLM memory envelope. These are setup outcomes,
not SGLang model-performance measurements.

## Selected GLM code: concrete implementation gaps

Profile v3 completed the native request with 16 input tokens and two generated
tokens in 355,986 ms, excluding loading; both ranks explicitly closed their
contexts. Nsight's report conversion is separate from inference time. Loading
the vLLM trial during that conversion was a scheduling mistake: free host
memory on ingress fell to 12.4--13.5 GiB and both independent monitors stopped
their owned runs. Neither checkpoint changed. The vLLM exit 137 followed the
monitor's stop, not a demonstrated model arithmetic failure. The retained raw
captures were incomplete: the egress importer rejected its stream, and the
ingress importer hit its 48 GiB cgroup limit. No kernel percentages can be
claimed from this failed conversion. A retry excludes initialization and
captures only 60 seconds after delays of 180/270 seconds on ingress/egress;
this is explicitly a sample, not a complete request timeline. External loading
now rejects a still-active profiler/importer. Profile v3's completed request
is a profiled observation, not the six-prompt unprofiled comparison.

The exact deployed `stage-0.cu` contains `glm_sparse_3_attention_attention`
with `if (blockIdx.x != 0 || threadIdx.x != 0) return`. The corresponding
`kernels/glm_hybrid_cuda_aot/dsa_sparse_attention_lower.mbt` explicitly emits
`qualification=serial-correctness-only-not-throughput-qualified` and a one-thread
launch. Thus this is selected code, not an unused reference implementation.
For each query/head/key it recomputes the QK dot for maximum, denominator, and
each of 256 value components: 258 dot evaluations instead of sharing the score.
Each loop scans the declared 2,051 selected slots, including invalid slots,
even for short history. The time contribution still requires the trace.

Selected BF16 projection code also uses one thread per output column, with a
serial full-width K reduction and adjacent threads accessing different weight
rows. It does not use the tiled matrix lowering established for other model
paths. Do not confuse this with the older single-thread recurrent projection
renderer, which is NOT the selected AOT projection. The installed vLLM model
uses column/row-parallel linears, fused MoE, chunked KDA prefill and fused
recurrent KDA decode; the current native CLI runs prefill one token at a time.

These are executable-coverage differences. Merely adding numerical IR or
proving arithmetic behavior does not select a throughput implementation.
Comparison remains external to the compiler; neither a timing advantage nor
agreement with another engine proves mathematical/model correctness.

## Recovered first-rank GLM execution sample

Profile v4 also crossed the 16 GiB host floor (10.64 GiB observed on ingress)
when report conversion began while the model remained resident. Its two model
units were stopped; no completed v4 request time is claimed. After both GPUs
were released, a separate CPU-only importer under 32 GiB/no swap recovered
the ingress stream successfully. Egress rejected its incomplete stream; that
failure remains preserved. No model overlapped either recovery attempt.

Root: `/tmp/lunaflux-model-bench-glm-profile-20261012-v4`, ingress report
`recovered-v1.nsys-rep`, downloaded locally without overwrite. Nsight's CSVs
and SQLite query agree on this first-rank sample:

| Selected family | Calls | Summed GPU seconds | Share of kernel time | Grid / block |
| --- | ---: | ---: | ---: | --- |
| Sparse attention | 16 | 25.882088 | 90.425% | 1 block / 1 thread |
| Hyper-connection function | 139 | 2.303595 | 8.048% | 1 block / 1 thread |
| Other kernels | 2,190 | 0.436984 | 1.527% | Various |

The 2,345 kernels sum to 28.622667 seconds within a 46.614597-second span
from first kernel start to last kernel end. These are sampled ingress values,
not a whole-request, two-rank or steady-state-decode attribution. The unsummed
time cannot be assigned to network, scheduler or pipeline waiting without the
missing paired rank timeline. Sparse calls take 1.588--1.656 seconds each;
their measured one-thread launches corroborate the selected source above.
This is direct evidence of a dominant executable implementation gap in this
sample, not evidence that numerical IR inherently forces slow execution.

The CUDA API sample also contains 13,755,938 `cuEventQuery` calls and
13,758,656 `cuCtxSetCurrent` calls. This exceptionally high polling volume
explains why even a bounded sample contains millions of trace records. Its
CPU cost and report-memory cost are separate from the 90.425% sparse GPU share;
do not add these percentages or treat profiled CPU costs as unprofiled latency.

## SGLang images ready and bounded launches (13 October)

At 01:14 CST both recipe images were installed on both Sparks: the GLM image
reports 30.5 GB and the DeepSeek image 33.3 GB. The successful 1ms pulls supersede
the partial-download status above. No additional payload hashes were computed.
Image help and installed source were inspected without GPU access. GLM's image
contains `Glm5NextForConditionalGeneration`; the DSpark image contains the
SM12x `b12x` backend. Model checkpoints and literal prompt IDs are unchanged.

The external MoonBit runner now selects the model-specific backends, limits
prefill to 256 tokens and decode graph capture to batch one, disables FlashInfer
autotuning, and records launch options. GLM uses TileLang DSA and CUTLASS NVFP4
MoE with BF16 KV; DeepSeek uses DSV4 attention and b12x MoE. These are external
engine configuration choices, not compiler comparisons or numerical oracles.
The existing 96 GiB container/no-process-swap limit and 16 GiB host reserve
remain enforced. Initial budget: the same six categories, one excluded warmup,
one measured request each, greedy target-only generation capped at eight tokens.
This is a first compatibility/timing smoke, not a complete quality evaluation.

GLM launch `/tmp/lunaflux-reference-bench-sglang-glm-20261013-v1` reached
distributed initialization and model construction, but not serving readiness.
The monitor observed 12,603,096 KiB available on ingress and 14,182,624 KiB on
egress and stopped both owned ranks. No benchmark timing resulted. The image
automatically disabled prefill CUDA graphs for its KDA hybrid model. Its
configuration, readiness log, memory observations and terminal journals are
preserved. The next launch skips the unused vision encoder with the supported
`--language-only` option and uses expandable allocator segments; the reserve
is not reduced. The similarly named `--language-model-only` is NOT supported
for GLM in this image and is not used.

DeepSeek launch `/tmp/lunaflux-reference-bench-sglang-deepseek-20261013-v1`
started after both GLM ranks released their GPUs. It recognizes the installed
text-only FP4-expert checkpoint and selects FP8 E4M3 KV with page size 256.
No speculative draft is enabled. Completion/results must be recorded separately
from successful image download or model initialization.

DeepSeek did not reach serving readiness either. Ingress stopped at 01:20:57
CST with exit 137. Docker events explicitly show SIGTERM followed 30 seconds
later by SIGKILL from the stop operation; this is not evidence of a native
model exception or a container-OOM event. Its last model messages report
output-A FP8 dequantization and conversion of FP8 shared experts into FP4 fused
MoE weights. Those default reference-engine transformations are retained in
the log, not treated as unchanged arithmetic. Ingress terminal journals were
downloaded to `terminal-0.stdout`; its GPU is released and host available memory
returned to about 117 GiB.

Egress `.179` subsequently stopped responding to SSH, including fresh
connections via the jump host and an SSH-banner check from `.178`; ICMP still
responds. The pending stop and egress GPU release cannot be confirmed. Host
OOM is a possibility, not a verified diagnosis without its logs. No further GPU
work is launched and no reboot or unrelated service mutation was attempted.
The user was asked to check the local console and stop only the named owned
DeepSeek benchmark container. GLM v2 is prepared, NOT launched.

This exposed a flaw in the external monitor: it stopped ranks serially with
a 30-second grace, and wrote observations only after both remote stops returned.
The runner now saves observations first, requests both stops concurrently with
five-second grace and bounded remote waits, and reports unconfirmed stops
explicitly. Regression tests cover a slow rank and an unreachable rank without
delaying/cancelling the other stop. This does not prove that SSH-dependent
monitoring can recover an unresponsive host; host recovery and a reviewed
loading-memory envelope are still required before another physical trial.
There are **no SGLang request timings or quality results** from these launches.

## Update: parallel controls and 16-row prefill, 13 October

A later selected full-model run reduces the same six native requests from
323.475 to **32.981 seconds (9.81x faster)**, with all six native output-token
vectors unchanged. The earlier same-day vLLM baseline remains 3.160961 seconds,
so the remaining completion-time ratio is **10.43x**. Prefill submissions fall
from 150 to 13, total submissions from 179 to 42. Both native ranks exit 0 with
zero process swap; minimum sampled host availability is 22.698 GiB. No hashing.

See [the implementation and full result](GLM_PARALLEL_CONTROLS_PREFILL_2026-10-13.md)
for per-prompt TTFT/completion times, pure-IR responsibilities, changed-kernel
tests and remaining limitations. This update supersedes the native timing in
the historical section below, not its preserved raw observations. vLLM's
algebraically parallel KDA prefill is not yet implemented in LunaFlux.

## Recovered Sparks: matched full-model GLM result, 13 October

The requested fresh retry recovered .179 through a new authenticated SSH
connection. Before loading, both hosts had no GPU owners or running containers
and approximately 117 GiB MemAvailable. No reboot or unrelated service change
was needed. The earlier unavailable-host diagnosis is historical, not current.

Runs were serialized across both GPUs:

- Native: `/tmp/lunaflux-model-bench-glm-retained-20261013-v1`, committed
  `015722c7` executable and retained-score AOT from
  `/tmp/lunaflux-checkpoint-build-glm-sparse-retained-20261013-v1`.
- Fresh vLLM: `/tmp/lunaflux-reference-bench-vllm-glm-20261013-v1`, same retained
  image/version `0.1.dev20051+g487ecf187`, not the earlier frozen timing run.
- Offline report: `comparison-v2.json` in the fresh vLLM root (v1 retained).
  Generated by
  `scripts/summarize-spark-model-comparison.mbtx`, which directly compares all
  six input-ID vectors, preserves output observations, and computes timing
  ratios. No comparison code enters the compiler/runtime. No payload hashes
  or integrity scans were performed.

The exact installed GLM-5.3-Flash-NVFP4 checkpoint, prompt tokens, greedy
sampling, EOS 154827, eight-output-token cap and concurrency one are matched.
All 45 layers execute. Each prompt has one measured sample; this is **not** a
long-context, concurrency-scaling or statistically replicated benchmark.

| Category | Input tokens | Output tokens, both | LunaFlux seconds | vLLM seconds | Completion-time ratio | Finish, both |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fact | 22 | 4 | 45.141 | 0.384751 | 117.3x | EOS |
| Arithmetic | 23 | 4 | 46.950 | 0.388855 | 120.7x | EOS |
| Chinese | 22 | 8 | 52.419 | 0.653645 | 80.2x | Length |
| JSON | 23 | 8 | 54.209 | 0.664909 | 81.5x | Length |
| Code | 24 | 8 | 56.036 | 0.670385 | 83.6x | Length |
| Retrieval | 36 | 3 | 68.720 | 0.398416 | 172.5x | EOS |

Total request time is **323.475 versus 3.160961 seconds: 102.3x**. This is a
major remaining full-model performance gap, despite the isolated sparse-kernel
improvement. It is not a claim that the sparse fix failed to execute.

Timing/strategy boundaries remain important:

- Native timings are resident CLI request completion, including single-row
  prefill, generation, diagnostic completion-file output and request retirement;
  vLLM times are buffered HTTP completion, including client/server overhead.
  Loading is excluded from both. This is not an identical-front-door test or
  an isolated measurement of the GPU kernels.
- Native's first fact request is cold; vLLM has one excluded warm-up. Remaining
  native requests reuse the resident engine. No equal-warmup or confidence
  interval claim is made.
- Native uses PP2/TP1 over the existing TCP pipeline with expanded BF16 KV.
  vLLM uses TP2/PP1 over CX7, compressed `fp8_ds_mla` KV,
  `FLASHINFER_MLA_SPARSE_SM120`, CUTLASS MoE and breakable CUDA graphs. This is
  a matched checkpoint/prompt comparison, **not identical execution strategy
  or identical arithmetic**.
- These are completion times, not TTFT or decode-only token/s. The report's
  output-tokens/request-second fields explicitly include prefill time.

### Actual output observations, not a correctness verdict

Four decoded completions match after removing only the native final EOS
marker that vLLM omits: fact, arithmetic, code and retrieval. Chinese agrees on
Paris but ends with native `。` versus vLLM `（`. The truncated JSON responses
also differ: native begins `Simple JSON request.</think>`; vLLM begins
`User wants JSON output only.</think>`. Neither reaches completed valid JSON
within this cap. The code response is also incomplete. All three capped cases
report length termination even when the visible text forms a sentence.

Native output IDs are retained. vLLM again returns null output-token IDs, so
decoded-text agreement must not be relabeled as token-vector or numerical
parity. The vLLM warning about unequal `w1_weight_scale_2`/`w3_weight_scale_2`
is preserved; this engine is not an unquestionable correctness oracle.

### Remaining implementation work indicated by this run

Native transport epochs record 25, 26, 29, 30, 31 and 38 serial model steps,
179 total. Request time divided by those counts is 1.806–1.808 seconds per
step. This explains why a longer prompt increases native latency even with a
shorter completion; it does not independently separate prefill, decode, host,
transport or GPU costs.

The selected new `stage-0.cu`, not merely a dormant source generator, still
contains `glm_hyper_attn_0_function` with `threadIdx.x != 0` returning. One
active thread computes the RMS over 16,384 elements, all 24 projection dots of
width 16,384, and the 4,096-column collapse. Corresponding attention/FFN
functions remain throughout the decoder. The prior ingress sample measured
this family at 8.048% before removing the 90.425% sparse-attention bottleneck;
those old percentages **are not current full-model attribution**. Its actual
remaining serial implementation and one-row prefill are concrete next targets.
Selected projections, pooling/index and host/pipeline gaps also need fresh
attribution; this turn did not run a new profiler or invent current counters.

### Memory and terminal state

Both campaigns kept the 96 GiB process/container cap, disabled process swap
and retained the 16 GiB host reserve. Minimum sampled host availability across
both ranks was **22.659 GiB native / 16.696 GiB vLLM**. Existing unrelated system
swap usage was not zero; benchmark-process swap is the separate quantity.
Native MemorySwapPeak is zero on both ranks. During vLLM loading, direct
container-cgroup checks showed zero swap and zero OOM events; both memory and
memory-plus-swap limits were 103079215104 bytes. No GPU workload or profiler
overlapped either model.

Native ranks exited 0 with explicit `context_closed=true`. vLLM returned HTTP
200 for its warm-up and all six measured requests before the query runner
requested both containers stop. Rank 0 exited 0; rank 1 needed the configured
five-second SIGTERM-to-SIGKILL escalation and exited 137. Docker events show
SIGTERM at 09:10:04 CST and SIGKILL at 09:10:09, followed by container removal.
The watcher consequently reported a failed terminal unit and attempted another
stop after removal. This is a monitor/shutdown classification limitation, not
an inference failure or an observed host-reserve breach. Its nonzero result
and reference shutdown resource-tracker warnings remain preserved.

Afterward both GPU-owner lists and running-container lists were empty; both
hosts returned to approximately 117 GiB available. No SGLang run was retried.
The offline reporter test and existing native/reference runner tests pass
(1 + 2 + 5 tests); formatting is checked only for the new script. Production
compiler/runtime source and unrelated dirty toolchain/TLS files were untouched
in this benchmark turn. No new full-suite or quality-pass claim is made.
