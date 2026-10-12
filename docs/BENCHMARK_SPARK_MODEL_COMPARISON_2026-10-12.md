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
image. No fresh comparative performance numbers have been obtained yet.

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
