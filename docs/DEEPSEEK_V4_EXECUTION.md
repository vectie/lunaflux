# DeepSeek V4 executable-support checklist

This checklist defines what “DeepSeek V4 support” must mean before LunaFlux may
advertise executable inference. Recognition, semantic admission, or a typed
model plan alone is not executable support.

## Exact source contract

The schema-v1 profiles are pinned to the official configuration documents for
DeepSeek V4 Flash, Flash Base, Pro, Pro Base, and Flash-0731. In particular,
the execution contract includes low-rank query/output projections, grouped
output projection, query/key normalization, the exact `compress_ratios`
schedule, learned-index ratio-4 compression, deterministic ratio-128
compression, four-stream hyper-connections, token-hash routing in the first
three MoE layers, score-based routing thereafter, MTP, and the Flash-0731
DSpark attachment.

Primary configuration evidence:

- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json>
- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json>
- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json>
- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/blob/main/config.json>
- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base/blob/main/config.json>
- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py>
- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/kernel.py>

## Completion evidence

Current executable integration (2026-10-10): the older inert catalog below is
not the only execution path. `integration/deepseek_v4_query_kv` now binds and
streams five actual compact checkpoint banks through Query-A, learned query
norm, Query-B, BF16 head norm, shared KV and learned KV norm. Its
`DeepSeekAttentionInputs` owner appends full-head rotary and KV simulation to
the same ordered executor. A backend-neutral precision plan distinguishes
compression-zero base RoPE from compressed-layer YaRN and excludes the rotary
suffix from block-64 E4M3/power-of-two simulation. Outputs remain BF16; these
effects do not own or commit persistent KV state. GPU component correctness,
determinism and three sanitizers pass on GB10; no whole-checkpoint generation
or throughput result follows. Current detailed results are in `STATUS.md`.

The subsequent `DeepSeekWindowAttention` owner now binds the actual learned
sink and commits request-owned retained window state for compression-zero
layers. Twelve prepared launches preserve one containing queue and exact
startup accounting. Chunked prefill larger than the retained window, successive
decode, reset and invalid-position isolation have component GPU/sanitizer
coverage. Its output remains before inverse RoPE and learned output projections;
compressed layers are rejected rather than replaced with window-only attention.
The full compressed-state plan and complete decoder remain unfinished.

The subsequent `DeepSeekWindowSublayer` joins inverse suffix RoPE, grouped
Output-A and Output-B to those twelve effects, producing hidden-width output
in one fifteen-stage prepared queue. Generic immutable precision plans own
geometry, numerical semantics and scratch budgets; CUDA source owns lowering,
and the model adapter alone supplies checkpoint names. Output-A's decoded FP8
parameter is rounded to BF16 before multiplication; Output-B instead uses
dynamic block-128 FP8 activation scaling. The GB10 small-fixture correctness,
deterministic replay and memory/race/synchronization checks pass. This closes
the single-rank compression-zero attention/output slice, not the learned
compressor/indexer, compressed attention, mHC decoder, or whole-model runner.

| Boundary | Evidence required for executable support | Current state |
|---|---|---|
| Configuration | Bounded JSON parser rejects unknown fields and admits only exact architecture/model-type/profile geometry | Implemented in the family-owned exact config parser for all five profiles |
| Semantic plan | Immutable ordered plan binds all base-layer attention, low-rank geometry, YaRN, mHC, hash/score MoE routing, MTP/DSpark, and model content into a deterministic digest | Implemented for the base decoder plus typed MTP/DSpark attachment; mHC carries separate control and normalization epsilons |
| Tokenizer | Approved tokenizer assets produce token IDs and decoded output matching a pinned official corpus | Not yet qualified |
| Tensor vocabulary | Exact tensor names, shapes, dtypes, quantization metadata, aliases, and unexpected/missing tensor rejection for every profile | Exact manifest binding and content+plan identity admission implemented |
| Materialization | Bounded safetensors streaming places BF16, FP8/UE8M0, packed FP4, and routing-index data into final aligned regions with deterministic cleanup | Authenticated raw host loading and family-neutral raw device upload implemented; the three official I64 hash tables narrow directly from the final host arena into checked, unique-row I32 sidecars with explicit zeroizing release and upload through an exact canonical three-region segmented layout; UE8M0 scalar interpretation is proven, while Output-A conversion, Output-B device materialization, FP4 interpretation, and launch binding remain unavailable |
| Reference execution | Architecture-neutral host oracle implements low-rank Q/K/V/O, CSA/HCA index selection, mHC mixing, exact MoE routing/normalization/SwiGLU, MTP, and DSpark | Exact versioned mHC block-control, pre-reduce, post-combine, and head-reduce phases implement normalized control projection, sigmoid gates, stable row softmax, 20 alternating Sinkhorn normalizations, and attention/FFN site distinctions; other primitive fixtures and one bounded composed block are implemented, while the full-profile low-rank/logit oracle remains incomplete |
| KV/state plan | Explicit layouts and capacity bounds cover sliding, compressed, and heavily compressed attention without scheduler family branches | Not implemented |
| Numeric contract | Every operation and tensor has exact storage, conversion, accumulation, ordering, and scale semantics bound into startup identity | Exact logical operation/tensor binding implemented for 887/887/1,249/1,249/931 operations; mHC control weights are F32 and activation boundaries are BF16; Output-B E4M3/UE8M0 block-128 semantics are exact, while packed FP4 and the broader query-path physical kernels remain incomplete |
| Kernel catalog | Content-addressed AOT artifacts cover every admitted operation/shape/profile and reject partial catalogs at startup | Deterministic non-bindable BF16 embedding, replicated block-FP8 Query-A and pre-normalization KeyValue, paired Q/K RMSNorm, final RMSNorm, language head, scaled-YaRN, routing, compressed-index/KV preparation, selected attention, grouped Output-A, block-FP8 Output-B, and all four mHC phases cover bounded slices of all five profiles. Query-A fixes `4096→1024` for Flash profiles and `7168→1536` for Pro profiles; KeyValue fixes `hidden→512`; Output-B is exact for one rank. The block-FP8 projections use dynamic per-row/block-128 UE8M0 activation scaling, E4M3 weights, UE8M0 block scales, ordered F32 accumulation, and one BF16-RNE output. Separate inert startup joins authenticate Query-A, KeyValue, Output-A conversion, and Output-B storage/layout against their candidate operands without granting interpretation, conversion, materialization, upload, tensor-parallel collective, or launch authority. Query-B remains unsupported. Flash-0731 structural coverage is 40 requirements, 22 inert candidates, and 18 exact gaps. Standalone token-hash integration retains artifact and deterministic compile self-consistency evidence while refusing producer, compiler-execution, loader, launch, and physical authority. Complete artifact admission now accepts KeyValue and rejects Query-B, so this evidence does not bind a full model artifact |
| Device planning | Activation, workspace, expert-routing scratch, indices, hyper-connection streams, MTP/DSpark, and persistent state have overflow-safe bounded layouts | Not implemented |
| Worker execution | Startup bootstrap binds model, weight, numeric, plan, kernel, and live-device identities; token steps allocate nothing and perform no filesystem/crypto work | Not implemented |
| Correctness | Deterministic prefill/decode logits and token sequences match an independent official implementation over hostile shapes and long-context transitions | Not implemented |
| Physical qualification | Sanitizer, leak, deterministic rebuild, CUDA correctness, graph-capture, and benchmark gates pass on each promoted device/profile | Not implemented |

The learned compressor now has a model-neutral per-feature gated-pooling plan,
five CUDA effects and a prepared persistent-state frame. It preserves F32
BF16-parameter projections, learned position biases, preceding-window overlap,
BF16 pooled publication and learned F32 RMSNorm. Small GB10 fixtures for ratio
4 overlap and ratio 128 pass chunk/decode state, output and sanitizer checks.
`DeepSeekCompressorPool` now connects each layer's attention or indexer pool to
the actual BF16 `wkv`/`wgate`, F32 `ape` and BF16 learned norm parameters. It
streams into final device banks with a single containing queue and checked
weights/state/frame accounting; no host F32 weight expansion is introduced.
Output RoPE/cache quantization, compressed-cache publication and learned index
query/scoring/selection remain open. Checkpoint-plane fixture binding is not a
whole-checkpoint numerical result.

The next executable integration boundary is compressed output transformation,
indexer selection and request-owned cache-state mutation, followed by
attention, mHC and complete decoder composition. Inverse rotary and learned
output projections are now connected for compression-zero layers as above.
The unsharded query-prefix component above does not implement Query-B/Output-B
tensor-parallel collectives. Compression and cache mutation feed
shared-KV preparation and live mHC/KV
buffer orchestration around the selected-attention candidate. Authenticated offline-builder
and compiler-execution provenance, live loader/launch ownership for inert
uploaded plans, qualified AOT implementations for already-declared numeric
operations, and full-profile differential fixtures also remain open. Official
`torch.topk` tie parity remains
explicitly unqualified because PyTorch does not promise stable tied indices;
LunaFlux fixes lower expert IDs deterministically. Indexed attention,
hyper-connections, and MoE must not be treated as dense Llama operations.
