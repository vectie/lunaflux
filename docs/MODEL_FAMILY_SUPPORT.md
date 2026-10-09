# Model-family admission and execution status

LunaFlux uses two separate meanings of model support:

1. **Semantic admission** means an exact, versioned official profile can be
   recognized, validated, bound to content identity, and projected into typed
   architecture and weight requirements.
2. **Executable support** additionally requires a complete architecture-neutral
   plan vocabulary, weight materializer, admitted AOT kernels, device executor,
   numerical corpus, physical resource-balance evidence, and benchmark gate.

Semantic admission never implies executable support. A missing operation or
storage encoding is a typed startup rejection; LunaFlux does not reinterpret a
new family as Llama or Mistral and does not add a family branch to scheduler or
KV policy.

## Registered profiles

| Family profile | Exact config + typed plan | Exact weight contract | Deterministic reference semantics | Executable today | Missing execution capabilities |
|---|---:|---:|---:|---:|---|
| DeepSeek V4 Flash/Base/Pro/Pro-Base/Flash-0731 | yes | exact coalesced logical schemas for 69,187/69,189/145,116/145,118/72,317 tensors and 887/887/1,249/1,249/931 operations; exact FP8/scale/packed-FP4 byte-layout admission, finite E4M3 and UE8M0 scalar decoding, bounded host loading, checked I64-to-I32 hash-table sidecars, raw weight upload, and exact three-region sidecar upload | MoE/hash routing, compressed indexing, MTP/DSpark order, and exact versioned mHC phases; exact candidates now include replicated Query-A, replicated pre-normalization KeyValue projection, and single-rank Output-B with dynamic per-row/block-128 UE8M0 activation scaling, E4M3 weights, UE8M0 block scales, ordered F32 accumulation, and one BF16-RNE output, plus window/index joining, shared-KV preparation, selected attention, inverse scaled-YaRN, and grouped BF16 Output-A; inert Query-A/KeyValue/Output-A/Output-B joins authenticate official storage/layout evidence without granting interpretation, materialization, upload, collective, or launch authority; Flash-0731 reports 40 requirements, 22 inert candidates, and 18 exact gaps | no | quantized Query-B, actual Query-A/KeyValue/Output-A/Output-B materialization, Query-B and Output-B tensor-parallel collectives, learned compressor projection/gating/APE/RMSNorm/forward-RoPE/QAT, incremental compressor state, live mHC/KV ownership, separately authenticated plain-theta sliding-layer RoPE, offline compilation, loader/launch authority, qualified artifacts, executor/runtime integration |
| MiniMax H3 FL2VA/Ref2VA | yes, immutable joint-diffusion plan and digest-bound I/O plan | exact denoiser (638) mixed F32/BF16, Qwen3-VL (1,058) BF16, VideoVae (703 F32 tensors; 10,415,475,936 payload bytes across three authenticated shards), and AudioVae (1,087 F32 tensors; 605,306,340 payload bytes in one authenticated file); component startup uploads weights with explicit leases and budgets | RF schedules and per-step metadata; text/vision/reference encoders; full DiT and both output decoders; prompt/media row placement; prepared joint request and deterministic source export. These are source/planning/lifecycle implementations, not full-model GPU numerical results; see the [current inventory](MINIMAX_H3_CODE_INTEGRATION_2026-09-26.md) | not physically qualified | compilation/admission of complete artifacts, real-checkpoint end-to-end bootstrap and GPU numerical/sanitizer/memory/performance validation, production worker/serving integration, external media codecs and transport |
| GLM 5.3 full text | yes, immutable hybrid-decoder plan | exact logical BF16/F32 manifest and 960-operation numeric schema; converted BF16/F32 approved-shard host materialization and split-tensor raw device upload; official block-FP8 fail-closed | DSA pooling/reuse, MoE/mHC adapter, exact layer schedule, deterministic BF16 decoder/DSA/dense-SwiGLU candidates, conventional outer residual add, and exact bit-preserving K/V assembly: packed KV-B `[T,64,448]` is split while the rotated shared 64-wide suffix is broadcast into contiguous K `[T,64,256]` and V `[T,64,256]`; 28 requirements, 24 inert candidates, and 4 exact gaps | no | live cache handoff and ownership, throughput-qualified paged-KV attention, remaining decoder numeric kernels, block-FP8 materialization, qualified artifacts, KV/executor/runtime integration |
| GLM 5.3 Flash multimodal | yes, immutable hybrid-decoder and vision plan | exact logical BF16/F32 language/vision manifest and 998-operation numeric schema; approved-shard host materialization and split-tensor raw device upload | DSA, bounded vision plan, exact KDA stages, routed/shared experts, exact staged hyper-connection function/collapse, 20-iteration positive Sinkhorn control, four-stream mHC outer mix, and zero-suffix bit-preserving K/V assembly that splits packed `[T,64,512]` into contiguous K/V `[T,64,256]`; 47 requirements, 32 inert candidates, and 15 exact gaps | no | broad legacy query low-rank projection, zero-work RoPE structure, auxiliary/MTP and vision operations, live recurrent-state/cache handoff and ownership, throughput-qualified paged-KV attention, qualified artifacts, KV/executor/runtime integration |

The aggregate `model/family_catalog` exposes these profiles through
`ModelFamilyRequirements`. A backend supplies a `ModelBackendCapabilities`
value during startup, and admission returns either
`SemanticCapabilitiesAvailable` or the complete ordered
`MissingCapabilities` list. This semantic join grants no artifact, device, or
execution authority and is not available to token-step scheduling.

## Package ownership

Each family has three focused owners:

~~~text
model/<family>_spec/     exact official semantic profile and content-bound metadata
model/<family>/          immutable semantic plan/requirements and backend admission
model/<family>_weights/  exact logical storage/tensor requirements
~~~

Shared code is deliberately small:

~~~text
model/family_contract/   architecture-neutral capability and admission vocabulary
model/family_catalog/    stable aggregate registration; no execution authority
model/advanced_decoder_execution_plan/ family-neutral advanced decoder vocabulary
model/advanced_decoder_kernel_requirements/ ordered semantic/AOT requirements plus complete shared kernel geometry
model/glm_hybrid_execution_plan/ family-neutral hybrid text/vision schedule vocabulary
model/glm_hybrid_kernel_requirements/ ordered hybrid semantic/AOT requirements plus shared decoder geometry
model/joint_diffusion_plan/ architecture-neutral audio/video pipeline plan and bounded shape resolution
model/joint_diffusion_io_plan/ digest-bound normalized media request/result metadata
model/joint_diffusion_kernel_requirements/ ordered diffusion semantic/AOT requirements plus complete request/denoiser geometry
model/workload_execution_plan/ family-neutral aggregate workload summary
model/advanced_decoder_reference/ bounded scalar decoder primitive oracle
model/block_fp8_ue8m0_reference/ finite E4M3/UE8M0 decoding, dynamic activation scaling, and block-128 BF16 boundary oracle
model/glm_hybrid_reference/ bounded scalar DSA/KDA/vision oracle
model/joint_diffusion_reference/ bounded schedule/RNG/layout/stage oracle
model/joint_diffusion_conditioning_reference/ request-bound conditioning geometry and typed fail-closed prerequisite assessment
model/joint_diffusion_transformer_reference/ bounded scalar joint-transformer denoise-step oracle
model/joint_diffusion_vae_reference/ bounded geometry/finite-latent oracle and typed missing-VAE gaps
model/minimax_h3_vae_reference/ thin exact AudioVAE schema-to-family-neutral decoder-prefix adapter
model/joint_audio_amp_parameter_layout/ family-neutral canonical 126-region first-stage AMP parameter arena
model/joint_audio_amp_workspace_layout/ family-neutral 97-stage workspace liveness and non-aliasing plan
model/streaming_safetensors/ family-neutral 64-bit approved-shard inspection and segmented copy
model/checked_integer_narrow/ checked startup-only I64-to-I32 table sidecars with explicit release
model/segmented_host_borrow/ device-free one-shot synchronous host-arena capability
model/segmented_device_materialize/ family-neutral bounded layout, opaque host borrow, upload, and cleanup owner
~~~

Family-owned adapters remain outside shared runtime packages:

~~~text
model/deepseek_v4_kernel_requirements/ exact profile-to-AOT requirement projection
model/deepseek_v4_host_materialization/ exact approved-shard segmented host loading
model/deepseek_v4_device_upload/ thin host-to-generic-device upload projection
model/deepseek_v4_numeric_plan/ exact logical tensor and operation numeric binding
model/deepseek_v4_quantized_payload/ bounded quantized byte-layout admission and reference decode
model/glm53_kernel_requirements/ exact profile-to-hybrid AOT requirement projection
model/glm53_host_materialize/ approved-shard mixed BF16/F32 direct host loading
model/glm53_device_upload/ thin split-tensor host-to-generic-device upload projection
model/glm53_numeric_plan/ exact logical tensor and operation numeric binding
model/minimax_h3_host_materialization/ exact denoiser/conditioner approved-shard host loading
model/minimax_h3_device_upload/ thin component-scoped host-to-generic-device upload projection
model/minimax_h3_kernel_requirements/ exact profile-to-diffusion AOT requirement projection
~~~

Content-addressed artifact evidence has separate family-neutral owners:

~~~text
kernels/advanced_decoder_artifact_admission/ advanced decoder modules, entrypoints, launches, operands
kernels/advanced_decoder_operand_region_plan/ family-neutral bounded allocation-relative operand metadata; never live pointers or launch authority
kernels/decoder_foundation_cuda_source/ family-neutral dynamic, paired, and fixed-row deterministic BF16 embedding/RMSNorm CUDA source
kernels/decoder_compressed_index_cuda_source/ family-neutral bounded deterministic compressed-index counts and padded I32 slots
kernels/decoder_compressed_kv_prepare_cuda_source/ family-neutral borrowed BF16 ordinary/compressed KV layout materialization
kernels/decoder_hyper_connection_cuda_source/ family-neutral BF16/F32 DeepSeek mHC phases plus staged GLM function/collapse and positive Sinkhorn source
kernels/decoder_projection_cuda_source/ family-neutral BF16 no-bias and fixed-row F32+bias dense projection CUDA source
kernels/block_fp8_ue8m0_projection_cuda_source/ family-neutral E4M3/UE8M0 block-128 projection CUDA source
kernels/decoder_token_hash_cuda_source/ family-neutral fail-closed I32 token-hash table lookup source
kernels/decoder_routing_cuda_source/ family-neutral stable biased top-k and selected-weight finalization source
kernels/decoder_scaled_rope_cuda_source/ family-neutral forward and in-place inverse compressed/scaled YaRN BF16 RoPE source
kernels/decoder_selected_sparse_attention_cuda_source/ family-neutral shared-KV selected-index BF16 attention with F32 sink/softmax semantics
kernels/decoder_window_compressed_join_cuda_source/ family-neutral deterministic prefill/decode window and compressed-index concatenation source
model/indexed_parameter_bank_layout/ family-neutral canonical checkpoint-slice-to-packed-bank layout with no materialization authority
kernels/decoder_interleaved_rope_cuda_source/ family-neutral interleaved-pair RoPE with bit-exact query-prefix copy
kernels/decoder_kv_assembly_cuda_source/ family-neutral bit-exact packed-KV split and shared-suffix broadcast source
kernels/advanced_decoder_cuda_aot/ advanced requirement/identity adapters for shared foundation/projection, routing, and scaled-RoPE source
kernels/glm_hybrid_artifact_admission/ hybrid decoder/vision modules, entrypoints, and launches
kernels/glm_hybrid_dsa_pooling_cuda_source/ family-neutral hybrid DSA key-pool compression source
kernels/glm_hybrid_dsa_index_cuda_source/ family-neutral correctness-only projected DSA score, stable-top-k, visible-tail, and index-reuse source
kernels/glm_hybrid_sparse_attention_cuda_source/ family-neutral serial selected-index BF16 sparse-attention oracle source
kernels/glm_hybrid_cuda_aot/ GLM requirement/identity adapter for shared embedding/RMSNorm/language-head/DSA projections/output, KV split/RMSNorm, interleaved RoPE, pooling, index-selection, and sparse-attention source
kernels/joint_diffusion_artifact_admission/ diffusion modules, entrypoints, launches, and operands
kernels/affine_modulation_cuda_source/ family-neutral single/multirow final-AdaLN parameter, packed modality gather, and fixed-row staged-BF16 modulation-to-F32 source
kernels/causal_short_convolution_cuda_source/ family-neutral CSR-bounded causal depthwise BF16 convolution with explicit history handoff
kernels/channel_major_projection_cuda_source/ family-neutral channel-major F32 pointwise, weight-normalized Conv1d, and weight-normalized ConvTranspose1d source
kernels/alias_free_activation_cuda_source/ family-neutral ordered F32 anti-aliased upsample, SnakeBeta, and downsample source
kernels/joint_audio_amp_cuda_source/ family-neutral ordered F32 residual and three-way-average source
kernels/hyper_connection_mix_cuda_source/ family-neutral four-stream BF16/F32 residual-mix source
kernels/router_projection_cuda_source/ family-neutral F32-input/BF16-weight bias-free projection source
kernels/decay_control_cuda_source/ family-neutral BF16-logit/F32-parameter safe decay-control source
kernels/gated_mlp_cuda_source/ family-neutral packed or separate-weight BF16 SwiGLU and dense-down sources
kernels/gated_rms_norm_cuda_source/ family-neutral sigmoid-gated per-head BF16 RMSNorm source
kernels/grouped_routing_cuda_source/ family-neutral corrected-choice group/expert selection and uncorrected-weight normalization source
kernels/selected_expert_mlp_cuda_source/ family-neutral selected-expert BF16 SwiGLU and weighted accumulation source
kernels/bf16_binary_cuda_source/ family-neutral non-aliasing BF16 binary-add source
kernels/joint_attention_cuda_source/ family-neutral staged QKV, Q/K normalization, optional rotary, full attention, and output source
kernels/aot_candidate_coverage/ family-neutral exact-gap and inert/bindable structural coverage evidence; never execution authority
kernels/<workload>_candidate_evidence/ separate family-neutral typed-candidate projections into structural coverage evidence
kernels/cuda_offline_compile_evidence/ family-neutral deterministic receipt/module self-consistency admission with no producer or execution authority
kernels/joint_diffusion_cuda_aot/ request-shape-specialized timestep conditioning composition, transformer attention and SwiGLU MLP, mixed-precision input/output projection, final-RMSNorm, final-AdaLN parameter/boundary, packed modality gather, and shifted-flow source/recipes
kernels/prefix_rms_norm_cuda_source/ family-neutral BF16 prefix RMSNorm with optional bit-exact suffix-copy source
kernels/recurrent_delta_cuda_source/ family-neutral normalized recurrent-delta correctness source with explicit F32 state handoff
integration/deepseek_v4_output_b_parameter_plan/ inert official Output-B storage/layout-to-candidate join with explicit collective gap
integration/deepseek_v4_key_value_parameter_plan/ inert official replicated KeyValue storage/layout-to-candidate join
integration/deepseek_v4_query_a_parameter_plan/ inert replicated Query-A storage/layout-to-candidate join
kernels/recurrent_attention_projection_cuda_source/ family-neutral KDA Q/K/V, forget-control, and beta projection source
kernels/sigmoid_correction_cuda_source/ family-neutral F32 sigmoid with distinct uncorrected and correction-biased score outputs
kernels/timestep_embedding_cuda_source/ family-neutral fixed-row F32 cosine/sine frequency and ordered dense-SiLU-dense source
kernels/two_stage_projection_cuda_source/ family-neutral bias-free BF16 low-rank projection source
integration/deepseek_v4_token_hash_operand_plan/ standalone inert join of live hash sidecars, complete four-operand region metadata, exact token-hash candidate ABI, artifact evidence, and compile self-consistency evidence; full-model artifact admission remains fail-closed at Query-B
integration/deepseek_v4_output_a_parameter_plan/ inert join of official Output-A conversion/layout evidence to exact candidate operand 2; conversion and runtime authority remain absent
integration/glm53_moe_expert_bank_plan/ inert join of exact per-expert checkpoint-bank layouts to routed/shared MoE candidate operands and recipes
tests/deepseek_v4_cuda_aot_integration/ official five-profile query-phase, decoder-head, mHC, deterministic-index/join, borrowed compressed-KV preparation, selected sparse-attention, inverse scaled-YaRN, expert-score, and token-hash proof
tests/glm53_cuda_aot_integration/ official full/Flash decoder-head, DSA, dense-SwiGLU, MoE router/sigmoid/correction/grouped-selection chain, KDA, pooling/index geometry, and manifest/numeric join proof
tests/minimax_h3_cuda_aot_integration/ official FL2VA/Ref2VA conditioning-gap, transformer, VAE prefix, AudioVAE `conv_pre`, first transposed convolution, and first alias-free AMP activation proof
~~~

These packages authenticate and canonically join inert evidence only. Their
status is always non-runnable, caller-declared qualification digests are not
authority, and each `require_execution_authority` operation fails. An opaque
qualification authority, admitted loader, launch ABI implementation, device
executor, physical correctness campaign, and benchmark gate still have to be
added before any family can move to executable support.

Semantic, plan, reference, and host-loading packages do not import scheduler,
KV, API, CUDA, device execution, or kernel implementation owners. The three
thin upload adapters import only their family host owner, the family-neutral
segmented upload owner, and the public device context. The shared upload owner
never imports a family package, and core device/runtime packages never import
DeepSeek, MiniMax, or GLM.

## Primary profile evidence

- [DeepSeek V4 Flash-0731 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json)
- [DeepSeek V4 reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py)
- [DeepSeek V4 sparse-attention kernel reference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/kernel.py)
- [MiniMax H3 official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [MiniMax H3 pipeline index](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/model_index.json)
- [MiniMax H3 official Diffusers transformer](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_minimax_h3.py)
- [MiniMax H3 official latent packing](https://github.com/huggingface/diffusers/blob/main/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py)
- [MiniMax H3 official VideoVAE implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py)
- [MiniMax H3 official AudioVAE implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py)
- [MiniMax H3 official decoder pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/modular_pipelines/minimax_h3/decoders.py)
- [GLM 5.3 full configuration](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)
- [GLM 5.3 Flash configuration](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/blob/main/config.json)
- [GLM 5 Next official implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py)

## Promotion rule

A profile moves to executable support only when every declared semantic and
numeric capability has one typed generic plan representation, exact weight
binding/materialization, AOT catalog and launch contract, deterministic
reference/correctness corpus, physical sanitizer/resource-balance campaign,
and benchmark evidence. The capability set changes only after those components
exist; documentation or profile recognition alone cannot make admission pass.
