#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

plan_packages="
$root/model/advanced_decoder_execution_plan
$root/model/advanced_decoder_kernel_requirements
$root/model/glm_hybrid_execution_plan
$root/model/glm_hybrid_kernel_requirements
$root/model/joint_audio_amp_parameter_layout
$root/model/joint_audio_amp_workspace_layout
$root/model/joint_diffusion_plan
$root/model/joint_diffusion_io_plan
$root/model/joint_diffusion_kernel_requirements
$root/model/checked_integer_narrow
$root/model/streaming_safetensors
$root/model/workload_execution_plan
"

family_catalog_packages="
$root/model/family_contract
$root/model/family_catalog
"

family_packages="
$root/model/deepseek_v4_spec
$root/model/deepseek_v4
$root/model/deepseek_v4_config
$root/model/deepseek_v4_host_materialization
$root/model/deepseek_v4_kernel_requirements
$root/model/deepseek_v4_numeric_plan
$root/model/deepseek_v4_quantized_payload
$root/model/deepseek_v4_weights
$root/model/glm53_spec
$root/model/glm53
$root/model/glm53_config
$root/model/glm53_host_materialize
$root/model/glm53_kernel_requirements
$root/model/glm53_numeric_plan
$root/model/glm53_weights
$root/model/minimax_h3_spec
$root/model/minimax_h3
$root/model/minimax_h3_config
$root/model/minimax_h3_host_materialization
$root/model/minimax_h3_kernel_requirements
$root/model/minimax_h3_weights
"

reference_packages="
$root/model/advanced_decoder_reference
$root/model/block_fp8_ue8m0_reference
$root/model/deepseek_v4_reference_fixture
$root/model/glm_hybrid_reference
$root/model/joint_diffusion_conditioning_reference
$root/model/joint_diffusion_reference
$root/model/joint_diffusion_transformer_reference
$root/model/joint_diffusion_vae_reference
$root/model/minimax_h3_vae_reference
"

artifact_packages="
$root/kernels/advanced_decoder_artifact_admission
$root/kernels/glm_hybrid_artifact_admission
$root/kernels/joint_diffusion_artifact_admission
"

compile_evidence_packages="
$root/kernels/cuda_offline_compile_evidence
"

operand_region_packages="
$root/kernels/advanced_decoder_operand_region_plan
"

parameter_layout_packages="
$root/model/indexed_parameter_bank_layout
"

candidate_coverage_packages="
$root/kernels/aot_candidate_coverage
"

candidate_evidence_packages="
$root/kernels/advanced_decoder_candidate_evidence
$root/kernels/glm_hybrid_candidate_evidence
$root/kernels/joint_diffusion_candidate_evidence
"

aot_candidate_packages="
$root/kernels/advanced_decoder_cuda_aot
$root/kernels/glm_hybrid_cuda_aot
$root/kernels/joint_diffusion_cuda_aot
"

decoder_foundation_aot_candidate_packages="
$root/kernels/advanced_decoder_cuda_aot
$root/kernels/glm_hybrid_cuda_aot
"

diffusion_aot_candidate_packages="
$root/kernels/joint_diffusion_cuda_aot
"

shared_cuda_source_packages="
$root/kernels/alias_free_activation_cuda_source
$root/kernels/affine_modulation_cuda_source
$root/kernels/bf16_binary_cuda_source
$root/kernels/block_fp8_ue8m0_projection_cuda_source
$root/kernels/causal_short_convolution_cuda_source
$root/kernels/channel_major_projection_cuda_source
$root/kernels/decay_control_cuda_source
$root/kernels/decoder_foundation_cuda_source
$root/kernels/decoder_compressed_index_cuda_source
$root/kernels/decoder_compressed_kv_prepare_cuda_source
$root/kernels/decoder_hyper_connection_cuda_source
$root/kernels/decoder_interleaved_rope_cuda_source
$root/kernels/decoder_kv_assembly_cuda_source
$root/kernels/decoder_routing_cuda_source
$root/kernels/decoder_scaled_rope_cuda_source
$root/kernels/decoder_selected_sparse_attention_cuda_source
$root/kernels/decoder_token_hash_cuda_source
$root/kernels/decoder_window_compressed_join_cuda_source
$root/kernels/gated_mlp_cuda_source
$root/kernels/gated_rms_norm_cuda_source
$root/kernels/grouped_routing_cuda_source
$root/kernels/hyper_connection_mix_cuda_source
$root/kernels/glm_hybrid_dsa_index_cuda_source
$root/kernels/glm_hybrid_dsa_pooling_cuda_source
$root/kernels/glm_hybrid_sparse_attention_cuda_source
$root/kernels/joint_attention_cuda_source
$root/kernels/joint_audio_amp_cuda_source
$root/kernels/prefix_rms_norm_cuda_source
$root/kernels/recurrent_delta_cuda_source
$root/kernels/timestep_embedding_cuda_source
$root/kernels/recurrent_attention_projection_cuda_source
$root/kernels/router_projection_cuda_source
$root/kernels/selected_expert_mlp_cuda_source
$root/kernels/sigmoid_correction_cuda_source
$root/kernels/two_stage_projection_cuda_source
"

decoder_projection_source_packages="
$root/kernels/decoder_projection_cuda_source
"

device_bridge_packages="
$root/model/segmented_device_materialize
"

host_borrow_packages="
$root/model/segmented_host_borrow
"

family_device_adapter_packages="
$root/model/deepseek_v4_device_upload
$root/model/glm53_device_upload
$root/model/minimax_h3_device_upload
"

integration_packages="
$root/tests/deepseek_v4_cuda_aot_integration
$root/tests/glm53_cuda_aot_integration
$root/tests/minimax_h3_cuda_aot_integration
"

composition_packages="
$root/integration/deepseek_v4_key_value_parameter_plan
$root/integration/deepseek_v4_query_a_parameter_plan
$root/integration/deepseek_v4_output_a_parameter_plan
$root/integration/deepseek_v4_output_b_parameter_plan
$root/integration/deepseek_v4_token_hash_operand_plan
$root/integration/glm53_moe_expert_bank_plan
"

deepseek_output_a_composition_packages="
$root/integration/deepseek_v4_output_a_parameter_plan
"

deepseek_query_a_composition_packages="
$root/integration/deepseek_v4_query_a_parameter_plan
"

deepseek_key_value_composition_packages="
$root/integration/deepseek_v4_key_value_parameter_plan
"

deepseek_output_b_composition_packages="
$root/integration/deepseek_v4_output_b_parameter_plan
"

deepseek_token_hash_composition_packages="
$root/integration/deepseek_v4_token_hash_operand_plan
"

glm_moe_composition_packages="
$root/integration/glm53_moe_expert_bank_plan
"

validated_packages="
model/advanced_decoder_execution_plan
model/advanced_decoder_kernel_requirements
model/advanced_decoder_reference
model/block_fp8_ue8m0_reference
model/checked_integer_narrow
model/deepseek_v4_reference_fixture
model/glm_hybrid_execution_plan
model/glm_hybrid_kernel_requirements
model/glm_hybrid_reference
model/joint_audio_amp_parameter_layout
model/joint_audio_amp_workspace_layout
model/joint_diffusion_conditioning_reference
model/joint_diffusion_plan
model/joint_diffusion_io_plan
model/joint_diffusion_kernel_requirements
model/joint_diffusion_reference
model/joint_diffusion_transformer_reference
model/joint_diffusion_vae_reference
model/minimax_h3_vae_reference
model/segmented_device_materialize
model/segmented_host_borrow
model/streaming_safetensors
model/workload_execution_plan
model/family_contract
model/family_catalog
model/deepseek_v4_spec
model/deepseek_v4
model/deepseek_v4_config
model/deepseek_v4_device_upload
model/deepseek_v4_host_materialization
model/deepseek_v4_kernel_requirements
model/deepseek_v4_numeric_plan
model/deepseek_v4_quantized_payload
model/deepseek_v4_weights
model/glm53_spec
model/glm53
model/glm53_config
model/glm53_device_upload
model/glm53_host_materialize
model/glm53_kernel_requirements
model/glm53_numeric_plan
model/glm53_weights
model/minimax_h3_spec
model/minimax_h3
model/minimax_h3_config
model/minimax_h3_device_upload
model/minimax_h3_host_materialization
model/minimax_h3_kernel_requirements
model/minimax_h3_weights
kernels/advanced_decoder_artifact_admission
kernels/advanced_decoder_operand_region_plan
model/indexed_parameter_bank_layout
kernels/advanced_decoder_cuda_aot
kernels/aot_candidate_coverage
kernels/advanced_decoder_candidate_evidence
kernels/cuda_offline_compile_evidence
kernels/alias_free_activation_cuda_source
kernels/affine_modulation_cuda_source
kernels/bf16_binary_cuda_source
kernels/block_fp8_ue8m0_projection_cuda_source
kernels/causal_short_convolution_cuda_source
kernels/channel_major_projection_cuda_source
kernels/decay_control_cuda_source
kernels/decoder_foundation_cuda_source
kernels/decoder_compressed_index_cuda_source
kernels/decoder_compressed_kv_prepare_cuda_source
kernels/decoder_hyper_connection_cuda_source
kernels/decoder_interleaved_rope_cuda_source
kernels/decoder_kv_assembly_cuda_source
kernels/decoder_projection_cuda_source
kernels/decoder_routing_cuda_source
kernels/decoder_scaled_rope_cuda_source
kernels/decoder_selected_sparse_attention_cuda_source
kernels/decoder_token_hash_cuda_source
kernels/decoder_window_compressed_join_cuda_source
kernels/gated_mlp_cuda_source
kernels/gated_rms_norm_cuda_source
kernels/grouped_routing_cuda_source
kernels/hyper_connection_mix_cuda_source
kernels/glm_hybrid_artifact_admission
kernels/glm_hybrid_candidate_evidence
kernels/glm_hybrid_cuda_aot
kernels/glm_hybrid_dsa_index_cuda_source
kernels/glm_hybrid_dsa_pooling_cuda_source
kernels/glm_hybrid_sparse_attention_cuda_source
kernels/joint_attention_cuda_source
kernels/joint_audio_amp_cuda_source
kernels/joint_diffusion_artifact_admission
kernels/joint_diffusion_candidate_evidence
kernels/joint_diffusion_cuda_aot
kernels/prefix_rms_norm_cuda_source
kernels/recurrent_delta_cuda_source
kernels/timestep_embedding_cuda_source
kernels/recurrent_attention_projection_cuda_source
kernels/router_projection_cuda_source
kernels/selected_expert_mlp_cuda_source
kernels/sigmoid_correction_cuda_source
kernels/two_stage_projection_cuda_source
integration/deepseek_v4_query_a_parameter_plan
integration/deepseek_v4_key_value_parameter_plan
integration/deepseek_v4_output_a_parameter_plan
integration/deepseek_v4_output_b_parameter_plan
integration/deepseek_v4_token_hash_operand_plan
integration/glm53_moe_expert_bank_plan
tests/deepseek_v4_cuda_aot_integration
tests/glm53_cuda_aot_integration
tests/minimax_h3_cuda_aot_integration
"

for package in $plan_packages $family_packages $reference_packages $parameter_layout_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|device("|/)|engine|kernels|internal/cuda)' \
    "$package/moon.pkg" >/dev/null; then
    echo "advanced model package crosses a runtime/native boundary: $package" >&2
    exit 1
  fi
done

for layout in \
  "$root/model/joint_audio_amp_parameter_layout:JointAudioAmpParameterLayout::new" \
  "$root/model/joint_audio_amp_workspace_layout:first_stage_amp_workspace_layout" \
  "$root/model/joint_audio_amp_workspace_layout:amp_workspace_layout" \
  "$root/model/indexed_parameter_bank_layout:build_indexed_parameter_bank_layout"
do
  package=${layout%%:*}
  symbol=${layout#*:}
  if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
    echo "advanced parameter/workspace layout lost its checked constructor: $package ($symbol)" >&2
    exit 1
  fi
  if ! rg -F 'digest' "$package" --glob '*.mbt' >/dev/null; then
    echo "advanced parameter/workspace layout lost canonical identity: $package" >&2
    exit 1
  fi
done

for symbol in \
  'build_moe_expert_bank_layout' \
  'Glm53MoeExpertBankLayout::materialization_authority'
do
  if ! rg -F "$symbol" "$root/model/glm53_weights" --glob '*.mbt' >/dev/null; then
    echo "GLM weights lost exact inert MoE expert-bank planning: $symbol" >&2
    exit 1
  fi
done

for package in $family_catalog_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|device("|/)|engine|kernels|internal/cuda)' \
    "$package/moon.pkg" >/dev/null; then
    echo "model-family semantic catalog crossed a runtime/native boundary: $package" >&2
    exit 1
  fi
done

for symbol in 'SemanticCapabilitiesAvailable' 'MissingCapabilities'
do
  if ! rg -F "$symbol" "$root/model/family_contract" --glob '*.mbt' \
    >/dev/null; then
    echo "model-family admission lost truthful semantic status: $symbol" >&2
    exit 1
  fi
done

for package in $decoder_projection_source_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|device("|/)|engine|internal/cuda|model/)' \
    "$package/moon.pkg" >/dev/null; then
    echo "shared decoder projection source acquired model/runtime authority: $package" >&2
    exit 1
  fi
  for symbol in \
    'dense_projection_bf16_source' \
    'dense_projection_f32_bias_bf16_source' \
    'dense_projection_f32_bias_f32_source' \
    'grouped_projection_bf16_source'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "shared decoder projection source lost a dense renderer: $package ($symbol)" >&2
      exit 1
    fi
  done
done

for package in $shared_cuda_source_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|device("|/)|engine|internal/cuda|model/)' \
    "$package/moon.pkg" >/dev/null; then
    echo "shared decoder CUDA source acquired model/runtime authority: $package" >&2
    exit 1
  fi
done

for renderer in \
  "$root/kernels/bf16_binary_cuda_source:bf16_add_source" \
  "$root/kernels/block_fp8_ue8m0_projection_cuda_source:block_fp8_ue8m0_projection_cuda_source" \
  "$root/kernels/decoder_kv_assembly_cuda_source:decoder_kv_assembly_bf16_source" \
  "$root/kernels/decoder_hyper_connection_cuda_source:staged_function_and_collapse_bf16_f32_source" \
  "$root/kernels/decoder_hyper_connection_cuda_source:staged_positive_sinkhorn_f32_source" \
  "$root/kernels/hyper_connection_mix_cuda_source:hyper_connection_mix_bf16_source" \
  "$root/kernels/joint_audio_amp_cuda_source:f32_residual_and_three_way_average_source" \
  "$root/kernels/selected_expert_mlp_cuda_source:selected_expert_swiglu_bf16_source" \
  "$root/kernels/channel_major_projection_cuda_source:f32_weight_normalized_dilated_conv1d_source"
do
  package=${renderer%%:*}
  symbol=${renderer#*:}
  if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
    echo "shared CUDA source lost an exact advanced renderer: $package ($symbol)" >&2
    exit 1
  fi
done

for symbol in \
  'token_embedding_bf16_source' \
  'rms_norm_bf16_source' \
  'paired_rms_norm_bf16_source' \
  'rms_norm_bf16_fixed_rows_source'
do
  if ! rg -F "$symbol" "$root/kernels/decoder_foundation_cuda_source" \
    --glob '*.mbt' >/dev/null; then
    echo "shared decoder CUDA source lost a foundation renderer: $symbol" >&2
    exit 1
  fi
done

if ! rg -F 'dsa_key_pool_compression_bf16_source' \
  "$root/kernels/glm_hybrid_dsa_pooling_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared GLM hybrid CUDA source lost its DSA pooling renderer" >&2
  exit 1
fi

if ! rg -F 'token_hash_lookup_i32_source' \
  "$root/kernels/decoder_token_hash_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared advanced decoder CUDA source lost its token-hash renderer" >&2
  exit 1
fi

if ! rg -F 'causal_complete_slots_i32_source' \
  "$root/kernels/decoder_compressed_index_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared advanced decoder CUDA source lost its deterministic compressed-index renderer" >&2
  exit 1
fi

if ! rg -F 'deterministic_window_compressed_join_i32_source' \
  "$root/kernels/decoder_window_compressed_join_cuda_source" \
  --glob '*.mbt' >/dev/null; then
  echo "shared advanced decoder CUDA source lost its window/compressed-index join renderer" >&2
  exit 1
fi

if ! rg -F 'non_overlapping_shared_kv_prepare_bf16_source' \
  "$root/kernels/decoder_compressed_kv_prepare_cuda_source" \
  --glob '*.mbt' >/dev/null; then
  echo "shared advanced decoder CUDA source lost its compressed-KV preparation renderer" >&2
  exit 1
fi

if ! rg -F 'scaled_yarn_rope_bf16_source' \
  "$root/kernels/decoder_scaled_rope_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared advanced decoder CUDA source lost its scaled-YaRN RoPE renderer" >&2
  exit 1
fi

if ! rg -F 'inverse_scaled_yarn_attention_output_bf16_source' \
  "$root/kernels/decoder_scaled_rope_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared advanced decoder CUDA source lost its inverse scaled-YaRN renderer" >&2
  exit 1
fi

if ! rg -F 'shared_key_value_selected_attention_bf16_source' \
  "$root/kernels/decoder_selected_sparse_attention_cuda_source" \
  --glob '*.mbt' >/dev/null; then
  echo "shared advanced decoder CUDA source lost its selected sparse-attention renderer" >&2
  exit 1
fi

if ! rg -F 'interleaved_pair_rope_with_query_prefix_copy_bf16_source' \
  "$root/kernels/decoder_interleaved_rope_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared decoder CUDA source lost its interleaved RoPE renderer" >&2
  exit 1
fi

if ! rg -F 'block_control_v1_bf16_f32_source' \
  "$root/kernels/decoder_hyper_connection_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared advanced decoder CUDA source lost its mHC block-control renderer" >&2
  exit 1
fi

if ! rg -F 'pre_reduce_v1_bf16_source' \
  "$root/kernels/decoder_hyper_connection_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared advanced decoder CUDA source lost its mHC pre-reduce renderer" >&2
  exit 1
fi

if ! rg -F 'post_combine_v1_bf16_source' \
  "$root/kernels/decoder_hyper_connection_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared advanced decoder CUDA source lost its mHC post-combine renderer" >&2
  exit 1
fi

if ! rg -F 'head_reduce_v1_bf16_f32_source' \
  "$root/kernels/decoder_hyper_connection_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared advanced decoder CUDA source lost its mHC head-reduce renderer" >&2
  exit 1
fi

if ! rg -F 'depthwise_causal_silu_bf16_source' \
  "$root/kernels/causal_short_convolution_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared decoder CUDA source lost its causal short-convolution renderer" >&2
  exit 1
fi

if ! rg -F 'causal_normalized_recurrent_delta_bf16_f32_source' \
  "$root/kernels/recurrent_delta_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared decoder CUDA source lost its recurrent-delta renderer" >&2
  exit 1
fi

if ! rg -F 'qkv_forget_input_gate_bf16_source' \
  "$root/kernels/recurrent_attention_projection_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared decoder CUDA source lost its KDA input projection renderer" >&2
  exit 1
fi

if ! rg -F 'bias_free_bf16_low_rank_projection_source' \
  "$root/kernels/two_stage_projection_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared decoder CUDA source lost its two-stage output-gate projection renderer" >&2
  exit 1
fi

if ! rg -F 'sigmoid_gated_rms_norm_bf16_source' \
  "$root/kernels/gated_rms_norm_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared decoder CUDA source lost its sigmoid-gated RMSNorm renderer" >&2
  exit 1
fi

if ! rg -F 'safe_lower_bound_sigmoid_decay_f32_source' \
  "$root/kernels/decay_control_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared decoder CUDA source lost its safe decay-control renderer" >&2
  exit 1
fi

if ! rg -F 'f32_pointwise_conv1d_source' \
  "$root/kernels/channel_major_projection_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared diffusion CUDA source lost its channel-major pointwise projection renderer" >&2
  exit 1
fi

if ! rg -F 'f32_weight_normalized_conv1d_source' \
  "$root/kernels/channel_major_projection_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared diffusion CUDA source lost its weight-normalized Conv1d renderer" >&2
  exit 1
fi

if ! rg -F 'f32_weight_normalized_conv_transpose1d_source' \
  "$root/kernels/channel_major_projection_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared diffusion CUDA source lost its weight-normalized ConvTranspose1d renderer" >&2
  exit 1
fi

if ! rg -F 'f32_input_bf16_weight_projection_source' \
  "$root/kernels/router_projection_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared decoder CUDA source lost its F32-input/BF16-weight projection renderer" >&2
  exit 1
fi

if ! rg -F 'sigmoid_and_correction_f32_source' \
  "$root/kernels/sigmoid_correction_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared decoder CUDA source lost its sigmoid-and-correction renderer" >&2
  exit 1
fi

if ! rg -F 'grouped_choice_and_selected_weight_f32_source' \
  "$root/kernels/grouped_routing_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared decoder CUDA source lost its grouped-routing renderer" >&2
  exit 1
fi

if ! rg -F 'f32_alias_free_snake_beta_source' \
  "$root/kernels/alias_free_activation_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared diffusion CUDA source lost its alias-free activation renderer" >&2
  exit 1
fi

for symbol in \
  'stable_biased_top_k_i32_source' \
  'selected_expert_weight_finalize_f32_source'
do
  if ! rg -F "$symbol" "$root/kernels/decoder_routing_cuda_source" \
    --glob '*.mbt' >/dev/null; then
    echo "shared advanced decoder CUDA source lost a routing renderer: $symbol" >&2
    exit 1
  fi
done

for symbol in \
  'projected_index_score_stable_topk_bf16_source' \
  'reuse_selected_indices_i32_source'
do
  if ! rg -F "$symbol" "$root/kernels/glm_hybrid_dsa_index_cuda_source" \
    --glob '*.mbt' >/dev/null; then
    echo "shared GLM hybrid CUDA source lost a DSA index renderer: $symbol" >&2
    exit 1
  fi
done

if ! rg -F 'selected_index_sparse_attention_bf16_source' \
  "$root/kernels/glm_hybrid_sparse_attention_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared GLM hybrid CUDA source lost its sparse-attention renderer" >&2
  exit 1
fi

if ! rg -F 'prefix_rms_norm_bf16_with_optional_suffix_copy_source' \
  "$root/kernels/prefix_rms_norm_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared decoder CUDA source lost its prefix-RMSNorm renderer" >&2
  exit 1
fi

if ! rg -F 'packed_swiglu_dense_bf16_rows_source' \
  "$root/kernels/gated_mlp_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared diffusion CUDA source lost its packed SwiGLU renderer" >&2
  exit 1
fi

if ! rg -F 'separate_gate_up_swiglu_dense_bf16_source' \
  "$root/kernels/gated_mlp_cuda_source" --glob '*.mbt' >/dev/null; then
  echo "shared decoder CUDA source lost its separate-weight SwiGLU renderer" >&2
  exit 1
fi

if ! rg -F 'staged_joint_attention_bf16_source' \
  "$root/kernels/joint_attention_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared diffusion CUDA source lost its staged attention renderer" >&2
  exit 1
fi

if ! rg -F 'cosine_sine_timestep_rows_source' \
  "$root/kernels/timestep_embedding_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared joint diffusion CUDA source lost its timestep renderer" >&2
  exit 1
fi

if ! rg -F 'dense_silu_dense_f32_rows_source' \
  "$root/kernels/timestep_embedding_cuda_source" --glob '*.mbt' \
  >/dev/null; then
  echo "shared joint diffusion CUDA source lost its timestep-MLP renderer" >&2
  exit 1
fi

for symbol in \
  'bf16_shift_scale_f32_fixed_rows_source' \
  'final_adaln_parameter_row_source' \
  'final_adaln_parameter_rows_source' \
  'packed_modality_adaln_gather_source' \
  'f32_channel_major_destandardization_source'
do
  if ! rg -F "$symbol" "$root/kernels/affine_modulation_cuda_source" \
    --glob '*.mbt' >/dev/null; then
    echo "shared joint diffusion CUDA source lost an affine renderer: $symbol" >&2
    exit 1
  fi
done


for package in $decoder_foundation_aot_candidate_packages
do
  if ! rg -F 'lower_language_model_head_cuda_aot_candidate' "$package" \
    --glob '*.mbt' >/dev/null; then
    echo "decoder AOT candidate lost its language-model-head lowerer: $package" >&2
    exit 1
  fi
done

if ! rg -F 'lower_expert_score_transform_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its expert-score lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_key_value_projection_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its key/value projection lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_token_hash_routing_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its token-hash lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_scaled_rotary_position_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its scaled-YaRN RoPE lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_deterministic_compressed_index_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its deterministic compressed-index lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_deterministic_window_compressed_join_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its window/compressed-index join lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_non_overlapping_compressed_kv_prepare_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its compressed-KV preparation lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_selected_sparse_attention_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its selected sparse-attention lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_attention_output_inverse_rotary_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its output inverse-RoPE lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_hyper_connection_block_control_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its mHC block-control lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_hyper_connection_pre_reduce_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its mHC pre-reduce lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_hyper_connection_post_combine_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its mHC post-combine lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_hyper_connection_head_reduce_cuda_aot_candidate' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its mHC head-reduce lowerer" >&2
  exit 1
fi

for symbol in \
  'lower_query_a_projection_cuda_aot_candidate' \
  'lower_no_auxiliary_loss_top_k_cuda_aot_candidate' \
  'lower_selected_expert_weight_finalize_cuda_aot_candidate' \
  'lower_attention_output_a_projection_cuda_aot_candidate' \
  'lower_attention_output_b_projection_cuda_aot_candidate'
do
  if ! rg -F "$symbol" "$root/kernels/advanced_decoder_cuda_aot" \
    --glob '*.mbt' >/dev/null; then
    echo "advanced decoder AOT candidate lost a routing lowerer: $symbol" >&2
    exit 1
  fi
done

if ! rg -F 'paired_rms_norm_bf16_source' \
  "$root/kernels/advanced_decoder_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder AOT candidate lost its paired Q/K RMSNorm lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_attention_output_projection_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its DSA output lowerer" >&2
  exit 1
fi

for symbol in \
  'lower_hyper_connection_function_cuda_aot_candidate' \
  'lower_hyper_connection_sinkhorn_cuda_aot_candidate'
do
  if ! rg -F "$symbol" "$root/kernels/glm_hybrid_cuda_aot" \
    --glob '*.mbt' >/dev/null; then
    echo "GLM hybrid AOT candidate lost a staged hyper-control lowerer: $symbol" >&2
    exit 1
  fi
done


if ! rg -F 'lower_dsa_key_pooling_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its DSA key-pooling lowerer" >&2
  exit 1
fi

for symbol in \
  'lower_dsa_index_selection_cuda_aot_candidate' \
  'lower_dsa_index_reuse_cuda_aot_candidate'
do
  if ! rg -F "$symbol" "$root/kernels/glm_hybrid_cuda_aot" \
    --glob '*.mbt' >/dev/null; then
    echo "GLM hybrid AOT candidate lost a DSA index lowerer: $symbol" >&2
    exit 1
  fi
done

if ! rg -F 'lower_dsa_sparse_attention_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its sparse-attention lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_dsa_key_value_split_norm_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its KV split/RMSNorm lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_dsa_kv_attention_assembly_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its exact K/V assembly lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_dsa_interleaved_rotary_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its interleaved RoPE lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_dense_swiglu_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its dense SwiGLU lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_moe_router_projection_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its MoE router projection lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_moe_sigmoid_correction_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its sigmoid-and-correction lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_moe_grouped_routing_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its grouped-routing lowerer" >&2
  exit 1
fi

for symbol in \
  'lower_moe_routed_expert_feed_forward_cuda_aot_candidate' \
  'lower_moe_shared_expert_feed_forward_cuda_aot_candidate' \
  'lower_moe_routed_shared_combine_cuda_aot_candidate' \
  'lower_moe_decoder_residual_add_cuda_aot_candidate' \
  'lower_hyper_connection_mix_cuda_aot_candidate'
do
  if ! rg -F "$symbol" "$root/kernels/glm_hybrid_cuda_aot" \
    --glob '*.mbt' >/dev/null; then
    echo "GLM hybrid AOT candidate lost an exact MoE execution lowerer: $symbol" >&2
    exit 1
  fi
done

if ! rg -F 'lower_kda_short_convolution_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its KDA short-convolution lowerer" >&2
  exit 1
fi

if ! rg -F 'lower_kda_recurrent_cuda_aot_candidate' \
  "$root/kernels/glm_hybrid_cuda_aot" --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid AOT candidate lost its KDA recurrent lowerer" >&2
  exit 1
fi

for symbol in \
  'lower_kda_projection_cuda_aot_candidate' \
  'lower_kda_output_gate_cuda_aot_candidate' \
  'lower_kda_gated_rms_norm_cuda_aot_candidate' \
  'lower_kda_output_projection_cuda_aot_candidate' \
  'lower_kda_decay_control_cuda_aot_candidate'
do
  if ! rg -F "$symbol" "$root/kernels/glm_hybrid_cuda_aot" \
    --glob '*.mbt' >/dev/null; then
    echo "GLM hybrid AOT candidate lost an exact KDA lowerer: $symbol" >&2
    exit 1
  fi
done

for symbol in \
  'lower_dsa_query_a_projection_cuda_aot_candidate' \
  'lower_dsa_query_b_projection_cuda_aot_candidate' \
  'lower_dsa_key_value_a_projection_cuda_aot_candidate' \
  'lower_dsa_key_value_b_projection_cuda_aot_candidate'
do
  if ! rg -F "$symbol" "$root/kernels/glm_hybrid_cuda_aot" \
    --glob '*.mbt' >/dev/null; then
    echo "GLM hybrid AOT candidate lost a DSA input lowerer: $symbol" >&2
    exit 1
  fi
done

for symbol in \
  'assess_ordered_conditioning_aot' \
  'require_ordered_conditioning_aot_candidate'
do
  if ! rg -F "$symbol" "$root/model/joint_diffusion_conditioning_reference" \
    --glob '*.mbt' >/dev/null; then
    echo "joint diffusion conditioning reference lost its fail-closed assessment: $symbol" >&2
    exit 1
  fi
done

for symbol in \
  'MiniMaxH3VideoVaeSchema::official_v1' \
  'MiniMaxH3VideoVaeSchema::tensor_count' \
  'MiniMaxH3VideoVaeSchema::tensor_bytes'
do
  if ! rg -F "$symbol" "$root/model/minimax_h3_weights" \
    --glob '*.mbt' >/dev/null; then
    echo "MiniMax H3 weights lost exact VideoVae schema evidence: $symbol" >&2
    exit 1
  fi
done

for symbol in \
  'MiniMaxH3AudioVaeSchema::official_v1' \
  'MiniMaxH3AudioVaeSchema::tensor_count' \
  'MiniMaxH3AudioVaeSchema::tensor_bytes'
do
  if ! rg -F "$symbol" "$root/model/minimax_h3_weights" \
    --glob '*.mbt' >/dev/null; then
    echo "MiniMax H3 weights lost exact AudioVae schema evidence: $symbol" >&2
    exit 1
  fi
done

for package in $aot_candidate_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|device("|/)|engine|internal/cuda|model/(deepseek_v4|glm53|minimax_h3))' \
    "$package/moon.pkg" >/dev/null; then
    echo "advanced AOT candidate crosses a family/runtime/native boundary: $package" >&2
    exit 1
  fi
  for symbol in 'manifest_bindable' 'source_digest' 'recipe_digest'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "advanced AOT candidate lost deterministic inert lowering: $package ($symbol)" >&2
      exit 1
    fi
  done
done

for package in $decoder_foundation_aot_candidate_packages
do
  if ! rg -F 'lower_foundation_cuda_aot_candidate' "$package" --glob '*.mbt' \
    >/dev/null; then
    echo "decoder foundation AOT candidate lost its exact lowerer: $package" >&2
    exit 1
  fi
done

for package in $diffusion_aot_candidate_packages
do
  for symbol in \
    'lower_latent_projection_cuda_aot_candidate' \
    'lower_output_projection_cuda_aot_candidate' \
    'lower_final_rms_norm_cuda_aot_candidate' \
    'lower_final_adaln_parameter_row_cuda_aot_candidate' \
    'lower_packed_modality_adaln_cuda_aot_candidate' \
    'lower_timestep_frequency_cuda_aot_candidate' \
    'lower_timestep_mlp_cuda_aot_candidate' \
    'lower_distinct_timestep_adaln_cuda_aot_candidate' \
    'compose_timestep_conditioning_candidates' \
    'lower_transformer_attention_cuda_aot_candidate' \
    'lower_transformer_mlp_cuda_aot_candidate' \
    'lower_final_adaln_f32_cuda_aot_candidate' \
    'lower_shifted_flow_cuda_aot_candidate' \
    'lower_video_vae_latent_destandardization_cuda_aot_candidate' \
    'lower_audio_vae_latent_destandardization_cuda_aot_candidate' \
    'lower_audio_vae_decoder_input_projection_cuda_aot_candidate' \
    'lower_audio_vae_decoder_conv_pre_cuda_aot_candidate' \
    'lower_audio_vae_decoder_first_upsample_cuda_aot_candidate' \
    'lower_audio_vae_decoder_first_amp_activation_cuda_aot_candidate' \
    'lower_audio_vae_decoder_first_amp_stage_cuda_aot_candidate' \
    'lower_audio_vae_decoder_upsample_transition_cuda_aot_candidate' \
    'lower_audio_vae_decoder_amp_activation_cuda_aot_candidate' \
    'lower_audio_vae_decoder_amp_stage_cuda_aot_candidate'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "joint diffusion AOT candidate lost an exact lowerer: $package ($symbol)" >&2
      exit 1
    fi
  done
done

for symbol in \
  'QueryLowRankAProjection' \
  'QueryLowRankBProjection' \
  'DeterministicCompressedIndex' \
  'DeterministicWindowCompressedIndexJoinV1' \
  'NonOverlappingCompressedKeyValuePrepareV1' \
  'SelectedSparseAttentionV1' \
  'AttentionOutputInverseRotaryV1' \
  'AttentionOutputAProjectionV1' \
  'AttentionOutputBProjectionV1'
do
  if ! rg -F "$symbol" "$root/model/advanced_decoder_kernel_requirements" \
    --glob '*.mbt' >/dev/null; then
    echo "advanced decoder requirements lost an exact query projection phase: $symbol" >&2
    exit 1
  fi
done

for symbol in \
  'MoeRouterProjection' \
  'MoeSigmoidAndCorrectionBias' \
  'MoeGroupedNoAuxiliaryLossTopK' \
  'RoutedExpertFeedForward' \
  'SharedExpertFeedForward' \
  'RoutedSharedExpertCombine' \
  'MoeDecoderResidualAdd' \
  'HyperConnectionMix' \
  'HyperConnectionFunction' \
  'HyperConnectionSinkhorn' \
  'DsaKeyValueAttentionAssembly' \
  'KdaDecayControl' \
  'KdaOutputGateProjection' \
  'KdaGatedRmsNorm' \
  'KdaOutputProjection'
do
  if ! rg -F "$symbol" "$root/model/glm_hybrid_kernel_requirements" \
    --glob '*.mbt' >/dev/null; then
    echo "GLM hybrid requirements lost an exact KDA phase: $symbol" >&2
    exit 1
  fi
done

for symbol in \
  'VideoVaeLatentDestandardization' \
  'VideoVaeDecoderBody' \
  'AudioVaeLatentDestandardization' \
  'AudioVaeDecoderInputProjection' \
  'AudioVaeDecoderConvPre' \
  'AudioVaeDecoderFirstUpsample' \
  'AudioVaeDecoderFirstAmpActivation' \
  'AudioVaeDecoderFirstAmpStage' \
  'AudioVaeDecoderUpsampleTransition' \
  'AudioVaeDecoderAmpActivation' \
  'AudioVaeDecoderAmpStage' \
  'AudioVaeDecoderBody'
do
  if ! rg -F "$symbol" "$root/model/joint_diffusion_kernel_requirements" \
    --glob '*.mbt' >/dev/null; then
    echo "joint diffusion requirements lost an exact VAE decode phase: $symbol" >&2
    exit 1
  fi
done

if rg -F 'QueryLowRankProjection(' \
  "$root/model/advanced_decoder_kernel_requirements" --glob '*.mbt' >/dev/null; then
  echo "advanced decoder requirements regressed to a fused query projection" >&2
  exit 1
fi

if rg -F 'DeterministicCompressedAttention' \
  "$root/model/advanced_decoder_kernel_requirements" \
  "$root/model/deepseek_v4_numeric_plan" --glob '*.mbt' --glob '*.mbti' \
  >/dev/null; then
  echo "advanced decoder requirements retained the obsolete fused compressed-attention placeholder" >&2
  exit 1
fi

for symbol in \
  'HyperConnectionBlockControlV1' \
  'HyperConnectionPreReduceV1' \
  'HyperConnectionPostCombineV1' \
  'HyperConnectionHeadReduceV1'
do
  if ! rg -F "$symbol" "$root/model/advanced_decoder_kernel_requirements" \
    --glob '*.mbt' >/dev/null; then
    echo "advanced decoder requirements lost an exact mHC phase: $symbol" >&2
    exit 1
  fi
done

for symbol in \
  'hyper_connection_block_controls_v1' \
  'hyper_connection_pre_reduce_v1' \
  'hyper_connection_post_combine_v1' \
  'hyper_connection_head_reduce_v1'
do
  if ! rg -F "$symbol" "$root/model/advanced_decoder_reference" \
    --glob '*.mbt' >/dev/null; then
    echo "advanced decoder reference lost an exact mHC phase: $symbol" >&2
    exit 1
  fi
done

for package in $operand_region_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|device("|/)|engine|internal/cuda|model/(deepseek_v4|glm53|minimax_h3))' \
    "$package/moon.pkg" >/dev/null; then
    echo "advanced operand-region planning crossed a family/runtime/native boundary: $package" >&2
    exit 1
  fi
  for symbol in \
    'plan_advanced_decoder_operand_regions' \
    'AdvancedDecoderOperandRegionInput::new' \
    'require_launch_authority'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "advanced operand-region planning lost inert validation: $package ($symbol)" >&2
      exit 1
    fi
  done
done

for package in $compile_evidence_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|device("|/)|engine|internal/cuda|model/(deepseek_v4|glm53|minimax_h3))' \
    "$package/moon.pkg" >/dev/null; then
    echo "offline compile evidence crossed a family/runtime/native boundary: $package" >&2
    exit 1
  fi
  for symbol in \
    'admit_deterministic_cuda_output' \
    'require_producer_authority' \
    'require_compiler_execution_proof' \
    'require_execution_authority'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "offline compile evidence lost inert fail-closed behavior: $package ($symbol)" >&2
      exit 1
    fi
  done
done

for symbol in \
  'plan_token_hash_sidecar_upload' \
  'validate_token_hash_sidecar_source' \
  'materialize_token_hash_sidecars'
do
  if ! rg -F "$symbol" "$root/model/deepseek_v4_device_upload" \
    --glob '*.mbt' >/dev/null; then
    echo "DeepSeek device adapter lost its token-hash sidecar seam: $symbol" >&2
    exit 1
  fi
done

for package in $host_borrow_packages
do
  if rg -n 'vectie/lunaflux/' "$package/moon.pkg" >/dev/null; then
    echo "advanced host-borrow seam acquired a LunaFlux dependency: $package" >&2
    exit 1
  fi
  for symbol in 'SegmentedHostArenaConsumer::accept' 'SegmentedHostArenaConsumer::revoke'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "advanced host-borrow seam lost one-shot behavior: $package ($symbol)" >&2
      exit 1
    fi
  done
done

for package in $family_device_adapter_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|engine|kernels|internal/cuda)' \
    "$package/moon.pkg" >/dev/null; then
    echo "advanced family device adapter crosses a runtime/kernel/native boundary: $package" >&2
    exit 1
  fi
  for symbol in 'plan_device_upload' 'validate_host_source' 'materialize_device_weights'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "advanced family device adapter lost its narrow upload seam: $package ($symbol)" >&2
      exit 1
    fi
  done
done

for package in $device_bridge_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|engine|kernels|internal/cuda|model/(deepseek_v4|glm53|minimax_h3))' \
    "$package/moon.pkg" >/dev/null; then
    echo "advanced device bridge crosses a family/runtime/native boundary: $package" >&2
    exit 1
  fi
  for symbol in 'plan_layout' 'materialize_from_borrowed_source' 'retry_close'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "advanced device bridge lost lifecycle behavior: $package ($symbol)" >&2
      exit 1
    fi
  done
done

for package in $artifact_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|device("|/)|engine|internal/cuda|model/(deepseek_v4|glm53|minimax_h3))' \
    "$package/moon.pkg" >/dev/null; then
    echo "advanced artifact package crosses a family/runtime/native boundary: $package" >&2
    exit 1
  fi
  for symbol in 'pub fn admit(' 'require_execution_authority'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "advanced artifact package lost inert admission behavior: $package ($symbol)" >&2
      exit 1
    fi
  done
done

for package in $candidate_coverage_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|device("|/)|engine|internal/cuda|model/(deepseek_v4|glm53|minimax_h3))' \
    "$package/moon.pkg" >/dev/null; then
    echo "AOT candidate coverage crossed a family/runtime/native boundary: $package" >&2
    exit 1
  fi
  for symbol in 'pub fn audit(' 'require_complete' 'require_all_claimed_bindable'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "AOT candidate coverage lost fail-closed behavior: $package ($symbol)" >&2
      exit 1
    fi
  done
done

for package in $candidate_evidence_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|device("|/)|engine|internal/cuda|model/(deepseek_v4|glm53|minimax_h3))' \
    "$package/moon.pkg" >/dev/null; then
    echo "AOT candidate evidence adapter crossed a family/runtime/native boundary: $package" >&2
    exit 1
  fi
  if ! rg -F 'pub fn from_' "$package" --glob '*.mbt' >/dev/null; then
    echo "AOT candidate evidence adapter lost its explicit projections: $package" >&2
    exit 1
  fi
done

for package in $composition_packages
do
  if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|engine|internal/cuda)' \
    "$package/moon.pkg" >/dev/null; then
    echo "advanced composition package crossed a runtime/native boundary: $package" >&2
    exit 1
  fi
done

for package in $deepseek_token_hash_composition_packages
do
  for symbol in \
    'plan_token_hash_table_operand' \
    'bind_token_hash_artifact' \
    'bind_token_hash_compile_evidence' \
    'bind_complete_token_hash_launch' \
    'require_compilation_provenance' \
    'require_loader_authority' \
    'require_launch_authority' \
    'require_physical_qualification'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "DeepSeek operand composition lost inert fail-closed behavior: $symbol" >&2
      exit 1
    fi
  done
done

for package in $deepseek_output_a_composition_packages
do
  for symbol in \
    'plan_output_a_parameter' \
    'require_conversion_authority' \
    'require_materialization_authority' \
    'require_upload_authority' \
    'require_launch_authority'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "DeepSeek output-A composition lost inert fail-closed behavior: $symbol" >&2
      exit 1
    fi
  done
done

for package in $deepseek_query_a_composition_packages
do
  for symbol in \
    'plan_query_a_parameter' \
    'require_interpretation_authority' \
    'require_materialization_authority' \
    'require_upload_authority' \
    'require_launch_authority'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "DeepSeek query-A composition lost inert fail-closed behavior: $symbol" >&2
      exit 1
    fi
  done
done

for package in $deepseek_key_value_composition_packages
do
  for symbol in \
    'plan_key_value_parameter' \
    'require_interpretation_authority' \
    'require_materialization_authority' \
    'require_upload_authority' \
    'require_launch_authority'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "DeepSeek key/value composition lost inert fail-closed behavior: $symbol" >&2
      exit 1
    fi
  done
done

for package in $deepseek_output_b_composition_packages
do
  for symbol in \
    'plan_output_b_parameter' \
    'require_interpretation_authority' \
    'require_materialization_authority' \
    'require_upload_authority' \
    'require_collective_authority' \
    'require_launch_authority'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "DeepSeek output-B composition lost inert fail-closed behavior: $symbol" >&2
      exit 1
    fi
  done
done

for package in $glm_moe_composition_packages
do
  for symbol in \
    'join_moe_expert_bank_candidates' \
    'materialization_authority' \
    'upload_authority' \
    'launch_authority'
  do
    if ! rg -F "$symbol" "$package" --glob '*.mbt' >/dev/null; then
      echo "GLM MoE composition lost inert fail-closed behavior: $symbol" >&2
      exit 1
    fi
  done
done

for projection in \
  'DeepSeekV4SemanticPlan::workload_execution_plan' \
  'build_full_workload_execution_plan' \
  'build_flash_workload_execution_plan' \
  'build_workload_execution_plan'
do
  if ! rg -F "$projection" \
    "$root/model/deepseek_v4" "$root/model/glm53" "$root/model/minimax_h3" \
    --glob '*.mbt' >/dev/null; then
    echo "advanced family lost workload-plan projection: $projection" >&2
    exit 1
  fi
done

for config_package in \
  "$root/model/deepseek_v4_config" \
  "$root/model/glm53_config" \
  "$root/model/minimax_h3_config"
do
  if ! rg -n 'pub fn (parse|parse_).*\(' "$config_package" --glob '*.mbt' \
    >/dev/null; then
    echo "advanced family lost bounded configuration admission: $config_package" >&2
    exit 1
  fi
done

for manifest_count in 69187 69189 145116 145118 72317 59585 38770 638 1058 703 1087
do
  if ! rg -F "$manifest_count" \
    "$root/model/deepseek_v4_weights" \
    "$root/model/glm53_weights" \
    "$root/model/minimax_h3_weights" \
    --glob '*test.mbt' >/dev/null; then
    echo "advanced family exact manifest-count evidence is missing: $manifest_count" >&2
    exit 1
  fi
done

for reference_symbol in \
  'route_top_k' \
  'route_token_hash' \
  'run_tiny_advanced_decoder_block' \
  'sinkhorn_normalize' \
  'learned_index_top_k' \
  'run_flash_0731' \
  'pool_dsa_index_keys' \
  'hyper_connection_function_and_collapse' \
  'hyper_connection_positive_sinkhorn' \
  'kda_recurrent_delta_update' \
  'vision_patch_projection' \
  'build_shifted_rectified_flow_schedule' \
  'CounterBasedJointLatentRng::normal' \
  'JointDiffusionStageMachine::advance' \
  'run_tiny_joint_transformer_step' \
  'assess_joint_vae_decode'
do
  if ! rg -F "$reference_symbol" $reference_packages --glob '*.mbt' \
    >/dev/null; then
    echo "advanced family lost deterministic reference evidence: $reference_symbol" >&2
    exit 1
  fi
done

for reference_symbol in \
  'decode_e4m3_finite' \
  'decode_ue8m0' \
  'activation_scale_ue8m0_code' \
  'block128_dot_bf16_boundary'
do
  if ! rg -F "$reference_symbol" "$root/model/block_fp8_ue8m0_reference" \
    --glob '*.mbt' >/dev/null; then
    echo "block-FP8/UE8M0 reference lost exact scalar semantics: $reference_symbol" >&2
    exit 1
  fi
done

for reference_symbol in \
  'bind_audio_decoder_upsample_transition' \
  'execute_audio_decoder_upsample_transition' \
  'bind_audio_decoder_second_upsample' \
  'bind_audio_decoder_amp_activation' \
  'execute_audio_decoder_amp_activation' \
  'bind_audio_decoder_second_amp_activation' \
  'bind_audio_decoder_third_upsample' \
  'bind_audio_decoder_third_amp_activation' \
  'bind_second_amp_parameter_layout'
do
  if ! rg -F "$reference_symbol" \
    "$root/model/joint_diffusion_vae_reference" \
    "$root/model/minimax_h3_vae_reference" --glob '*.mbt' >/dev/null; then
    echo "MiniMax AudioVAE reference lost its exact upsample transition: $reference_symbol" >&2
    exit 1
  fi
done


if ! rg -F 'dsa_kv_attention_assembly' "$root/model/glm_hybrid_reference" \
  --glob '*.mbt' >/dev/null; then
  echo "GLM hybrid reference lost exact K/V assembly semantics" >&2
  exit 1
fi

for execution_contract_symbol in \
  'derive_requirements' \
  'require_current_generic_capabilities' \
  'require_current_generic_decoder_capabilities' \
  'pub fn project(' \
  'build_full_kernel_requirements' \
  'build_flash_kernel_requirements' \
  'build_full_numeric_plan' \
  'build_flash_numeric_plan' \
  'materialize_safetensors_shards' \
  'Glm53HostWeights::close' \
  'DeepSeekV4HostWeights::release' \
  'DeepSeekV4HostWeights::materialize_token_hash_tables' \
  'DeepSeekV4HostWeights::require_executable_weights' \
  'MiniMaxH3HostComponentArena::release' \
  'validate_weight_manifest' \
  'query_a_quantized_layout' \
  'key_value_quantized_layout' \
  'decode_finite_fp8_e4m3' \
  'require_physical_materialization' \
  'pub fn normalize(' \
  'inspect_shards' \
  'copy_ranges' \
  'copy_segmented_ranges' \
  'narrow_unique_i64_le_rows' \
  'JointDiffusionResultMetadata::succeeded' \
  'JointDiffusionResultMetadata::cancelled' \
  'JointDiffusionProjectionContract::mixed_precision_float32_heads' \
  'JointDiffusionProjectionContract::output_storage' \
  'JointPackedTimestepSelection::new' \
  'JointPackedModalitySelection::new' \
  'JointDistinctTimestepValues::new'
do
  if ! rg -F "$execution_contract_symbol" \
    "$root/model/advanced_decoder_kernel_requirements" \
    "$root/model/checked_integer_narrow" \
    "$root/model/deepseek_v4_kernel_requirements" \
    "$root/model/deepseek_v4_host_materialization" \
    "$root/model/deepseek_v4_numeric_plan" \
    "$root/model/deepseek_v4_quantized_payload" \
    "$root/model/deepseek_v4_weights" \
    "$root/model/glm_hybrid_kernel_requirements" \
    "$root/model/glm53_kernel_requirements" \
    "$root/model/glm53_host_materialize" \
    "$root/model/glm53_numeric_plan" \
    "$root/model/joint_diffusion_io_plan" \
    "$root/model/joint_diffusion_plan" \
    "$root/model/joint_diffusion_kernel_requirements" \
    "$root/model/minimax_h3_host_materialization" \
    "$root/model/minimax_h3_kernel_requirements" \
    "$root/model/streaming_safetensors" \
    --glob '*.mbt' >/dev/null; then
    echo "advanced family lost an execution-boundary contract: $execution_contract_symbol" >&2
    exit 1
  fi
done

for package in $plan_packages $parameter_layout_packages
do
  if rg -n '(DeepSeek|deepseek|MiniMax|minimax|GLM-?5|glm53|glm_?5)' \
    "$package" --glob '*.mbt' --glob '!**/*_test.mbt' >/dev/null; then
    echo "generic workload plan contains a model-family branch: $package" >&2
    exit 1
  fi
done

for runtime_owner in \
  "$root/scheduler" \
  "$root/kv" \
  "$root/prefix" \
  "$root/api" \
  "$root/device" \
  "$root/kernels"
do
  if [ -d "$runtime_owner" ] && \
    rg -n '(DeepSeekV4|deepseek_v4|MiniMaxH3|minimax_h3|Glm53|glm53)' \
      "$runtime_owner" --glob '*.mbt' --glob 'moon.pkg' >/dev/null; then
    echo "core runtime owner contains an advanced model-family branch: $runtime_owner" >&2
    exit 1
  fi
done

for package in $plan_packages $family_catalog_packages $family_packages $reference_packages $artifact_packages $compile_evidence_packages $operand_region_packages $parameter_layout_packages $candidate_coverage_packages $candidate_evidence_packages $aot_candidate_packages $shared_cuda_source_packages $decoder_projection_source_packages $device_bridge_packages $host_borrow_packages $family_device_adapter_packages $composition_packages $integration_packages
do
  for file in "$package"/*.mbt "$package"/*.mbt.md
  do
    if [ ! -f "$file" ]; then
      continue
    fi
    lines=$(wc -l < "$file")
    if [ "$lines" -ge 500 ]; then
      echo "advanced model source exceeds the preferred size boundary: $file ($lines)" >&2
      exit 1
    fi
  done
done

moon fmt --check $validated_packages
moon check --target native --deny-warn --warn-list +73 $validated_packages
moon test --target native --deny-warn --warn-list +73 $validated_packages

echo "Advanced model-family modularity boundary: ok"
