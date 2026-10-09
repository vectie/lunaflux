///|
fn checked_positive_mul(left : Int64, right : Int64) -> Int64? {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    None
  } else {
    Some(left * right)
  }
}

///|
fn exact_rows(byte_count : Int64, row_bytes : Int64) -> Int64? {
  if row_bytes <= 0L || byte_count <= 0L || byte_count % row_bytes != 0L {
    None
  } else {
    Some(byte_count / row_bytes)
  }
}

///|
fn valid_function_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
fn valid_operand_metadata(
  operands : ReadOnlyArray[GlmHybridLaunchOperand],
) -> Bool {
  if operands.is_empty() || operands.length() > 32 {
    return false
  }
  operands.all(operand => {
    operand.byte_count > 0L &&
    operand.alignment > 0L &&
    operand.alignment <= 4096L &&
    (operand.alignment & (operand.alignment - 1L)) == 0L
  })
}

///|
fn operand_is(
  operand : GlmHybridLaunchOperand,
  role : GlmHybridLaunchOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> Bool {
  operand.role == role &&
  operand.byte_count == bytes &&
  operand.alignment == alignment
}

///|
fn exact_launch(
  launch : @launch_contract.AotLaunchDimensions,
  work : Int64,
  shared_memory_bytes : Int,
) -> Bool {
  if work <= 0L || work > 9223372036854775552L {
    return false
  }
  let grid = (work + 255L) / 256L
  grid <= 2147483647L &&
  launch.grid_x() == grid.to_int() &&
  launch.grid_y() == 1 &&
  launch.grid_z() == 1 &&
  launch.block_x() == 256 &&
  launch.block_y() == 1 &&
  launch.block_z() == 1 &&
  launch.shared_memory_bytes() == shared_memory_bytes
}

///|
fn exact_embedding_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  guard claim.operands is [counts, token_ids, weights, output] else {
    return false
  }
  guard exact_rows(token_ids.byte_count, 4L) is Some(rows) else { return false }
  guard checked_positive_mul(rows, geometry.hidden_size().to_int64())
    is Some(output_elements) else {
    return false
  }
  guard checked_positive_mul(
      geometry.vocabulary_size().to_int64(),
      geometry.hidden_size().to_int64(),
    )
    is Some(weight_elements) else {
    return false
  }
  guard checked_positive_mul(output_elements, 2L) is Some(output_bytes) else {
    return false
  }
  guard checked_positive_mul(weight_elements, 2L) is Some(weight_bytes) else {
    return false
  }
  claim.function_symbol == "lunaflux_glm_hybrid_embedding_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(token_ids, TokenIdsInput, rows * 4L, 4L) &&
  operand_is(weights, ModelWeights, weight_bytes, 16L) &&
  operand_is(output, HiddenOutput, output_bytes, 16L) &&
  exact_launch(claim.launch, output_elements, 0)
}

///|
fn exact_norm_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  claim : GlmHybridArtifactClaimInput,
  final_norm : Bool,
) -> Bool {
  guard claim.operands is [counts, input, weights, output] else { return false }
  let width = geometry.hidden_size().to_int64()
  guard checked_positive_mul(width, 2L) is Some(row_bytes) else { return false }
  guard exact_rows(input.byte_count, row_bytes) is Some(rows) else {
    return false
  }
  if rows > 2147483647L {
    return false
  }
  let expected_symbol = if final_norm {
    "lunaflux_glm_hybrid_final_rms_norm_bf16_v1"
  } else {
    "lunaflux_glm_hybrid_decoder_rms_norm_bf16_v1"
  }
  claim.function_symbol == expected_symbol &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, HiddenInput, input.byte_count, 16L) &&
  operand_is(weights, ModelWeights, row_bytes, 16L) &&
  operand_is(output, HiddenOutput, input.byte_count, 16L) &&
  claim.launch.grid_x() == rows.to_int() &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 256 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 1024
}

///|
fn exact_language_model_head_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  guard claim.operands is [counts, input, weights, output] else { return false }
  let input_width = geometry.hidden_size().to_int64()
  let output_width = geometry.vocabulary_size().to_int64()
  guard checked_positive_mul(input_width, 2L) is Some(input_row_bytes) else {
    return false
  }
  guard checked_positive_mul(output_width, 2L) is Some(output_row_bytes) else {
    return false
  }
  guard exact_rows(input.byte_count, input_row_bytes) is Some(rows) else {
    return false
  }
  guard checked_positive_mul(rows, output_row_bytes) is Some(output_bytes) else {
    return false
  }
  guard checked_positive_mul(input_width, output_width) is Some(weight_elements) else {
    return false
  }
  guard checked_positive_mul(weight_elements, 2L) is Some(weight_bytes) else {
    return false
  }
  guard checked_positive_mul(rows, output_width) is Some(work) else {
    return false
  }
  claim.function_symbol == "lunaflux_glm_hybrid_language_model_head_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, HiddenInput, input.byte_count, 16L) &&
  operand_is(weights, ModelWeights, weight_bytes, 16L) &&
  operand_is(output, LogitsOutput, output_bytes, 16L) &&
  exact_launch(claim.launch, work, 0)
}

///|
fn exact_attention_output_projection_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  output_width : Int,
  input_width : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if output_width != geometry.hidden_size() || input_width <= 0 {
    return false
  }
  guard claim.operands is [counts, input, weights, output] else { return false }
  guard checked_positive_mul(input_width.to_int64(), 2L)
    is Some(input_row_bytes) else {
    return false
  }
  guard checked_positive_mul(output_width.to_int64(), 2L)
    is Some(output_row_bytes) else {
    return false
  }
  guard exact_rows(input.byte_count, input_row_bytes) is Some(rows) else {
    return false
  }
  guard checked_positive_mul(rows, output_row_bytes) is Some(output_bytes) else {
    return false
  }
  guard checked_positive_mul(input_width.to_int64(), output_width.to_int64())
    is Some(weight_elements) else {
    return false
  }
  guard checked_positive_mul(weight_elements, 2L) is Some(weight_bytes) else {
    return false
  }
  guard checked_positive_mul(rows, output_width.to_int64()) is Some(work) else {
    return false
  }
  claim.function_symbol ==
  "lunaflux_glm_hybrid_attention_output_projection_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, HiddenInput, input.byte_count, 16L) &&
  operand_is(weights, ModelWeights, weight_bytes, 16L) &&
  operand_is(output, HiddenOutput, output_bytes, 16L) &&
  exact_launch(claim.launch, work, 0)
}

///|
fn dsa_index_head_dimension(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int? {
  let mut result = None
  for requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexCompute(_, head_dimension, _, _) {
      if result is Some(_) {
        return None
      }
      result = Some(head_dimension)
    }
  }
  result
}

///|
fn exact_dsa_pooling_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  pool_size : Int,
  compress : Bool,
  always_select_tail : Bool,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if pool_size <= 0 || !compress || !always_select_tail {
    return false
  }
  guard dsa_index_head_dimension(requirements) is Some(head_dimension) else {
    return false
  }
  guard claim.operands
    is [
      counts,
      keys,
      gate_scores,
      valid_tokens,
      pool_ape,
      pooled_keys,
      raw_indices,
      pool_valid,
      pool_count,
    ] else {
    return false
  }
  guard exact_rows(valid_tokens.byte_count, 4L) is Some(maximum_tokens) else {
    return false
  }
  if maximum_tokens < pool_size.to_int64() ||
    maximum_tokens > requirements.geometry().context_length().to_int64() {
    return false
  }
  let pool_capacity = maximum_tokens / pool_size.to_int64()
  guard checked_positive_mul(maximum_tokens, head_dimension.to_int64())
    is Some(input_elements) else {
    return false
  }
  guard checked_positive_mul(pool_capacity, head_dimension.to_int64())
    is Some(output_elements) else {
    return false
  }
  guard checked_positive_mul(pool_size.to_int64(), head_dimension.to_int64())
    is Some(ape_elements) else {
    return false
  }
  guard checked_positive_mul(pool_capacity, pool_size.to_int64())
    is Some(index_elements) else {
    return false
  }
  claim.function_symbol == "lunaflux_glm_hybrid_dsa_key_pool_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(keys, DsaIndexKeysInput, input_elements * 2L, 16L) &&
  operand_is(gate_scores, DsaIndexGateScoresInput, input_elements * 2L, 16L) &&
  operand_is(valid_tokens, DsaIndexValidityInput, maximum_tokens * 4L, 4L) &&
  operand_is(pool_ape, ModelWeights, ape_elements * 2L, 16L) &&
  operand_is(pooled_keys, DsaPooledKeysOutput, output_elements * 2L, 16L) &&
  operand_is(raw_indices, DsaPooledRawIndicesOutput, index_elements * 4L, 4L) &&
  operand_is(pool_valid, DsaPooledValidityOutput, pool_capacity * 4L, 4L) &&
  operand_is(pool_count, DsaPooledCountOutput, 4L, 4L) &&
  claim.launch.grid_x() == 1 &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 1 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}

///|
fn exact_supported_launch_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if !valid_function_symbol(claim.function_symbol) ||
    !valid_operand_metadata(claim.operands) {
    return false
  }
  let geometry = requirements.geometry()
  match claim.operation {
    TokenEmbedding => exact_embedding_abi(geometry, claim)
    DecoderRmsNorm(_, _) => exact_norm_abi(geometry, claim, false)
    FinalRmsNorm(_, _) => exact_norm_abi(geometry, claim, true)
    LanguageModelHead(_, _) => exact_language_model_head_abi(geometry, claim)
    AttentionOutputProjection(output_width, input_width) =>
      exact_attention_output_projection_abi(
        geometry, output_width, input_width, claim,
      )
    DsaIndexPooling(pool_size, compress, always_select_tail) =>
      exact_dsa_pooling_abi(
        requirements, pool_size, compress, always_select_tail, claim,
      )
    DsaIndexCompute(head_count, head_dimension, top_k, rope_interleave) =>
      exact_dsa_index_compute_abi(
        requirements, head_count, head_dimension, top_k, rope_interleave, claim,
      )
    DsaIndexReuse => exact_dsa_index_reuse_abi(requirements, claim)
    DsaSparseAttention(query_heads, top_k, qk_dimension, value_dimension) =>
      exact_dsa_sparse_attention_abi(
        requirements, query_heads, top_k, qk_dimension, value_dimension, claim,
      )
    DsaQueryAProjection(input_width, query_rank) =>
      exact_dsa_a_projection_abi(
        geometry,
        input_width,
        query_rank,
        true,
        HiddenInput,
        DsaQueryCompressedOutput,
        "lunaflux_glm_hybrid_dsa_query_a_projection_bf16_v1",
        claim,
      )
    DsaKeyValueAProjection(input_width, compressed_width, rope_width) =>
      exact_dsa_a_projection_abi(
        geometry,
        input_width,
        compressed_width + rope_width,
        true,
        HiddenInput,
        DsaKeyValueCompressedRopeOutput,
        "lunaflux_glm_hybrid_dsa_key_value_a_projection_bf16_v1",
        claim,
      )
    DsaQueryARmsNorm(width, epsilon) =>
      exact_dsa_query_a_norm_abi(geometry, width, epsilon, claim)
    DsaQueryBProjection(input_width, head_count, head_dimension) =>
      exact_dsa_a_projection_abi(
        geometry,
        input_width,
        head_count * head_dimension,
        false,
        DsaQueryNormalizedInput,
        DsaQueryHeadsOutput,
        "lunaflux_glm_hybrid_dsa_query_b_projection_bf16_v1",
        claim,
      )
    KeyValueLowRankProjection(
      compressed_width,
      head_count,
      key_nope_width,
      value_width
    ) =>
      exact_dsa_a_projection_abi(
        geometry,
        compressed_width,
        head_count * (key_nope_width + value_width),
        false,
        DsaKeyValueNormalizedInput,
        DsaKeyValueHeadsOutput,
        "lunaflux_glm_hybrid_dsa_key_value_b_projection_bf16_v1",
        claim,
      )
    QueryKeyRmsNorm(epsilon) =>
      exact_dsa_kv_split_norm_abi(requirements, epsilon, claim)
    GlmRotaryPosition(theta, interleaved) =>
      exact_glm_interleaved_rotary_abi(requirements, theta, interleaved, claim)
    DenseSwiGlu(hidden_size, intermediate_size) =>
      exact_dense_swiglu_abi(geometry, hidden_size, intermediate_size, claim)
    KdaProjection(head_count, head_dimension) =>
      exact_kda_projection_abi(geometry, head_count, head_dimension, claim)
    KdaOutputGateProjection(hidden_width, head_count, head_dimension) =>
      exact_kda_output_gate_projection_abi(
        geometry, hidden_width, head_count, head_dimension, claim,
      )
    KdaGatedRmsNorm(head_count, head_dimension, epsilon) =>
      exact_kda_gated_rms_norm_abi(
        geometry, head_count, head_dimension, epsilon, claim,
      )
    KdaOutputProjection(input_width, output_width) =>
      exact_kda_output_projection_abi(
        geometry, input_width, output_width, claim,
      )
    KdaDecayControl(head_count, head_dimension, lower_bound) =>
      exact_kda_decay_control_abi(
        geometry, head_count, head_dimension, lower_bound, claim,
      )
    MoeRouterProjection(hidden_width, expert_count, float32_router) =>
      exact_moe_router_projection_abi(
        geometry, hidden_width, expert_count, float32_router, claim,
      )
    MoeSigmoidAndCorrectionBias(expert_count) =>
      exact_moe_sigmoid_correction_abi(requirements, expert_count, claim)
    MoeGroupedNoAuxiliaryLossTopK(
      routed_expert_count,
      group_count,
      group_score_topk,
      selected_group_count,
      experts_per_token,
      normalize,
      scaling
    ) =>
      exact_moe_grouped_routing_abi(
        requirements, routed_expert_count, group_count, group_score_topk, selected_group_count,
        experts_per_token, normalize, scaling, claim,
      )
    RoutedExpertFeedForward(routed_expert_count, intermediate_size, _) =>
      exact_moe_routed_expert_abi(
        requirements, routed_expert_count, intermediate_size, claim,
      )
    SharedExpertFeedForward(shared_expert_count, intermediate_size, _) =>
      exact_moe_shared_expert_abi(
        requirements, shared_expert_count, intermediate_size, claim,
      )
    RoutedSharedExpertCombine(experts_per_token, normalize, scaling) =>
      exact_moe_routed_shared_combine_abi(
        requirements, experts_per_token, normalize, scaling, claim,
      )
    MoeDecoderResidualAdd(hidden_width) =>
      exact_moe_decoder_residual_add_abi(requirements, hidden_width, claim)
    HyperConnectionMix(stream_count) =>
      exact_hyper_connection_mix_abi(requirements, stream_count, claim)
    HyperConnectionFunction(
      stream_count,
      hidden_width,
      control_epsilon,
      normalization_epsilon
    ) =>
      exact_hyper_connection_function_abi(
        geometry, stream_count, hidden_width, control_epsilon, normalization_epsilon,
        claim,
      )
    HyperConnectionSinkhorn(stream_count, iterations, epsilon) =>
      exact_hyper_connection_sinkhorn_abi(
        geometry, stream_count, iterations, epsilon, claim,
      )
    DsaKeyValueAttentionAssembly(heads, nope, rope, value) =>
      exact_dsa_kv_attention_assembly_abi(
        requirements, heads, nope, rope, value, claim,
      )
    KdaShortConvolution(channel_count, kernel_size) =>
      exact_kda_short_convolution_abi(
        requirements, channel_count, kernel_size, claim,
      )
    KdaRecurrentUpdate(head_count, head_dimension, gate_lower_bound) =>
      exact_kda_recurrent_update_abi(
        requirements, head_count, head_dimension, gate_lower_bound, claim,
      )
    _ => true
  }
}
