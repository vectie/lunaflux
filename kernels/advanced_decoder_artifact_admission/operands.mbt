///|
fn expected_operand_roles(
  operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation,
) -> ReadOnlyArray[AdvancedDecoderLaunchOperandRole] {
  match operation {
    TokenEmbedding =>
      [StepCountsInput, TokenIdsInput, ModelWeights, HiddenOutput]
    QueryLowRankAProjection(_) =>
      [
        StepCountsInput,
        HiddenInput,
        QueryAWeightCodesInput,
        QueryAWeightScalesInput,
        QueryLowRankAOutput,
      ]
    QueryLowRankBProjection(_, _, _) => [HiddenInput]
    KeyValueProjection(_, _, _) => key_value_projection_roles()
    AttentionOutputAProjectionV1(_, _, _, _) =>
      [
        StepCountsInput,
        AttentionOutputProjectionInput,
        AttentionOutputAWeightsInput,
        AttentionOutputRankOutput,
      ]
    AttentionOutputBProjectionV1(_, _, _) =>
      [
        StepCountsInput,
        AttentionOutputRankInput,
        AttentionOutputBWeightCodesInput,
        AttentionOutputBWeightScalesInput,
        HiddenOutput,
      ]
    FinalRmsNorm(_) =>
      [StepCountsInput, HiddenInput, ModelWeights, HiddenOutput]
    QueryKeyRmsNorm(_, _, _) =>
      [
        StepCountsInput,
        QueryInput,
        QueryNormWeights,
        QueryOutput,
        KeyValueInput,
        KeyValueNormWeights,
        KeyValueOutput,
      ]
    RotaryPositionEncoding(_, _, _, _, _, _) =>
      [
        StepCountsInput,
        PositionInput,
        QueryInput,
        QueryOutput,
        KeyValueInput,
        KeyValueOutput,
      ]
    HyperConnectionBlockControlV1(_, _, _, _) => block_control_roles()
    HyperConnectionPreReduceV1(_) => pre_reduce_roles()
    HyperConnectionPostCombineV1(_) => post_combine_roles()
    HyperConnectionHeadReduceV1(_, _, _) => head_reduce_roles()
    SlidingWindowCausalAttention(_) | IndexedCompressedAttention(_) =>
      [HiddenInput, ModelWeights, HiddenOutput, Workspace]
    NonOverlappingCompressedKeyValuePrepareV1(_, _) =>
      [
        StepCountsInput,
        PreparedKeyValueModesInput,
        PreparedKeyValueSequenceLengthsInput,
        UncompressedKeyValueInput,
        CompressedKeyValueInput,
        PreparedKeyValueLengthsOutput,
        PreparedKeyValueOutput,
      ]
    AttentionOutputInverseRotaryV1(_, _, _, _, _) =>
      [StepCountsInput, PositionInput, AttentionOutputRotaryInOut]
    LearnedCompressedIndex(_, _, _, _) =>
      [HiddenInput, ModelWeights, RoutingIndicesOutput, Workspace]
    DeterministicCompressedIndex(_) =>
      [
        StepCountsInput,
        CompressedQueryPositionsInput,
        CompressedCacheOffsetsInput,
        CompressedIndexCountsOutput,
        CompressedIndicesOutput,
      ]
    DeterministicWindowCompressedIndexJoinV1(_, _) =>
      [
        StepCountsInput,
        WindowQueryPositionsInput,
        WindowSequenceLengthsInput,
        WindowQueryModesInput,
        CompressedIndexCountsInput,
        CompressedIndicesInput,
        JoinedIndexCountsOutput,
        JoinedIndicesOutput,
      ]
    SelectedSparseAttentionV1(_, _) =>
      [
        StepCountsInput,
        SparseAttentionQueryInput,
        SparseAttentionQueryBatchIdsInput,
        SparseAttentionKeyValueInput,
        SparseAttentionCacheLengthsInput,
        SparseAttentionSinkInput,
        SparseAttentionSelectedIndicesInput,
        SparseAttentionOutput,
      ]
    ExpertScoreTransform(_, _) =>
      [StepCountsInput, RoutingScoresInput, RoutingScoresOutput]
    TokenHashRouting(_, _, _, _) =>
      [
        StepCountsInput,
        TokenIdsInput,
        TokenHashTableInput,
        RoutingIndicesOutput,
      ]
    NoAuxiliaryLossTopK(_, _) =>
      [
        StepCountsInput,
        RoutingScoresInput,
        RoutingSelectionBiasInput,
        RoutingIndicesOutput,
      ]
    SelectedExpertWeightFinalize(_, _, _, _) =>
      [
        StepCountsInput,
        RoutingScoresInput,
        RoutingIndicesInput,
        RoutingWeightsOutput,
      ]
    RoutedExpertFeedForward(_, _, _) =>
      [HiddenInput, RoutingIndicesInput, ModelWeights, ExpertOutput]
    SharedExpertFeedForward(_, _, _) =>
      [HiddenInput, ModelWeights, ExpertOutput]
    RoutedSharedExpertCombine(_, _, _) =>
      [ExpertInput, RoutingScoresInput, HiddenOutput]
    MultiTokenPredictionInputProjection =>
      [HiddenInput, ModelWeights, AuxiliaryOutput]
    DsparkTargetLayerRead(_) => [HiddenInput, AuxiliaryOutput]
    DsparkMainProjection
    | DsparkNoiseEmbedding(_, _)
    | AuxiliaryDecoderStage(_)
    | AuxiliaryOutputHead
    | DsparkMarkovAdjustment(_)
    | DsparkConfidence => [AuxiliaryInput, ModelWeights, AuxiliaryOutput]
    LanguageModelHead =>
      [StepCountsInput, HiddenInput, ModelWeights, LogitsOutput]
  }
}

///|
fn operands_are_exact(
  operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  geometry : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelGeometry,
  maximum : Int,
) -> Bool {
  if operands.length() > maximum {
    return false
  }
  let expected = expected_operand_roles(operation)
  if operands.length() != expected.length() {
    return false
  }
  for index, operand in operands {
    if operand.role() != expected[index] {
      return false
    }
  }
  match operation {
    QueryLowRankAProjection(rank) =>
      query_a_projection_operands_are_exact(
        rank, operands, dimensions, geometry,
      )
    QueryLowRankBProjection(_, _, _) => false
    KeyValueProjection(heads, head_dimension, rope_dimension) =>
      key_value_projection_operands_are_exact(
        heads, head_dimension, rope_dimension, operands, dimensions, geometry,
      )
    AttentionOutputAProjectionV1(rank, groups, heads, head_dimension) =>
      output_a_projection_operands_are_exact(
        rank, groups, heads, head_dimension, operands, dimensions, geometry,
      )
    AttentionOutputBProjectionV1(rank, groups, hidden) =>
      output_b_projection_operands_are_exact(
        rank, groups, hidden, operands, dimensions, geometry,
      )
    NonOverlappingCompressedKeyValuePrepareV1(window_size, compression_ratio) =>
      compressed_kv_prepare_operands_are_exact(
        window_size, compression_ratio, operands, dimensions, geometry,
      )
    HyperConnectionBlockControlV1(
      width,
      iterations,
      control_epsilon,
      normalization_epsilon
    ) => {
      let hidden = geometry.hidden_size()
      if width <= 1 ||
        width > 16 ||
        hidden <= 0 ||
        hidden > 65536 ||
        iterations <= 0 ||
        iterations > 64 ||
        control_epsilon.is_nan() ||
        control_epsilon.is_inf() ||
        control_epsilon <= 0.0 ||
        normalization_epsilon.is_nan() ||
        normalization_epsilon.is_inf() ||
        normalization_epsilon <= 0.0 ||
        dimensions.grid_x() <= 0 ||
        dimensions.grid_y() != 1 ||
        dimensions.grid_z() != 1 ||
        dimensions.block_x() != 1 ||
        dimensions.block_y() != 1 ||
        dimensions.block_z() != 1 ||
        dimensions.shared_memory_bytes() != 0 {
        return false
      }
      let rows = dimensions.grid_x().to_int64()
      let stream = width.to_int64()
      let feature = hidden.to_int64()
      if stream > 9223372036854775807L / feature {
        return false
      }
      let flattened = stream * feature
      if rows > 9223372036854775807L / flattened ||
        rows * flattened > 9223372036854775807L / 2L ||
        6L * stream > 9223372036854775807L / flattened ||
        6L * stream * flattened > 9223372036854775807L / 4L ||
        rows > 9223372036854775807L / stream ||
        rows * stream > 9223372036854775807L / stream ||
        rows * stream * stream > 9223372036854775807L / 4L {
        return false
      }
      let state_bytes = rows * flattened * 2L
      let function_bytes = 6L * stream * flattened * 4L
      let vector_bytes = rows * stream * 4L
      let combination_bytes = rows * stream * stream * 4L
      operands[0].byte_count() == 20L &&
      operands[0].alignment() == 4L &&
      operands[1].byte_count() == state_bytes &&
      operands[1].alignment() == 16L &&
      operands[2].byte_count() == function_bytes &&
      operands[2].alignment() == 16L &&
      operands[3].byte_count() == 6L * stream * 4L &&
      operands[3].alignment() == 16L &&
      operands[4].byte_count() == 12L &&
      operands[4].alignment() == 4L &&
      operands[5].byte_count() == vector_bytes &&
      operands[5].alignment() == 16L &&
      operands[6].byte_count() == vector_bytes &&
      operands[6].alignment() == 16L &&
      operands[7].byte_count() == combination_bytes &&
      operands[7].alignment() == 16L
    }
    HyperConnectionPreReduceV1(width) => {
      let hidden = geometry.hidden_size()
      if width <= 1 ||
        width > 16 ||
        hidden <= 0 ||
        hidden > 65536 ||
        dimensions.grid_x() <= 0 ||
        dimensions.grid_y() != 1 ||
        dimensions.grid_z() != 1 ||
        dimensions.block_x() != 256 ||
        dimensions.block_y() != 1 ||
        dimensions.block_z() != 1 ||
        dimensions.shared_memory_bytes() != 0 {
        return false
      }
      let rows = dimensions.grid_x().to_int64()
      let stream = width.to_int64()
      let feature = hidden.to_int64()
      if stream > 9223372036854775807L / feature ||
        rows > 9223372036854775807L / stream ||
        rows * stream > 9223372036854775807L / 4L ||
        rows * stream > 9223372036854775807L / feature ||
        rows * stream * feature > 9223372036854775807L / 2L ||
        rows > 9223372036854775807L / feature ||
        rows * feature > 9223372036854775807L / 2L {
        return false
      }
      let state_bytes = rows * stream * feature * 2L
      let pre_bytes = rows * stream * 4L
      let output_bytes = rows * feature * 2L
      operands[0].byte_count() == 20L &&
      operands[0].alignment() == 4L &&
      operands[1].byte_count() == state_bytes &&
      operands[1].alignment() == 16L &&
      operands[2].byte_count() == pre_bytes &&
      operands[2].alignment() == 16L &&
      operands[3].byte_count() == output_bytes &&
      operands[3].alignment() == 16L
    }
    HyperConnectionPostCombineV1(width) => {
      let hidden = geometry.hidden_size()
      if width <= 1 ||
        width > 16 ||
        hidden <= 0 ||
        hidden > 65536 ||
        dimensions.grid_x() <= 0 ||
        dimensions.grid_y() != 1 ||
        dimensions.grid_z() != 1 ||
        dimensions.block_x() != 256 ||
        dimensions.block_y() != 1 ||
        dimensions.block_z() != 1 ||
        dimensions.shared_memory_bytes() != 0 {
        return false
      }
      let rows = dimensions.grid_x().to_int64()
      let stream = width.to_int64()
      let feature = hidden.to_int64()
      if rows > 9223372036854775807L / feature ||
        rows * feature > 9223372036854775807L / 2L ||
        stream > 9223372036854775807L / feature ||
        rows > 9223372036854775807L / stream ||
        rows * stream > 9223372036854775807L / 4L ||
        rows * stream > 9223372036854775807L / feature ||
        rows * stream * feature > 9223372036854775807L / 2L ||
        rows * stream > 9223372036854775807L / stream ||
        rows * stream * stream > 9223372036854775807L / 4L {
        return false
      }
      let branch_bytes = rows * feature * 2L
      let state_bytes = rows * stream * feature * 2L
      let post_bytes = rows * stream * 4L
      let combination_bytes = rows * stream * stream * 4L
      operands[0].byte_count() == 20L &&
      operands[0].alignment() == 4L &&
      operands[1].byte_count() == branch_bytes &&
      operands[1].alignment() == 16L &&
      operands[2].byte_count() == state_bytes &&
      operands[2].alignment() == 16L &&
      operands[3].byte_count() == post_bytes &&
      operands[3].alignment() == 16L &&
      operands[4].byte_count() == combination_bytes &&
      operands[4].alignment() == 16L &&
      operands[5].byte_count() == state_bytes &&
      operands[5].alignment() == 16L
    }
    DeterministicCompressedIndex(compression_ratio) =>
      deterministic_compressed_index_operands_are_exact(
        compression_ratio, operands, dimensions, geometry,
      )
    DeterministicWindowCompressedIndexJoinV1(window_size, compression_ratio) =>
      window_compressed_join_operands_are_exact(
        window_size, compression_ratio, operands, dimensions, geometry,
      )
    SelectedSparseAttentionV1(window_size, compression_ratio) =>
      selected_sparse_attention_operands_are_exact(
        window_size, compression_ratio, operands, dimensions, geometry,
      )
    HyperConnectionHeadReduceV1(width, _, _) =>
      head_reduce_operands_are_exact(width, operands, dimensions, geometry)
    RotaryPositionEncoding(_, _, original_positions, _, _, _) => {
      let query_head_count = geometry.attention_head_count()
      let key_value_head_count = geometry.kv_head_count()
      let rope_dimension = geometry.rope_head_dimension()
      if original_positions <= 0 ||
        original_positions > geometry.maximum_position_count() ||
        query_head_count <= 0 ||
        key_value_head_count <= 0 ||
        rope_dimension != 64 ||
        dimensions.grid_x() <= 0 ||
        dimensions.grid_y() != query_head_count + key_value_head_count ||
        dimensions.grid_z() != 1 ||
        dimensions.block_x() != 32 ||
        dimensions.block_y() != 1 ||
        dimensions.block_z() != 1 ||
        dimensions.shared_memory_bytes() != 0 {
        return false
      }
      let rows = dimensions.grid_x().to_int64()
      let query_heads = query_head_count.to_int64()
      let key_value_heads = key_value_head_count.to_int64()
      if rows > 9223372036854775807L / query_heads ||
        rows * query_heads > 9223372036854775807L / 128L ||
        rows > 9223372036854775807L / key_value_heads ||
        rows * key_value_heads > 9223372036854775807L / 128L {
        return false
      }
      let query_bytes = rows * query_heads * 128L
      let key_value_bytes = rows * key_value_heads * 128L
      operands[0].byte_count() == 20L &&
      operands[0].alignment() == 4L &&
      operands[1].byte_count() == rows * 4L &&
      operands[1].alignment() == 4L &&
      operands[2].byte_count() == query_bytes &&
      operands[2].alignment() == 16L &&
      operands[3].byte_count() == query_bytes &&
      operands[3].alignment() == 16L &&
      operands[4].byte_count() == key_value_bytes &&
      operands[4].alignment() == 16L &&
      operands[5].byte_count() == key_value_bytes &&
      operands[5].alignment() == 16L
    }
    AttentionOutputInverseRotaryV1(_, original_positions, _, _, _) =>
      attention_output_inverse_rotary_operands_are_exact(
        original_positions, operands, dimensions, geometry,
      )
    LanguageModelHead =>
      operands[0].byte_count() == 20L && operands[0].alignment() == 4L
    ExpertScoreTransform(_, routed_expert_count) =>
      expert_score_operands_are_exact(routed_expert_count, operands, dimensions)
    QueryKeyRmsNorm(_, query_width, key_value_width) =>
      query_key_norm_operands_are_exact(
        query_width, key_value_width, operands, dimensions,
      )
    TokenHashRouting(
      hash_layer_count,
      vocabulary_size,
      experts_per_token,
      routed_expert_count
    ) => {
      if hash_layer_count <= 0 ||
        vocabulary_size <= 0 ||
        experts_per_token <= 0 ||
        experts_per_token > 256 ||
        routed_expert_count <= 0 ||
        experts_per_token > routed_expert_count ||
        dimensions.grid_x() <= 0 ||
        dimensions.grid_y() != 1 ||
        dimensions.grid_z() != 1 ||
        dimensions.block_x() != 256 ||
        dimensions.block_y() != 1 ||
        dimensions.block_z() != 1 ||
        dimensions.shared_memory_bytes() != 0 {
        return false
      }
      let rows = dimensions.grid_x().to_int64()
      let selected = experts_per_token.to_int64()
      let vocabulary = vocabulary_size.to_int64()
      if rows > 9223372036854775807L / 4L ||
        vocabulary > 9223372036854775807L / selected ||
        vocabulary * selected > 9223372036854775807L / 4L ||
        rows > 9223372036854775807L / selected ||
        rows * selected > 9223372036854775807L / 4L {
        return false
      }
      operands[0].byte_count() == 20L &&
      operands[0].alignment() == 4L &&
      operands[1].byte_count() == rows * 4L &&
      operands[1].alignment() == 4L &&
      operands[2].byte_count() == vocabulary * selected * 4L &&
      operands[2].alignment() == 4L &&
      operands[3].byte_count() == rows * selected * 4L &&
      operands[3].alignment() == 4L
    }
    NoAuxiliaryLossTopK(routed_expert_count, top_k)
    | SelectedExpertWeightFinalize(routed_expert_count, top_k, _, _) => {
      if routed_expert_count <= 0 ||
        top_k <= 0 ||
        top_k > routed_expert_count ||
        top_k > 256 ||
        dimensions.grid_x() <= 0 ||
        dimensions.grid_y() != 1 ||
        dimensions.grid_z() != 1 ||
        dimensions.block_x() != 1 ||
        dimensions.block_y() != 1 ||
        dimensions.block_z() != 1 ||
        dimensions.shared_memory_bytes() != 0 {
        return false
      }
      let rows = dimensions.grid_x().to_int64()
      let routed = routed_expert_count.to_int64()
      let selected = top_k.to_int64()
      if rows > 9223372036854775807L / routed ||
        rows * routed > 9223372036854775807L / 4L ||
        rows > 9223372036854775807L / selected ||
        rows * selected > 9223372036854775807L / 4L {
        return false
      }
      let scores_bytes = rows * routed * 4L
      let selected_bytes = rows * selected * 4L
      if operation is NoAuxiliaryLossTopK(_, _) {
        operands[0].byte_count() == 20L &&
        operands[0].alignment() == 4L &&
        operands[1].byte_count() == scores_bytes &&
        operands[1].alignment() == 4L &&
        operands[2].byte_count() == routed * 4L &&
        operands[2].alignment() == 4L &&
        operands[3].byte_count() == selected_bytes &&
        operands[3].alignment() == 4L
      } else {
        operands[0].byte_count() == 20L &&
        operands[0].alignment() == 4L &&
        operands[1].byte_count() == scores_bytes &&
        operands[1].alignment() == 4L &&
        operands[2].byte_count() == selected_bytes &&
        operands[2].alignment() == 4L &&
        operands[3].byte_count() == selected_bytes &&
        operands[3].alignment() == 4L
      }
    }
    _ => true
  }
}
