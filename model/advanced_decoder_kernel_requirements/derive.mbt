///|
fn add_unique_requirement(
  entries : Array[AdvancedDecoderKernelRequirement],
  operation : AdvancedDecoderSemanticOperation,
  capability : AdvancedDecoderAotCapability,
  limits : AdvancedDecoderKernelRequirementLimits,
) -> Unit raise AdvancedDecoderKernelRequirementsError {
  for index, entry in entries {
    if entry.operation == operation {
      if entry.capability == capability {
        return
      }
      raise DuplicateSemanticOperation(index)
    }
  }
  if entries.length() >= limits.maximum_requirements {
    raise PlanExceedsLimits
  }
  entries.push({ operation, capability, })
}

///|
fn add_attention_requirements(
  entries : Array[AdvancedDecoderKernelRequirement],
  attention : @advanced_decoder_execution_plan.AdvancedAttentionPlan,
  position : @advanced_decoder_execution_plan.PositionEncodingPlan,
  limits : AdvancedDecoderKernelRequirementLimits,
) -> Unit raise AdvancedDecoderKernelRequirementsError {
  match attention {
    SlidingWindowCausal(window_size~) =>
      add_unique_requirement(
        entries,
        SlidingWindowCausalAttention(window_size),
        SlidingWindowCausalAttentionAot,
        limits,
      )
    IndexedCompressed(
      compression_ratio~,
      index_head_count~,
      index_head_dimension~,
      index_top_k~
    ) => {
      add_unique_requirement(
        entries,
        LearnedCompressedIndex(
          compression_ratio, index_head_count, index_head_dimension, index_top_k,
        ),
        LearnedCompressedIndexAot,
        limits,
      )
      add_unique_requirement(
        entries,
        IndexedCompressedAttention(compression_ratio),
        IndexedCompressedAttentionAot,
        limits,
      )
    }
    DeterministicCompressed(compression_ratio~, window_size~) => {
      add_unique_requirement(
        entries,
        DeterministicCompressedIndex(compression_ratio),
        DeterministicCompressedIndexAot,
        limits,
      )
      add_unique_requirement(
        entries,
        DeterministicWindowCompressedIndexJoinV1(window_size, compression_ratio),
        DeterministicWindowCompressedIndexJoinV1Aot,
        limits,
      )
      add_unique_requirement(
        entries,
        NonOverlappingCompressedKeyValuePrepareV1(
          window_size, compression_ratio,
        ),
        NonOverlappingCompressedKeyValuePrepareV1Aot,
        limits,
      )
      add_unique_requirement(
        entries,
        SelectedSparseAttentionV1(window_size, compression_ratio),
        SelectedSparseAttentionV1Aot,
        limits,
      )
      add_unique_requirement(
        entries,
        AttentionOutputInverseRotaryV1(
          position.compressed_rope_theta(),
          position.original_position_count(),
          position.scaling_factor(),
          position.beta_fast(),
          position.beta_slow(),
        ),
        AttentionOutputInverseRotaryV1Aot,
        limits,
      )
    }
  }
}

///|
fn add_auxiliary_requirements(
  entries : Array[AdvancedDecoderKernelRequirement],
  auxiliary : @advanced_decoder_execution_plan.AuxiliaryPredictionPlan,
  limits : AdvancedDecoderKernelRequirementLimits,
) -> Unit raise AdvancedDecoderKernelRequirementsError {
  let (stage_count, dspark) = match auxiliary {
    MultiTokenPrediction(layer_count~) => (layer_count, None)
    MultiTokenPredictionWithDspark(layer_count~, dspark~) =>
      (layer_count, Some(dspark))
  }
  if stage_count > limits.maximum_auxiliary_stages {
    raise PlanExceedsLimits
  }
  match dspark {
    None =>
      add_unique_requirement(
        entries,
        MultiTokenPredictionInputProjection,
        AuxiliaryProjectionAot,
        limits,
      )
    Some(dspark) => {
      if dspark.target_layer_ids().length() > limits.maximum_layers {
        raise PlanExceedsLimits
      }
      for layer in dspark.target_layer_ids() {
        add_unique_requirement(
          entries,
          DsparkTargetLayerRead(layer),
          DsparkAot,
          limits,
        )
      }
      add_unique_requirement(entries, DsparkMainProjection, DsparkAot, limits)
      add_unique_requirement(
        entries,
        DsparkNoiseEmbedding(dspark.block_size(), dspark.noise_token_id()),
        DsparkAot,
        limits,
      )
    }
  }
  for stage in 0..<stage_count {
    add_unique_requirement(
      entries,
      AuxiliaryDecoderStage(stage),
      AuxiliaryDecoderStageAot,
      limits,
    )
  }
  add_unique_requirement(
    entries,
    AuxiliaryOutputHead,
    AuxiliaryProjectionAot,
    limits,
  )
  match dspark {
    Some(dspark) => {
      add_unique_requirement(
        entries,
        DsparkMarkovAdjustment(dspark.markov_rank()),
        DsparkAot,
        limits,
      )
      add_unique_requirement(entries, DsparkConfidence, DsparkAot, limits)
    }
    None => ()
  }
}

///|
fn ordered_capabilities(
  entries : ArrayView[AdvancedDecoderKernelRequirement],
) -> ReadOnlyArray[AdvancedDecoderAotCapability] {
  let capabilities : Array[AdvancedDecoderAotCapability] = []
  for entry in entries {
    if !capabilities.contains(entry.capability) {
      capabilities.push(entry.capability)
    }
  }
  ReadOnlyArray::from_array(capabilities)
}

///|
/// Derives the complete, unique, first-use-ordered semantic/AOT requirement
/// set. This performs bounded startup allocation and no artifact selection.
pub fn derive_requirements(
  plan : @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan,
  limits~ : AdvancedDecoderKernelRequirementLimits,
) -> AdvancedDecoderKernelRequirements raise AdvancedDecoderKernelRequirementsError {
  if plan.schema_version() != 1 {
    raise UnsupportedSchema
  }
  if plan.layers().length() > limits.maximum_layers {
    raise PlanExceedsLimits
  }
  let source_identity = @spec.ModelIdentity::new(
    plan.content_digest(),
    plan.plan_digest(),
  ) catch {
    _ => raise InvalidSourceIdentity
  }
  let entries : Array[AdvancedDecoderKernelRequirement] = []
  let shape = plan.shape()
  let low_rank = plan.low_rank_attention()
  let position = plan.position_encoding()
  let hyper = plan.hyper_connection()
  let moe = plan.moe()
  add_unique_requirement(entries, TokenEmbedding, EmbeddingLookupAot, limits)
  add_unique_requirement(
    entries,
    QueryLowRankAProjection(low_rank.query_rank()),
    QueryLowRankAProjectionAot,
    limits,
  )
  add_unique_requirement(
    entries,
    KeyValueProjection(
      shape.kv_head_count(),
      shape.attention_head_dimension(),
      shape.rope_head_dimension(),
    ),
    LowRankAttentionProjectionAot,
    limits,
  )
  add_unique_requirement(
    entries,
    QueryKeyRmsNorm(
      low_rank.query_key_norm_epsilon(),
      low_rank.query_rank(),
      shape.attention_head_dimension(),
    ),
    QueryKeyRmsNormAot,
    limits,
  )
  add_unique_requirement(
    entries,
    QueryLowRankBProjection(
      low_rank.query_rank(),
      shape.attention_head_count(),
      shape.attention_head_dimension(),
    ),
    QueryLowRankBProjectionAot,
    limits,
  )
  add_unique_requirement(
    entries,
    RotaryPositionEncoding(
      position.rope_theta(),
      position.compressed_rope_theta(),
      position.original_position_count(),
      position.scaling_factor(),
      position.beta_fast(),
      position.beta_slow(),
    ),
    ScaledRotaryPositionAot,
    limits,
  )
  add_unique_requirement(
    entries,
    HyperConnectionBlockControlV1(
      hyper.width(),
      hyper.sinkhorn_iterations(),
      hyper.control_epsilon(),
      hyper.normalization_epsilon(),
    ),
    HyperConnectionBlockControlV1Aot,
    limits,
  )
  add_unique_requirement(
    entries,
    HyperConnectionPreReduceV1(hyper.width()),
    HyperConnectionPreReduceV1Aot,
    limits,
  )
  add_unique_requirement(
    entries,
    HyperConnectionPostCombineV1(hyper.width()),
    HyperConnectionPostCombineV1Aot,
    limits,
  )
  add_unique_requirement(
    entries,
    HyperConnectionHeadReduceV1(
      hyper.width(),
      hyper.control_epsilon(),
      hyper.normalization_epsilon(),
    ),
    HyperConnectionHeadReduceV1Aot,
    limits,
  )
  for layer in plan.layers() {
    add_attention_requirements(entries, layer.attention(), position, limits)
  }
  add_unique_requirement(
    entries,
    AttentionOutputAProjectionV1(
      low_rank.output_rank(),
      low_rank.output_group_count(),
      shape.attention_head_count(),
      shape.attention_head_dimension(),
    ),
    AttentionOutputAProjectionV1Aot,
    limits,
  )
  add_unique_requirement(
    entries,
    AttentionOutputBProjectionV1(
      low_rank.output_rank(),
      low_rank.output_group_count(),
      shape.hidden_size(),
    ),
    AttentionOutputBProjectionV1Aot,
    limits,
  )
  add_unique_requirement(
    entries,
    ExpertScoreTransform(moe.score_function(), moe.routed_expert_count()),
    ExpertScoreTransformAot,
    limits,
  )
  match moe.selection_method() {
    NoAuxiliaryLossTokenChoice => ()
    TokenHashThenNoAuxiliaryLossTokenChoice(hash_layer_count~) =>
      add_unique_requirement(
        entries,
        TokenHashRouting(
          hash_layer_count,
          shape.vocabulary_size(),
          moe.experts_per_token(),
          moe.routed_expert_count(),
        ),
        TokenHashRoutingAot,
        limits,
      )
  }
  add_unique_requirement(
    entries,
    NoAuxiliaryLossTopK(moe.routed_expert_count(), moe.experts_per_token()),
    StableTopKRoutingAot,
    limits,
  )
  add_unique_requirement(
    entries,
    SelectedExpertWeightFinalize(
      moe.routed_expert_count(),
      moe.experts_per_token(),
      moe.normalize_selected_probabilities(),
      moe.routed_scaling_factor(),
    ),
    SelectedExpertWeightFinalizeAot,
    limits,
  )
  add_unique_requirement(
    entries,
    RoutedExpertFeedForward(
      moe.routed_expert_count(),
      moe.intermediate_size(),
      moe.activation(),
    ),
    RoutedExpertFeedForwardAot,
    limits,
  )
  if moe.shared_expert_count() > 0 {
    add_unique_requirement(
      entries,
      SharedExpertFeedForward(
        moe.shared_expert_count(),
        moe.intermediate_size(),
        moe.activation(),
      ),
      SharedExpertFeedForwardAot,
      limits,
    )
  }
  add_unique_requirement(
    entries,
    RoutedSharedExpertCombine(
      moe.experts_per_token(),
      moe.normalize_selected_probabilities(),
      moe.routed_scaling_factor(),
    ),
    RoutedSharedExpertCombineAot,
    limits,
  )
  add_auxiliary_requirements(entries, plan.auxiliary_prediction(), limits)
  add_unique_requirement(
    entries,
    FinalRmsNorm(low_rank.query_key_norm_epsilon()),
    RmsNormAot,
    limits,
  )
  add_unique_requirement(entries, LanguageModelHead, DenseProjectionAot, limits)
  if entries.is_empty() {
    abort("advanced decoder requirements unexpectedly empty")
  }
  let digest = canonical_requirements_digest(source_identity, entries)
  let geometry : AdvancedDecoderKernelGeometry = {
    vocabulary_size: shape.vocabulary_size(),
    hidden_size: shape.hidden_size(),
    decoder_layer_count: shape.decoder_layer_count(),
    attention_head_count: shape.attention_head_count(),
    kv_head_count: shape.kv_head_count(),
    attention_head_dimension: shape.attention_head_dimension(),
    rope_head_dimension: shape.rope_head_dimension(),
    maximum_position_count: shape.maximum_position_count(),
  }
  {
    source_identity,
    source_schema_version: plan.schema_version(),
    geometry,
    digest,
    requirements: ReadOnlyArray::from_array(entries),
    capabilities: ordered_capabilities(entries),
  }
}
