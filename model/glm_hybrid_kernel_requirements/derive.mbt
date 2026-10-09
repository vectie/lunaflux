///|
fn ordered_capabilities(
  entries : ArrayView[GlmHybridKernelRequirement],
) -> ReadOnlyArray[GlmHybridAotCapability] {
  let result : Array[GlmHybridAotCapability] = []
  for entry in entries {
    if !result.contains(entry.capability) {
      result.push(entry.capability)
    }
  }
  ReadOnlyArray::from_array(result)
}

///|
fn add_hyper(
  entries : Array[GlmHybridKernelRequirement],
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
  limits : GlmHybridKernelRequirementLimits,
) -> Unit raise GlmHybridKernelRequirementsError {
  guard plan.hyper_connection() is Some(hyper) else { return }
  add_requirement(
    entries,
    HyperConnectionFunction(
      hyper.width(),
      plan.hidden_size(),
      hyper.control_epsilon(),
      hyper.normalization_epsilon(),
    ),
    HyperConnectionFunctionAot,
    limits,
  )
  add_requirement(
    entries,
    HyperConnectionSinkhorn(
      hyper.width(),
      hyper.sinkhorn_iterations(),
      hyper.epsilon(),
    ),
    advanced(HyperConnectionSinkhornAot),
    limits,
  )
  add_requirement(
    entries,
    HyperConnectionMix(hyper.width()),
    advanced(HyperConnectionMixAot),
    limits,
  )
}

///|
fn add_auxiliary(
  entries : Array[GlmHybridKernelRequirement],
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
  limits : GlmHybridKernelRequirementLimits,
) -> Unit raise GlmHybridKernelRequirementsError {
  let stages = match plan.auxiliary_prediction() {
    MultiTokenPrediction(layer_count~) => layer_count
    MultiTokenPredictionWithDspark(..) => raise UnsupportedSchema
  }
  add_requirement(
    entries,
    MultiTokenPredictionInputProjection,
    advanced(AuxiliaryProjectionAot),
    limits,
  )
  for stage in 0..<stages {
    add_requirement(
      entries,
      AuxiliaryDecoderStage(stage),
      advanced(AuxiliaryDecoderStageAot),
      limits,
    )
  }
  add_requirement(
    entries,
    AuxiliaryOutputHead,
    advanced(AuxiliaryProjectionAot),
    limits,
  )
}

///|
fn add_vision(
  entries : Array[GlmHybridKernelRequirement],
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
  limits : GlmHybridKernelRequirementLimits,
) -> Unit raise GlmHybridKernelRequirementsError {
  guard plan.vision_conditioning() is Some(vision) else { return }
  if vision.layer_count() > limits.maximum_vision_layers {
    raise PlanExceedsLimits
  }
  add_requirement(
    entries,
    VisionPatchProjection(
      vision.temporal_patch_size(),
      vision.spatial_patch_size(),
      3,
      vision.hidden_size(),
    ),
    VisionPatchProjectionAot,
    limits,
  )
  add_requirement(
    entries,
    VisionRmsNorm(vision.hidden_size()),
    VisionRmsNormAot,
    limits,
  )
  add_requirement(
    entries,
    VisionQkvProjection(vision.hidden_size(), vision.attention_head_count()),
    VisionQkvProjectionAot,
    limits,
  )
  add_requirement(
    entries,
    VisionAttention(
      vision.attention_head_count(),
      vision.hidden_size() / vision.attention_head_count(),
    ),
    VisionAttentionAot,
    limits,
  )
  add_requirement(
    entries,
    VisionOutputProjection(vision.hidden_size()),
    VisionOutputProjectionAot,
    limits,
  )
  add_requirement(
    entries,
    VisionSwiGlu(vision.hidden_size(), vision.intermediate_size()),
    VisionSwiGluAot,
    limits,
  )
  add_requirement(
    entries,
    VisionSpatialMerge(
      vision.hidden_size(),
      vision.decoder_projection_size(),
      vision.spatial_merge_size(),
    ),
    VisionSpatialMergeAot,
    limits,
  )
  add_requirement(
    entries,
    VisionSwiGlu(
      vision.decoder_projection_size(),
      vision.projection_intermediate_size(),
    ),
    VisionSwiGluAot,
    limits,
  )
  add_requirement(
    entries,
    VisionDecoderProjection(
      vision.decoder_projection_size(),
      plan.hidden_size(),
    ),
    VisionDecoderProjectionAot,
    limits,
  )
  add_requirement(
    entries,
    VisionPostProjectionNorm(plan.hidden_size()),
    VisionPostProjectionNormAot,
    limits,
  )
}

///|
pub fn derive_requirements(
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
  limits~ : GlmHybridKernelRequirementLimits,
) -> GlmHybridKernelRequirements raise GlmHybridKernelRequirementsError {
  if plan.schema_version() != 1 {
    raise UnsupportedSchema
  }
  if plan.layers().length() > limits.maximum_layers {
    raise PlanExceedsLimits
  }
  let entries : Array[GlmHybridKernelRequirement] = []
  add_requirement(entries, TokenEmbedding, advanced(EmbeddingLookupAot), limits)
  for layer in plan.layers() {
    add_requirement(
      entries,
      DecoderRmsNorm(
        plan.hidden_size(),
        plan.sparse_attention().rms_norm_epsilon(),
      ),
      DecoderRmsNormAot,
      limits,
    )
    add_hyper(entries, plan, limits)
    match layer.attention() {
      DynamicSparse(action) => add_dsa(entries, plan, action, limits)
      KimiDelta => add_kda(entries, plan, limits)
    }
    match layer.feed_forward() {
      DenseSwiGlu =>
        add_requirement(
          entries,
          DenseSwiGlu(plan.hidden_size(), plan.dense_intermediate_size()),
          DenseSwiGluAot,
          limits,
        )
      RoutedMoe => add_moe(entries, plan, limits)
    }
    if plan.hyper_connection() is None && layer.feed_forward() is RoutedMoe {
      add_requirement(
        entries,
        MoeDecoderResidualAdd(plan.hidden_size()),
        MoeDecoderResidualAddAot,
        limits,
      )
    }
  }
  add_auxiliary(entries, plan, limits)
  add_vision(entries, plan, limits)
  add_requirement(
    entries,
    FinalRmsNorm(plan.hidden_size(), plan.sparse_attention().rms_norm_epsilon()),
    advanced(RmsNormAot),
    limits,
  )
  add_requirement(
    entries,
    LanguageModelHead(plan.hidden_size(), plan.vocabulary_size()),
    advanced(DenseProjectionAot),
    limits,
  )
  let digest = canonical_digest(plan.identity(), entries)
  let geometry : GlmHybridKernelGeometry = {
    vocabulary_size: plan.vocabulary_size(),
    hidden_size: plan.hidden_size(),
    dense_intermediate_size: plan.dense_intermediate_size(),
    decoder_layer_count: plan.decoder_layer_count(),
    context_length: plan.context_length(),
  }
  {
    source_identity: plan.identity(),
    source_schema_version: plan.schema_version(),
    geometry,
    digest,
    requirements: ReadOnlyArray::from_array(entries),
    capabilities: ordered_capabilities(entries),
  }
}
