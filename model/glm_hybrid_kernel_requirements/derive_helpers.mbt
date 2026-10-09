///|
fn add_requirement(
  entries : Array[GlmHybridKernelRequirement],
  operation : GlmHybridSemanticOperation,
  capability : GlmHybridAotCapability,
  limits : GlmHybridKernelRequirementLimits,
) -> Unit raise GlmHybridKernelRequirementsError {
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
fn advanced(
  capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability,
) -> GlmHybridAotCapability {
  Advanced(capability)
}

///|
fn add_dsa(
  entries : Array[GlmHybridKernelRequirement],
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
  action : @glm_hybrid_execution_plan.GlmSparseIndexAction,
  limits : GlmHybridKernelRequirementLimits,
) -> Unit raise GlmHybridKernelRequirementsError {
  let sparse = plan.sparse_attention()
  add_requirement(
    entries,
    DsaQueryAProjection(plan.hidden_size(), sparse.query_lora_rank()),
    DsaQueryAProjectionAot,
    limits,
  )
  add_requirement(
    entries,
    DsaQueryARmsNorm(sparse.query_lora_rank(), sparse.rms_norm_epsilon()),
    DsaQueryARmsNormAot,
    limits,
  )
  add_requirement(
    entries,
    DsaQueryBProjection(
      sparse.query_lora_rank(),
      sparse.query_head_count(),
      sparse.qk_nope_head_dimension() + sparse.qk_rope_head_dimension(),
    ),
    DsaQueryBProjectionAot,
    limits,
  )
  add_requirement(
    entries,
    DsaKeyValueAProjection(
      plan.hidden_size(),
      sparse.kv_lora_rank(),
      sparse.qk_rope_head_dimension(),
    ),
    DsaKeyValueAProjectionAot,
    limits,
  )
  add_requirement(
    entries,
    QueryLowRankProjection(sparse.query_lora_rank()),
    advanced(LowRankAttentionProjectionAot),
    limits,
  )
  add_requirement(
    entries,
    KeyValueLowRankProjection(
      sparse.kv_lora_rank(),
      sparse.kv_head_count(),
      sparse.qk_nope_head_dimension(),
      sparse.value_head_dimension(),
    ),
    advanced(LowRankAttentionProjectionAot),
    limits,
  )
  add_requirement(
    entries,
    QueryKeyRmsNorm(sparse.rms_norm_epsilon()),
    advanced(QueryKeyRmsNormAot),
    limits,
  )
  add_requirement(
    entries,
    GlmRotaryPosition(sparse.rope_theta(), sparse.rope_interleave()),
    GlmRotaryPositionAot,
    limits,
  )
  add_requirement(
    entries,
    DsaKeyValueAttentionAssembly(
      sparse.kv_head_count(),
      sparse.qk_nope_head_dimension(),
      sparse.qk_rope_head_dimension(),
      sparse.value_head_dimension(),
    ),
    DsaKeyValueAttentionAssemblyAot,
    limits,
  )
  match action {
    ComputeTopK => {
      add_requirement(
        entries,
        DsaIndexCompute(
          sparse.index_head_count(),
          sparse.index_head_dimension(),
          sparse.index_topk(),
          sparse.indexer_rope_interleave(),
        ),
        DsaIndexComputeAot,
        limits,
      )
      match sparse.index_pooling() {
        Some(pool) =>
          add_requirement(
            entries,
            DsaIndexPooling(
              pool.kernel_size(),
              pool.compress(),
              pool.always_select_tail(),
            ),
            DsaIndexPoolingAot,
            limits,
          )
        None => ()
      }
    }
    ReusePreviousTopK =>
      add_requirement(entries, DsaIndexReuse, DsaIndexReuseAot, limits)
  }
  add_requirement(
    entries,
    DsaSparseAttention(
      sparse.query_head_count(),
      sparse.index_topk(),
      sparse.qk_nope_head_dimension() + sparse.qk_rope_head_dimension(),
      sparse.value_head_dimension(),
    ),
    DsaSparseAttentionAot,
    limits,
  )
  add_requirement(
    entries,
    AttentionOutputProjection(
      plan.hidden_size(),
      sparse.query_head_count() * sparse.value_head_dimension(),
    ),
    AttentionOutputProjectionAot,
    limits,
  )
}

///|
fn add_kda(
  entries : Array[GlmHybridKernelRequirement],
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
  limits : GlmHybridKernelRequirementLimits,
) -> Unit raise GlmHybridKernelRequirementsError {
  guard plan.kda() is Some(kda) else { raise UnsupportedSchema }
  add_requirement(
    entries,
    KdaProjection(kda.head_count(), kda.head_dimension()),
    KdaProjectionAot,
    limits,
  )
  add_requirement(
    entries,
    KdaShortConvolution(
      kda.projected_width(),
      kda.short_convolution_kernel_size(),
    ),
    KdaShortConvolutionAot,
    limits,
  )
  add_requirement(
    entries,
    KdaDecayControl(
      kda.head_count(),
      kda.head_dimension(),
      kda.gate_lower_bound(),
    ),
    KdaDecayControlAot,
    limits,
  )
  add_requirement(
    entries,
    KdaRecurrentUpdate(
      kda.head_count(),
      kda.head_dimension(),
      kda.gate_lower_bound(),
    ),
    KdaRecurrentUpdateAot,
    limits,
  )
  add_requirement(
    entries,
    KdaOutputGateProjection(
      plan.hidden_size(),
      kda.head_count(),
      kda.head_dimension(),
    ),
    KdaOutputGateProjectionAot,
    limits,
  )
  add_requirement(
    entries,
    KdaGatedRmsNorm(
      kda.head_count(),
      kda.head_dimension(),
      plan.sparse_attention().rms_norm_epsilon(),
    ),
    KdaGatedRmsNormAot,
    limits,
  )
  add_requirement(
    entries,
    KdaOutputProjection(kda.projected_width(), plan.hidden_size()),
    KdaOutputProjectionAot,
    limits,
  )
}

///|
fn add_moe(
  entries : Array[GlmHybridKernelRequirement],
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
  limits : GlmHybridKernelRequirementLimits,
) -> Unit raise GlmHybridKernelRequirementsError {
  let moe = plan.moe()
  add_requirement(
    entries,
    MoeRouterProjection(
      plan.hidden_size(),
      moe.routed_expert_count(),
      plan.moe_router_float32(),
    ),
    MoeRouterProjectionAot,
    limits,
  )
  if plan.moe_correction_bias() {
    if moe.score_function() != Sigmoid {
      raise UnsupportedSchema
    }
    add_requirement(
      entries,
      MoeSigmoidAndCorrectionBias(moe.routed_expert_count()),
      MoeSigmoidAndCorrectionBiasAot,
      limits,
    )
  } else {
    add_requirement(
      entries,
      ExpertScoreTransform(moe.score_function()),
      advanced(ExpertScoreTransformAot),
      limits,
    )
  }
  add_requirement(
    entries,
    MoeGroupedNoAuxiliaryLossTopK(
      moe.routed_expert_count(),
      1,
      2,
      1,
      moe.experts_per_token(),
      moe.normalize_selected_probabilities(),
      moe.routed_scaling_factor(),
    ),
    MoeGroupedRoutingAot,
    limits,
  )
  add_requirement(
    entries,
    RoutedExpertFeedForward(
      moe.routed_expert_count(),
      moe.intermediate_size(),
      moe.activation(),
    ),
    advanced(RoutedExpertFeedForwardAot),
    limits,
  )
  add_requirement(
    entries,
    SharedExpertFeedForward(
      moe.shared_expert_count(),
      moe.intermediate_size(),
      moe.activation(),
    ),
    advanced(SharedExpertFeedForwardAot),
    limits,
  )
  add_requirement(
    entries,
    RoutedSharedExpertCombine(
      moe.experts_per_token(),
      moe.normalize_selected_probabilities(),
      moe.routed_scaling_factor(),
    ),
    advanced(RoutedSharedExpertCombineAot),
    limits,
  )
}
