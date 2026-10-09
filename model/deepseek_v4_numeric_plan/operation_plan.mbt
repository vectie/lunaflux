///|
fn append_operation(
  operations : Array[DeepSeekV4NumericOperationBinding],
  scope : DeepSeekV4NumericOperationScope,
  kind : DeepSeekV4NumericOperationKind,
  execution : DeepSeekV4OperationExecution,
) -> Unit raise DeepSeekV4NumericPlanError {
  if operations.length() >= 4096 {
    raise LogicalSchemaLimitExceeded
  }
  operations.push({ ordinal: operations.length(), scope, kind, execution, })
}

///|
fn append_numeric(
  operations : Array[DeepSeekV4NumericOperationBinding],
  scope : DeepSeekV4NumericOperationScope,
  kind : DeepSeekV4NumericOperationKind,
  contract : @numeric_contract.OperationExecutionContract,
) -> Unit raise DeepSeekV4NumericPlanError {
  append_operation(operations, scope, kind, NumericContract(contract))
}

///|
fn append_hyper_connection(
  operations : Array[DeepSeekV4NumericOperationBinding],
  scope : DeepSeekV4NumericOperationScope,
  site : DeepSeekV4HyperConnectionBlockSite,
) -> Unit raise DeepSeekV4NumericPlanError {
  append_numeric(
    operations,
    scope,
    HyperConnectionBlockControl(site),
    bf16_f32_control_to_f32_execution(),
  )
  append_numeric(
    operations,
    scope,
    HyperConnectionPreReduce(site),
    bf16_f32_control_execution(),
  )
  append_numeric(
    operations,
    scope,
    HyperConnectionPostCombine(site),
    bf16_f32_control_execution(),
  )
}

///|
fn append_attention(
  operations : Array[DeepSeekV4NumericOperationBinding],
  scope : DeepSeekV4NumericOperationScope,
  attention : @advanced_decoder_execution_plan.AdvancedAttentionPlan,
) -> Unit raise DeepSeekV4NumericPlanError {
  append_numeric(
    operations,
    scope,
    LowRankAttentionProjection,
    fp8_block128_ue8m0_projection_execution(),
  )
  append_numeric(operations, scope, QueryKeyRmsNorm, bf16_tensor_execution())
  append_numeric(
    operations,
    scope,
    ScaledRotaryPosition,
    bf16_activation_execution(),
  )
  match attention {
    SlidingWindowCausal(..) =>
      append_numeric(
        operations,
        scope,
        SlidingWindowAttention,
        bf16_activation_execution(),
      )
    IndexedCompressed(..) => {
      append_numeric(
        operations,
        scope,
        LearnedIndexSelection,
        fp8_projection_execution(),
      )
      append_numeric(
        operations,
        scope,
        IndexedCompressedAttention,
        bf16_activation_execution(),
      )
    }
    DeterministicCompressed(..) => {
      append_numeric(
        operations,
        scope,
        DeterministicCompressedSelection,
        bf16_activation_execution(),
      )
      append_numeric(
        operations,
        scope,
        DeterministicWindowCompressedIndexJoinV1,
        bf16_activation_execution(),
      )
      append_numeric(
        operations,
        scope,
        NonOverlappingCompressedKeyValuePrepareV1,
        bf16_f32_control_execution(),
      )
      append_numeric(
        operations,
        scope,
        SelectedSparseAttentionV1,
        bf16_activation_execution(),
      )
      append_numeric(
        operations,
        scope,
        AttentionOutputInverseRotaryV1,
        bf16_activation_execution(),
      )
    }
  }
  append_numeric(
    operations,
    scope,
    AttentionOutputAProjectionV1,
    bf16_tensor_execution(),
  )
  append_numeric(
    operations,
    scope,
    AttentionOutputBProjectionV1,
    fp8_block128_ue8m0_projection_execution(),
  )
}

///|
fn append_feed_forward(
  operations : Array[DeepSeekV4NumericOperationBinding],
  scope : DeepSeekV4NumericOperationScope,
  hash_routed : Bool,
  fp4_routed : Bool,
) -> Unit raise DeepSeekV4NumericPlanError {
  append_numeric(
    operations,
    scope,
    MoeRouterProjection,
    bf16_router_execution(),
  )
  append_numeric(
    operations,
    scope,
    ExpertScoreTransform,
    f32_activation_execution(),
  )
  if hash_routed {
    append_operation(
      operations,
      scope,
      TokenHashRouting,
      RequiresCheckedInt64ToInt32TokenHash,
    )
  }
  append_numeric(
    operations,
    scope,
    StableTopKRouting,
    f32_activation_execution(),
  )
  append_operation(
    operations,
    scope,
    RoutedExpertClampedSwiGlu,
    if fp4_routed {
      UnsupportedPackedFp4E2M1
    } else {
      NumericContract(fp8_gated_execution())
    },
  )
  append_numeric(
    operations,
    scope,
    SharedExpertClampedSwiGlu,
    fp8_gated_execution(),
  )
  append_numeric(
    operations,
    scope,
    RoutedSharedExpertCombine,
    bf16_f32_control_execution(),
  )
}

///|
fn append_decoder_block(
  operations : Array[DeepSeekV4NumericOperationBinding],
  scope : DeepSeekV4NumericOperationScope,
  attention : @advanced_decoder_execution_plan.AdvancedAttentionPlan,
  hash_routed : Bool,
  fp4_routed : Bool,
) -> Unit raise DeepSeekV4NumericPlanError {
  append_numeric(operations, scope, BlockRmsNorm, bf16_tensor_execution())
  append_hyper_connection(operations, scope, AttentionSite)
  append_attention(operations, scope, attention)
  append_numeric(operations, scope, BlockRmsNorm, bf16_tensor_execution())
  append_hyper_connection(operations, scope, FeedForwardSite)
  append_feed_forward(operations, scope, hash_routed, fp4_routed)
}

///|
fn auxiliary_attention(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
  stage : Int,
) -> @advanced_decoder_execution_plan.AdvancedAttentionPlan raise DeepSeekV4NumericPlanError {
  guard spec.compression_ratio(spec.layer_count() + stage) is Some(ratio) else {
    raise ProfilePlanMismatch
  }
  match ratio {
    0 =>
      try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::sliding_window_causal(
        spec.sliding_window(),
      )
    4 =>
      try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::indexed_compressed(
        compression_ratio=4,
        index_head_count=spec.index_head_count(),
        index_head_dimension=spec.index_head_dimension(),
        index_top_k=spec.index_top_k(),
      )
    value =>
      try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::deterministic_compressed(
        compression_ratio=value,
        window_size=spec.sliding_window(),
      )
  }
}

///|
fn build_operations(
  plan : @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan,
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> Array[DeepSeekV4NumericOperationBinding] raise DeepSeekV4NumericPlanError {
  let operations : Array[DeepSeekV4NumericOperationBinding] = []
  append_numeric(operations, Global, TokenEmbedding, bf16_parameter_execution())
  let fp4_routed = spec.expert_storage() is Fp4E2M1
  for layer in plan.layers() {
    append_decoder_block(
      operations,
      DecoderLayer(layer.layer_index()),
      layer.attention(),
      layer.layer_index() < spec.hash_layer_count(),
      fp4_routed,
    )
  }
  match plan.auxiliary_prediction() {
    MultiTokenPrediction(layer_count~) => {
      append_numeric(
        operations,
        PredictionStage(0),
        MultiTokenPredictionInputProjection,
        fp8_projection_execution(),
      )
      for stage in 0..<layer_count {
        append_decoder_block(
          operations,
          PredictionStage(stage),
          auxiliary_attention(spec, stage),
          false,
          fp4_routed,
        )
      }
      append_numeric(
        operations,
        PredictionStage(layer_count - 1),
        HyperConnectionHeadReduce,
        bf16_f32_control_execution(),
      )
      append_numeric(
        operations,
        PredictionStage(layer_count - 1),
        AuxiliaryOutputHead,
        bf16_tensor_execution(),
      )
    }
    MultiTokenPredictionWithDspark(layer_count~, dspark~) => {
      for _ in dspark.target_layer_ids() {
        append_numeric(
          operations,
          PredictionStage(0),
          DsparkTargetLayerRead,
          bf16_activation_execution(),
        )
      }
      append_numeric(
        operations,
        PredictionStage(0),
        DsparkMainProjection,
        fp8_projection_execution(),
      )
      append_numeric(
        operations,
        PredictionStage(0),
        DsparkNoiseEmbedding,
        bf16_parameter_execution(),
      )
      for stage in 0..<layer_count {
        append_decoder_block(
          operations,
          PredictionStage(stage),
          auxiliary_attention(spec, stage),
          false,
          fp4_routed,
        )
      }
      let final_scope = PredictionStage(layer_count - 1)
      append_numeric(
        operations,
        final_scope,
        HyperConnectionHeadReduce,
        bf16_f32_control_execution(),
      )
      append_numeric(
        operations,
        final_scope,
        AuxiliaryOutputHead,
        bf16_tensor_execution(),
      )
      append_numeric(
        operations,
        final_scope,
        DsparkMarkovAdjustment,
        bf16_tensor_execution(),
      )
      append_numeric(
        operations,
        final_scope,
        DsparkConfidence,
        bf16_tensor_execution(),
      )
    }
  }
  append_numeric(
    operations,
    Global,
    HyperConnectionHeadReduce,
    bf16_f32_control_execution(),
  )
  append_numeric(operations, Global, FinalRmsNorm, bf16_tensor_execution())
  append_numeric(operations, Global, LanguageModelHead, bf16_tensor_execution())
  operations
}
