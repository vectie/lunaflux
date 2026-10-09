///|
fn append_decoder_layers(
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
  bindings : Array[Glm53NumericOperationBinding],
  schemas : Array[@numeric_contract.NumericOperationSchema],
) -> Unit {
  let pooled = plan.sparse_attention().index_pooling() is Some(_)
  let has_hyper = plan.hyper_connection() is Some(_)
  for layer in plan.layers() {
    let scope = DecoderLayer(layer_index=layer.layer_index())
    append_layer_norm(bindings, schemas, scope)
    if has_hyper {
      append_hyper_connection(bindings, schemas, scope)
    }
    match layer.attention() {
      DynamicSparse(action) =>
        append_dsa(bindings, schemas, scope, action, pooled)
      KimiDelta => append_kda(bindings, schemas, scope)
    }
    append_layer_norm(bindings, schemas, scope)
    if has_hyper {
      append_hyper_connection(bindings, schemas, scope)
    }
    append_feed_forward(bindings, schemas, scope, layer.feed_forward())
  }
}

///|
fn append_auxiliary_layer(
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
  bindings : Array[Glm53NumericOperationBinding],
  schemas : Array[@numeric_contract.NumericOperationSchema],
) -> Unit raise Glm53NumericPlanError {
  let stage_count = match plan.auxiliary_prediction() {
    MultiTokenPrediction(layer_count~) => layer_count
    MultiTokenPredictionWithDspark(layer_count~, ..) => layer_count
  }
  if stage_count != 1 {
    raise ProfilePlanMismatch
  }
  let scope = AuxiliaryLayer(stage_index=0)
  append_operation(
    bindings,
    schemas,
    scope,
    MtpFusionProjection,
    bf16_tensor_execution(),
  )
  append_layer_norm(bindings, schemas, scope)
  append_dsa(
    bindings,
    schemas,
    scope,
    ComputeTopK,
    plan.sparse_attention().index_pooling() is Some(_),
  )
  append_layer_norm(bindings, schemas, scope)
  append_feed_forward(bindings, schemas, scope, RoutedMoe)
  append_operation(
    bindings,
    schemas,
    scope,
    SharedHeadNorm,
    bf16_tensor_execution(),
  )
}

///|
fn append_vision(
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
  bindings : Array[Glm53NumericOperationBinding],
  schemas : Array[@numeric_contract.NumericOperationSchema],
) -> Unit {
  guard plan.vision_conditioning() is Some(vision) else { return }
  append_operation(
    bindings,
    schemas,
    VisionFrontend,
    VisionPatchProjection,
    bf16_tensor_execution(),
  )
  for layer in 0..<vision.layer_count() {
    let scope = VisionLayer(layer_index=layer)
    append_layer_norm(bindings, schemas, scope)
    append_operation(
      bindings,
      schemas,
      scope,
      VisionQkvProjection,
      bf16_tensor_execution(),
    )
    append_operation(
      bindings,
      schemas,
      scope,
      VisionAttention,
      bf16_activation_execution(),
    )
    append_operation(
      bindings,
      schemas,
      scope,
      VisionOutputProjection,
      bf16_tensor_execution(),
    )
    append_layer_norm(bindings, schemas, scope)
    append_operation(
      bindings,
      schemas,
      scope,
      VisionSwiGlu,
      bf16_tensor_execution(),
    )
  }
  append_layer_norm(bindings, schemas, VisionProjector)
  append_operation(
    bindings,
    schemas,
    VisionProjector,
    VisionSpatialMerge,
    bf16_tensor_execution(),
  )
  append_operation(
    bindings,
    schemas,
    VisionProjector,
    VisionSwiGlu,
    bf16_tensor_execution(),
  )
  append_operation(
    bindings,
    schemas,
    VisionProjector,
    VisionProjector,
    bf16_tensor_execution(),
  )
  append_layer_norm(bindings, schemas, VisionProjector)
}

///|
fn build_operation_schemas(
  plan : @glm_hybrid_execution_plan.GlmHybridExecutionPlan,
) -> (
  Array[@numeric_contract.NumericOperationSchema],
  Array[Glm53NumericOperationBinding],
) raise Glm53NumericPlanError {
  let schemas : Array[@numeric_contract.NumericOperationSchema] = []
  let bindings : Array[Glm53NumericOperationBinding] = []
  append_operation(
    bindings,
    schemas,
    Global,
    TokenEmbedding,
    bf16_parameter_execution(),
  )
  append_decoder_layers(plan, bindings, schemas)
  append_auxiliary_layer(plan, bindings, schemas)
  append_operation(
    bindings,
    schemas,
    Global,
    FinalNorm,
    bf16_tensor_execution(),
  )
  append_operation(
    bindings,
    schemas,
    Global,
    LanguageModelHead,
    bf16_tensor_execution(),
  )
  append_vision(plan, bindings, schemas)
  (schemas, bindings)
}
