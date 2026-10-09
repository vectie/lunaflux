///|
fn append_dsa(
  bindings : Array[Glm53NumericOperationBinding],
  schemas : Array[@numeric_contract.NumericOperationSchema],
  scope : Glm53NumericOperationScope,
  action : @glm_hybrid_execution_plan.GlmSparseIndexAction,
  pooled : Bool,
) -> Unit {
  append_operation(
    bindings,
    schemas,
    scope,
    DsaLowRankProjection,
    bf16_tensor_execution(),
  )
  match action {
    ComputeTopK => {
      append_operation(
        bindings,
        schemas,
        scope,
        DsaIndexerProjection,
        bf16_tensor_execution(),
      )
      if pooled {
        append_operation(
          bindings,
          schemas,
          scope,
          DsaIndexPooling,
          bf16_tensor_execution(),
        )
      }
      append_operation(
        bindings,
        schemas,
        scope,
        DsaIndexSelection,
        bf16_activation_execution(),
      )
    }
    ReusePreviousTopK =>
      append_operation(
        bindings,
        schemas,
        scope,
        DsaIndexReuse,
        bf16_activation_execution(),
      )
  }
  append_operation(
    bindings,
    schemas,
    scope,
    DsaSparseAttention,
    bf16_activation_execution(),
  )
  append_operation(
    bindings,
    schemas,
    scope,
    AttentionOutputProjection,
    bf16_tensor_execution(),
  )
}

///|
fn append_kda(
  bindings : Array[Glm53NumericOperationBinding],
  schemas : Array[@numeric_contract.NumericOperationSchema],
  scope : Glm53NumericOperationScope,
) -> Unit {
  append_operation(
    bindings,
    schemas,
    scope,
    KdaProjection,
    bf16_tensor_execution(),
  )
  append_operation(
    bindings,
    schemas,
    scope,
    KdaShortConvolution,
    bf16_tensor_execution(),
  )
  append_operation(
    bindings,
    schemas,
    scope,
    KdaF32DecayControl,
    bf16_f32_to_f32_tensor_execution(),
  )
  append_operation(
    bindings,
    schemas,
    scope,
    KdaRecurrentUpdate,
    bf16_f32_activation_execution(),
  )
  append_operation(
    bindings,
    schemas,
    scope,
    KdaOutputGateProjection,
    bf16_tensor_execution(),
  )
  append_operation(
    bindings,
    schemas,
    scope,
    KdaGatedRmsNorm,
    bf16_f32_tensor_execution(),
  )
  append_operation(
    bindings,
    schemas,
    scope,
    KdaOutputProjection,
    bf16_tensor_execution(),
  )
}
