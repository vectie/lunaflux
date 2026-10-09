///|
fn append_operation(
  bindings : Array[Glm53NumericOperationBinding],
  schemas : Array[@numeric_contract.NumericOperationSchema],
  scope : Glm53NumericOperationScope,
  kind : Glm53NumericOperationKind,
  execution : @numeric_contract.OperationExecutionContract,
) -> Unit {
  let ordinal = bindings.length()
  bindings.push({ ordinal, scope, kind, execution, })
  schemas.push(
    @numeric_contract.NumericOperationSchema::new(
      ordinal=try! @numeric_contract.NumericOperationOrdinal::new(ordinal),
      execution~,
    ),
  )
}

///|
fn append_layer_norm(
  bindings : Array[Glm53NumericOperationBinding],
  schemas : Array[@numeric_contract.NumericOperationSchema],
  scope : Glm53NumericOperationScope,
) -> Unit {
  append_operation(bindings, schemas, scope, LayerNorm, bf16_tensor_execution())
}

///|
fn append_hyper_connection(
  bindings : Array[Glm53NumericOperationBinding],
  schemas : Array[@numeric_contract.NumericOperationSchema],
  scope : Glm53NumericOperationScope,
) -> Unit {
  append_operation(
    bindings,
    schemas,
    scope,
    HyperConnectionBf16Function,
    bf16_f32_tensor_execution(),
  )
  append_operation(
    bindings,
    schemas,
    scope,
    HyperConnectionF32Control,
    f32_tensor_execution(),
  )
}

///|
fn append_feed_forward(
  bindings : Array[Glm53NumericOperationBinding],
  schemas : Array[@numeric_contract.NumericOperationSchema],
  scope : Glm53NumericOperationScope,
  feed_forward : @glm_hybrid_execution_plan.GlmLayerFeedForwardPlan,
) -> Unit {
  match feed_forward {
    DenseSwiGlu =>
      append_operation(
        bindings,
        schemas,
        scope,
        DenseSwiGlu,
        bf16_tensor_execution(),
      )
    RoutedMoe => {
      append_operation(
        bindings,
        schemas,
        scope,
        MoeRouterProjection,
        f32_router_projection_execution(),
      )
      append_operation(
        bindings,
        schemas,
        scope,
        MoeF32SigmoidAndCorrection,
        f32_tensor_execution(),
      )
      append_operation(
        bindings,
        schemas,
        scope,
        MoeF32GroupedRouting,
        f32_tensor_execution(),
      )
      append_operation(
        bindings,
        schemas,
        scope,
        RoutedExpertSwiGlu,
        bf16_tensor_execution(),
      )
      append_operation(
        bindings,
        schemas,
        scope,
        SharedExpertSwiGlu,
        bf16_tensor_execution(),
      )
      append_operation(
        bindings,
        schemas,
        scope,
        RoutedSharedExpertBf16Combine,
        bf16_tensor_execution(),
      )
    }
  }
}
