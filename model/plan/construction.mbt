///|
/// Validates and owns an explicitly stateless ordered operation graph.
///
/// This constructor is for attention-free plans and bounded semantic
/// subgraphs. Complete decoder plans, including stateless full-context plans,
/// use `new_with_kv_execution` so layer coverage is explicit and validated.
pub fn ModelPlan::new_stateless(
  content_digest~ : @spec.ContentDigest,
  limits~ : ModelPlanLimits,
  shape_constraints~ : ShapeConstraints,
  kv_cache_geometry~ : KvCacheGeometry,
  workspace_bounds~ : WorkspaceBounds,
  operations~ : ArrayView[PlanOperation],
  final_output~ : ActivationRef,
  numeric_schema~ : @numeric_contract.ModelNumericSchema,
) -> ModelPlan raise PlanValidationError {
  limits.check_operation_count(operations.length())
  for operation in operations {
    if operation.kind() is (RotaryEmbedding | CausalAttention) ||
      operation.kv_layer() is Some(_) {
      raise InvalidGraph(KvExecutionContract)
    }
  }
  construct_model_plan(
    content_digest~,
    limits~,
    shape_constraints~,
    kv_cache_geometry~,
    workspace_bounds~,
    operations~,
    final_output~,
    numeric_schema~,
    kv_execution=StatelessFullContext,
    require_kv_coverage=false,
  )
}

///|
/// Validates and owns a model plan with an explicit full-context or paged-KV
/// semantic contract. This selects no device implementation or kernel.
pub fn ModelPlan::new_with_kv_execution(
  content_digest~ : @spec.ContentDigest,
  limits~ : ModelPlanLimits,
  shape_constraints~ : ShapeConstraints,
  kv_cache_geometry~ : KvCacheGeometry,
  workspace_bounds~ : WorkspaceBounds,
  operations~ : ArrayView[PlanOperation],
  final_output~ : ActivationRef,
  kv_execution~ : ModelKvExecution,
  numeric_schema~ : @numeric_contract.ModelNumericSchema,
) -> ModelPlan raise PlanValidationError {
  construct_model_plan(
    content_digest~,
    limits~,
    shape_constraints~,
    kv_cache_geometry~,
    workspace_bounds~,
    operations~,
    final_output~,
    numeric_schema~,
    kv_execution~,
    require_kv_coverage=true,
  )
}

///|
fn construct_model_plan(
  content_digest~ : @spec.ContentDigest,
  limits~ : ModelPlanLimits,
  shape_constraints~ : ShapeConstraints,
  kv_cache_geometry~ : KvCacheGeometry,
  workspace_bounds~ : WorkspaceBounds,
  operations~ : ArrayView[PlanOperation],
  final_output~ : ActivationRef,
  numeric_schema~ : @numeric_contract.ModelNumericSchema,
  kv_execution~ : ModelKvExecution,
  require_kv_coverage~ : Bool,
) -> ModelPlan raise PlanValidationError {
  if kv_cache_geometry.kv_head_count() != shape_constraints.kv_head_count() ||
    kv_cache_geometry.head_dimension() != shape_constraints.head_dimension() {
    raise InvalidShape(KvCacheShapeMismatch)
  }
  if operations.is_empty() {
    raise InvalidGraph(EmptyOperationList)
  }
  let canonical_bytes = preflight_model_plan(
    limits, shape_constraints, kv_cache_geometry, workspace_bounds, operations, final_output,
    kv_execution, numeric_schema,
  )
  let numeric_binding = map_numeric_schema(numeric_schema)
  if numeric_binding.operations().length() != operations.length() {
    raise InvalidGraph(NumericOperationBinding)
  }
  let mut compound_activation_scale_plan = false
  for binding in numeric_binding.operations() {
    if binding.execution().activation_scale() ==
      @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1() ||
      binding.execution().activation_scale() ==
      @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2() {
      compound_activation_scale_plan = true
    }
  }
  let produced_numeric : Array[@numeric_contract.ComputeDType] = []
  let required : Array[KernelCapabilityId] = []
  let seen_capabilities = FixedArray::make(11, false)
  let used_parameters : Array[Bool] = []
  for _ in 0..<numeric_binding.tensors().length() {
    used_parameters.push(false)
  }
  let mut next_kv_layer = 0
  for index, operation in operations {
    validate_operation(
      operation, index, shape_constraints, workspace_bounds, next_kv_layer,
    )
    let operation_numeric = numeric_binding.operations()[index]
    if operation_numeric.operation() != operation.id() {
      raise InvalidGraph(NumericOperationBinding)
    }
    for input in operation.inputs() {
      match input {
        Activation(reference) =>
          if reference.as_int() >= produced_numeric.length() {
            raise InvalidGraph(MissingActivationInput)
          }
        ModelInput(_) | Tensor(_) => ()
      }
    }
    validate_operation_numeric(
      operation,
      operation_numeric.execution(),
      numeric_binding.tensors(),
      produced_numeric,
      shape_constraints,
      used_parameters,
      compound_activation_scale_plan,
    )
    if operation.kind() == CausalAttention {
      next_kv_layer += 1
    }
    for output in operation.outputs() {
      if output.as_int() != produced_numeric.length() {
        raise InvalidGraph(ActivationOutputOrder)
      }
      produced_numeric.push(operation_numeric.execution().output())
    }
    let capability = operation.required_capability()
    let capability_value = capability.as_int()
    if !seen_capabilities[capability_value] {
      seen_capabilities[capability_value] = true
      required.push(capability)
    }
  }
  if final_output.as_int() != produced_numeric.length() - 1 {
    raise InvalidGraph(FinalOutputNotLast)
  }
  // Stateless construction permits attention-free operation subgraphs.
  // Explicit execution-mode construction always owns every semantic KV layer.
  if (require_kv_coverage || next_kv_layer != 0) &&
    next_kv_layer != kv_cache_geometry.layer_count() {
    raise InvalidGraph(KvLayerCoverage)
  }
  let owned_operations = ReadOnlyArray::from_array(operations)
  validate_kv_execution_mode(kv_execution, owned_operations)
  validate_complete_numeric_tensor_use(
    numeric_binding.tensors(),
    used_parameters,
  )
  let identity = derive_model_identity(
    content_digest,
    canonical_bytes,
    shape_constraints,
    kv_cache_geometry,
    workspace_bounds,
    owned_operations,
    required,
    final_output,
    kv_execution,
    numeric_binding.digest(),
  )
  {
    identity,
    shape_constraints,
    kv_cache_geometry,
    workspace_bounds,
    operations: owned_operations,
    required_capabilities: ReadOnlyArray::from_array(required),
    final_output,
    kv_execution,
    numeric_binding,
  }
}
