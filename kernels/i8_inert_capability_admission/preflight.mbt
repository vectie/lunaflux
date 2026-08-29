///|
priv struct I8AdmissionPreflight {
  operation_count : Int
  weight_count : Int
  canonical_bytes : Int
}

///|
fn preflight_i8_admission(
  plan : @plan.ModelPlan,
  resolved_v4 : @catalog.ResolvedKernelCatalog,
  limits : I8InertKernelCapabilityAdmissionLimits,
) -> I8AdmissionPreflight raise I8InertKernelCapabilityAdmissionError {
  let mut operation_count = 0
  let mut weight_count = 0
  let mut max_workspace_bytes = 0L
  let mut max_workspace_alignment = 1L
  for index, operation in plan.operations() {
    let numeric = plan.numeric_binding().operations()[index]
    if numeric.operation() != operation.id() {
      raise I8InertKernelCapabilityRejected(BindingOrder)
    }
    let execution = numeric.execution()
    let binding = resolved_v4.bindings()[index]
    ignore(validate_i8_catalog_binding(operation, execution, binding))
    if binding.workspace_bytes() > max_workspace_bytes {
      max_workspace_bytes = binding.workspace_bytes()
    }
    if binding.workspace_alignment() > max_workspace_alignment {
      max_workspace_alignment = binding.workspace_alignment()
    }
    let exact_i8 = exact_i8_weight_only_execution(execution)
    let mentions_i8 = execution.tensor_input() ==
      @numeric_contract.ComputeDType::i8() ||
      operation_has_i8_tensor(plan, operation)
    if mentions_i8 && !exact_i8 {
      raise I8InertKernelCapabilityRejected(OperationExecution)
    }
    if exact_i8 {
      if !i8_operation_kind_allowed(operation.kind()) {
        raise I8InertKernelCapabilityRejected(UnsupportedI8Operation)
      }
      if operation_count >= limits.max_operations {
        raise I8InertKernelCapabilityRejected(Limit)
      }
      let selected_weights = count_i8_weight_references(plan, operation)
      if selected_weights > limits.max_weights - weight_count {
        raise I8InertKernelCapabilityRejected(Limit)
      }
      operation_count += 1
      weight_count += selected_weights
    }
  }
  if operation_count == 0 {
    raise I8InertKernelCapabilityRejected(MissingI8Operation)
  }
  if max_workspace_bytes != resolved_v4.max_workspace_bytes() ||
    max_workspace_alignment != resolved_v4.max_workspace_alignment() {
    raise I8InertKernelCapabilityRejected(Workspace)
  }
  guard i8_admission_v1_size_within(
      operation_count,
      weight_count,
      limits.max_canonical_bytes,
    )
    is Some(canonical_bytes) else {
    raise I8InertKernelCapabilityRejected(CanonicalBytes)
  }
  { operation_count, weight_count, canonical_bytes }
}

///|
fn validate_i8_scale_uniqueness_for_plan(
  plan : @plan.ModelPlan,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  let scale_seen : Array[Bool] = []
  for _ in 0..<plan.numeric_binding().tensors().length() {
    scale_seen.push(false)
  }
  for index, operation in plan.operations() {
    if exact_i8_weight_only_execution(
        plan.numeric_binding().operations()[index].execution(),
      ) {
      validate_i8_scale_uniqueness(plan, operation, scale_seen)
    }
  }
}
