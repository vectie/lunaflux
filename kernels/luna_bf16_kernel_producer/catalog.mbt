///|
fn operation_semantics(
  operation : @plan.PlanOperation,
  layout : @device_layout.DeviceKvLayout,
) -> @catalog.KernelExecutionSemantics raise LunaBf16KernelProducerError {
  let kv_layout = match operation.persistent_state() {
    NoPersistentState => None
    KvCacheReadWrite(_) =>
      Some(
        @catalog.PagedKvLayoutContract::new(
          layout=layout.geometry().layout(),
          stable_version=layout.stable_layout_version(),
          tokens_per_page=layout.tokens_per_page(),
        ) catch {
          _ => raise Invalid(Catalog)
        },
      )
  }
  @catalog.KernelExecutionSemantics::paged_v3(
    kv_layer=operation.kv_layer(),
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout~,
  ) catch {
    _ => raise Invalid(Catalog)
  }
}

///|
fn candidate_workspace_alignment(
  operation : @plan.PlanOperation,
  operands : ReadOnlyArray[@launch_contract.AotOperand],
) -> Int64 raise LunaBf16KernelProducerError {
  if operation.workspace_bytes() == 0L {
    for operand in operands {
      if operand.role() == Workspace {
        raise Mismatch(Workspace)
      }
    }
    return 1L
  }
  let mut found : Int64? = None
  for operand in operands {
    if operand.role() == Workspace {
      if found is Some(_) || operand.byte_count() != operation.workspace_bytes() {
        raise Mismatch(Workspace)
      }
      found = Some(operand.alignment())
    }
  }
  guard found is Some(alignment) else { raise Mismatch(Workspace) }
  alignment
}

///|
fn reusable_key_matches(
  binding : ReusableFamilyBinding,
  operation : @plan.PlanOperation,
  semantics : @catalog.KernelExecutionSemantics,
  workspace_alignment : Int64,
) -> Bool {
  binding.operation_kind == operation.kind() &&
  binding.operation_shape == operation.shape() &&
  binding.capability == operation.required_capability() &&
  binding.semantics == semantics &&
  binding.workspace_bytes == operation.workspace_bytes() &&
  binding.workspace_alignment == workspace_alignment
}

///|
fn reusable_binding_index(
  bindings : ArrayView[ReusableFamilyBinding],
  operation : @plan.PlanOperation,
  semantics : @catalog.KernelExecutionSemantics,
  workspace_alignment : Int64,
) -> Int? {
  for index, binding in bindings {
    if reusable_key_matches(binding, operation, semantics, workspace_alignment) {
      return Some(index)
    }
  }
  None
}

///|
fn compilation_index(
  compilations : ArrayView[LunaBf16OfflineCompilation],
  operation_id : @plan.OperationId,
) -> Int? raise LunaBf16KernelProducerError {
  let mut found : Int? = None
  for index, compilation in compilations {
    if candidate_operation_id(compilation.candidate) == operation_id {
      if found is Some(_) {
        raise Invalid(OperationCoverage)
      }
      found = Some(index)
    }
  }
  found
}

///|
fn require_complete_kinds(
  operations : ReadOnlyArray[@plan.PlanOperation],
) -> Unit raise LunaBf16KernelProducerError {
  let seen = FixedArray::make(9, false)
  for operation in operations {
    let index : Int? = match operation.kind() {
      Embedding => Some(0)
      RmsNorm => Some(1)
      QkvProjection => Some(2)
      QkRmsNorm => None
      RotaryEmbedding => Some(3)
      CausalAttention => Some(4)
      OutputProjection => Some(5)
      ResidualAdd => Some(6)
      GatedMlp => Some(7)
      LanguageModelHead => Some(8)
    }
    match index {
      Some(value) => seen[value] = true
      None => ()
    }
  }
  for value in seen {
    if !value {
      raise Invalid(CompleteGraph)
    }
  }
}
