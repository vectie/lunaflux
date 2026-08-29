///|
fn kv_write_parameters(
  operation : @plan.PlanOperation,
  layout : @device_layout.DeviceKvLayout,
) -> PointwiseParameters raise PointwiseAotError {
  guard operation.shape()
    is AttentionShape(query_heads~, kv_heads~, head_dimension~) else {
    raise Unsupported(Operation)
  }
  if operation.kind() != CausalAttention ||
    operation.required_capability() !=
    @plan.KernelCapabilityId::paged_causal_attention() ||
    operation.workspace_bytes() != 0L ||
    operation.inputs().length() != 1 ||
    !(operation.inputs()[0] is Activation(_)) ||
    operation.outputs().length() != 1 ||
    !exact_runtime_inputs(operation.runtime_inputs(), [
      StepCounts,
      QueryPositions,
      QueryRowOffsets,
      SequenceLengths,
      PageTableOffsets,
      PageTablePageIndices,
    ]) {
    raise Mismatch(Operation)
  }
  let width = qkv_width(query_heads, kv_heads, head_dimension)
  guard operation.kv_layer() is Some(layer) else {
    raise Mismatch(PersistentState)
  }
  guard operation.persistent_state() is KvCacheReadWrite(persistent_layer) else {
    raise Mismatch(PersistentState)
  }
  if layer != persistent_layer ||
    layout.stable_layout_version() != 1 ||
    layout.geometry().dtype() != BFloat16 ||
    layout.geometry().layout() != LayerMajorSplitKeyValueV1 ||
    layout.geometry().kv_head_count() != kv_heads ||
    layout.geometry().head_dimension() != head_dimension ||
    layer.as_int() >= layout.geometry().layer_count() ||
    layout.page_stride_bytes() % 2L != 0L {
    raise Mismatch(KvLayout)
  }
  {
    family: KvCacheWrite,
    width,
    vocabulary_size: 0,
    query_heads,
    kv_heads,
    head_dimension,
    epsilon: 0.0,
    rope_theta: 0.0,
    tokens_per_page: layout.tokens_per_page(),
    total_page_count: layout.total_page_count(),
    page_stride_bytes: layout.page_stride_bytes(),
  }
}

///|
fn kv_write_roles(
  operation : @plan.PlanOperation,
) -> Array[@launch_contract.AotOperandRole] raise PointwiseAotError {
  guard operation.inputs()[0] is Activation(input) else {
    raise Mismatch(Operation)
  }
  guard operation.kv_layer() is Some(layer) else { raise Mismatch(Operation) }
  [
    RuntimeInput(StepCounts),
    RuntimeInput(QueryPositions),
    RuntimeInput(QueryRowOffsets),
    RuntimeInput(SequenceLengths),
    RuntimeInput(PageTableOffsets),
    RuntimeInput(PageTablePageIndices),
    ActivationInput(input),
    PersistentKvComponent(layer, Key),
    PersistentKvComponent(layer, Value),
  ]
}

///|
fn validate_kv_write_operands(
  operation : @plan.PlanOperation,
  parameters : PointwiseParameters,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  operands : ArrayView[@launch_contract.AotOperand],
) -> Unit raise PointwiseAotError {
  let roles = kv_write_roles(operation)
  if operands.length() != roles.length() {
    raise Mismatch(OperandOrder)
  }
  for index, operand in operands {
    let role = roles[index]
    if operand.role() != role {
      raise Mismatch(OperandOrder)
    }
    let expected_bytes = match role {
      RuntimeInput(runtime) => runtime_bytes(runtime, profile)
      ActivationInput(_) =>
        bf16_matrix_bytes(profile.max_query_tokens(), parameters.width)
      PersistentKvComponent(_, _) => layout.component_stride_bytes()
      _ => raise Mismatch(OperandOrder)
    }
    if operand.byte_count() != expected_bytes {
      raise Mismatch(OperandSize)
    }
    let minimum_alignment = match role {
      RuntimeInput(_) => INT32_BYTES
      PersistentKvComponent(_, _) => layout.segment_alignment_bytes()
      _ => BF16_BYTES
    }
    if operand.alignment() < minimum_alignment ||
      operand.alignment() % minimum_alignment != 0L {
      raise Mismatch(OperandAlignment)
    }
  }
}
