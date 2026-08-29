///|
fn paged_kv_write_checked_mul(
  left : Int64,
  right : Int64,
  rule : PagedKvWriteAotRule,
) -> Int64 raise PagedKvWriteAotError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise SizeOverflow(rule)
  }
  left * right
}

///|
fn paged_kv_write_expected_roles(
  input : PositionedQkvActivation,
  layer : @plan.KvLayerId,
) -> Array[@launch_contract.AotOperandRole] {
  [
    RuntimeInput(StepCounts),
    RuntimeInput(QueryPositions),
    RuntimeInput(QueryRowOffsets),
    RuntimeInput(SequenceLengths),
    RuntimeInput(PageTableOffsets),
    RuntimeInput(PageTablePageIndices),
    ActivationInput(input.qkv),
    PersistentKvComponent(layer, Key),
    PersistentKvComponent(layer, Value),
  ]
}

///|
fn paged_kv_write_validate_operands(
  operands : ArrayView[@launch_contract.AotOperand],
  roles : ArrayView[@launch_contract.AotOperandRole],
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  qkv_width : Int,
) -> Unit raise PagedKvWriteAotError {
  if operands.length() != roles.length() {
    raise Mismatch(OperandOrder)
  }
  let rows = profile.max_query_rows().to_int64()
  let tokens = profile.max_query_tokens().to_int64()
  let pages = profile.max_page_table_entries().to_int64()
  let qkv_bytes = paged_kv_write_checked_mul(
    paged_kv_write_checked_mul(tokens, qkv_width.to_int64(), OperandSize),
    2L,
    OperandSize,
  )
  let sizes : Array[Int64] = [
    20L,
    paged_kv_write_checked_mul(tokens, 4L, OperandSize),
    paged_kv_write_checked_mul(rows + 1L, 4L, OperandSize),
    paged_kv_write_checked_mul(rows, 4L, OperandSize),
    paged_kv_write_checked_mul(rows + 1L, 4L, OperandSize),
    paged_kv_write_checked_mul(pages, 4L, OperandSize),
    qkv_bytes,
    layout.component_stride_bytes(),
    layout.component_stride_bytes(),
  ]
  for index, operand in operands {
    if operand.role() != roles[index] {
      raise Mismatch(OperandOrder)
    }
    if operand.byte_count() != sizes[index] {
      raise Mismatch(OperandSize)
    }
    let alignment = if index < 6 {
      4L
    } else if index == 6 {
      2L
    } else {
      layout.segment_alignment_bytes()
    }
    if operand.alignment() < alignment || operand.alignment() % alignment != 0L {
      raise Mismatch(OperandAlignment)
    }
  }
}

///|
fn paged_kv_write_validate_positioned_input(
  plan : @plan.ModelPlan,
  attention : @plan.PlanOperation,
  input : PositionedQkvActivation,
) -> (Int, Int, Int, @plan.KvLayerId) raise PagedKvWriteAotError {
  let operation_index = attention.id().as_int()
  guard operation_index > 0 &&
    plan.operations().get(operation_index) is Some(bound_attention) &&
    bound_attention == attention &&
    plan.operations().get(operation_index - 1) is Some(rotary) &&
    rotary.kind() == RotaryEmbedding &&
    rotary.id().as_int() + 1 == attention.id().as_int() &&
    rotary.outputs().length() == 1 &&
    rotary.outputs()[0] == input.qkv &&
    attention.inputs().length() == 1 &&
    attention.inputs()[0] == Activation(input.qkv) &&
    attention.shape()
    is AttentionShape(query_heads~, kv_heads~, head_dimension~) &&
    rotary.shape()
    is RotaryShape(
      query_heads=rotary_query_heads,
      kv_heads=rotary_kv_heads,
      head_dimension=rotary_head_dimension,
      ..
    ) &&
    rotary_query_heads == query_heads &&
    rotary_kv_heads == kv_heads &&
    rotary_head_dimension == head_dimension &&
    attention.kv_layer() is Some(layer) &&
    rotary.kv_layer() == Some(layer) &&
    attention.persistent_state() == KvCacheReadWrite(layer) &&
    rotary.persistent_state() == NoPersistentState &&
    rotary.required_capability() ==
    @plan.KernelCapabilityId::positioned_rotary_embedding() else {
    raise Mismatch(Identity)
  }
  (query_heads, kv_heads, head_dimension, layer)
}

///|
fn paged_kv_write_validate(
  plan : @plan.ModelPlan,
  attention : @plan.PlanOperation,
  input : PositionedQkvActivation,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  fallback : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
) -> (Int, Int, Int, @plan.KvLayerId) raise PagedKvWriteAotError {
  guard attention.kind() == CausalAttention else { raise Mismatch(Operation) }
  let (query_heads, kv_heads, head_dimension, layer) = paged_kv_write_validate_positioned_input(
    plan, attention, input,
  )
  let kv_width = kv_heads * head_dimension
  let qkv_width = (query_heads + 2 * kv_heads) * head_dimension
  if query_heads <= 0 ||
    kv_heads <= 0 ||
    head_dimension <= 0 ||
    query_heads % kv_heads != 0 ||
    kv_width <= 0 ||
    kv_width > 256 ||
    qkv_width <= 0 {
    raise Unsupported(Shape)
  }
  if profile.max_query_rows() <= 0 ||
    profile.max_query_tokens() < profile.max_query_rows() ||
    profile.max_page_table_entries() < profile.max_query_rows() {
    raise Unsupported(Profile)
  }
  if !layout.is_canonical_for(plan.identity(), plan.kv_cache_geometry()) ||
    layout.stable_layout_version() != 1 ||
    layout.geometry().dtype() != BFloat16 ||
    layout.geometry().layout() != LayerMajorSplitKeyValueV1 ||
    layout.geometry().kv_head_count() != kv_heads ||
    layout.geometry().head_dimension() != head_dimension ||
    layout.tokens_per_page() <= 0 ||
    layout.total_page_count() <= 0 {
    raise Mismatch(Layout)
  }
  if !target.supports_bf16() || target.compute_major() < 8 {
    raise Unsupported(Target)
  }
  if compiler.optimization_level() != 3 ||
    compiler.fused_multiply_add() ||
    compiler.reassociate() ||
    compiler.max_registers() < 64 ||
    compiler.max_registers() > 255 {
    raise Unsupported(Compiler)
  }
  if fallback.operation_id() != attention.id() ||
    fallback.target() != target ||
    fallback.profile() != profile ||
    fallback.compiler_policy() != compiler ||
    fallback.query_heads() != query_heads ||
    fallback.kv_heads() != kv_heads ||
    fallback.head_dimension() != head_dimension ||
    fallback.tokens_per_page() != layout.tokens_per_page() ||
    fallback.page_stride_bytes() != layout.page_stride_bytes() ||
    fallback.physical_page_capacity() != layout.total_page_count().to_int64() {
    raise Mismatch(Fallback)
  }
  paged_kv_write_validate_operands(
    operands,
    paged_kv_write_expected_roles(input, layer),
    profile,
    layout,
    qkv_width,
  )
  (query_heads, kv_heads, head_dimension, layer)
}
