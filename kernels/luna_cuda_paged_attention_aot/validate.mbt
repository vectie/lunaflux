///|
const PAGED_ATTENTION_BF16_BYTES : Int64 = 2L

///|
const PAGED_ATTENTION_PAGE_ALIGNMENT : Int64 = 256L

///|
fn paged_attention_checked_product(
  left : Int64,
  right : Int64,
  rule : PagedAttentionAotRule,
) -> Int64 raise PagedAttentionAotError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise SizeOverflow(rule)
  }
  left * right
}

///|
fn paged_attention_align_page(
  value : Int64,
) -> Int64 raise PagedAttentionAotError {
  if value <= 0L || value > 9223372036854775552L {
    raise SizeOverflow(KvLayout)
  }
  let remainder = value % PAGED_ATTENTION_PAGE_ALIGNMENT
  if remainder == 0L {
    value
  } else {
    value + PAGED_ATTENTION_PAGE_ALIGNMENT - remainder
  }
}

///|
fn paged_attention_exact_runtime_inputs(
  actual : ReadOnlyArray[@plan.OperationRuntimeInput],
) -> Bool {
  let expected : Array[@plan.OperationRuntimeInput] = [
    StepCounts,
    QueryPositions,
    QueryRowOffsets,
    SequenceLengths,
    PageTableOffsets,
    PageTablePageIndices,
  ]
  if actual.length() != expected.length() {
    return false
  }
  for index, value in actual {
    if value != expected[index] {
      return false
    }
  }
  true
}

///|
fn paged_attention_expected_dimensions(
  profile : @launch_contract.PagedKernelProfile,
  query_heads : Int,
) -> @launch_contract.AotLaunchDimensions raise PagedAttentionAotError {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=Int::min(profile.max_query_tokens(), 32),
    grid_y=query_heads,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise Unsupported(Dimensions)
  }
}

///|
fn paged_attention_operand_role_matches(
  role : @launch_contract.AotOperandRole,
  index : Int,
  layer : @plan.KvLayerId,
) -> Bool {
  match index {
    0 => role == RuntimeInput(StepCounts)
    1 => role == RuntimeInput(QueryPositions)
    2 => role == RuntimeInput(QueryRowOffsets)
    3 => role == RuntimeInput(SequenceLengths)
    4 => role == RuntimeInput(PageTableOffsets)
    5 => role == RuntimeInput(PageTablePageIndices)
    6 => role is ActivationInput(_)
    7 => role is ActivationOutput(_)
    8 => role == PersistentKvComponent(layer, Key)
    9 => role == PersistentKvComponent(layer, Value)
    _ => false
  }
}

///|
fn paged_attention_validate_operand_bytes(
  operands : ArrayView[@launch_contract.AotOperand],
  profile : @launch_contract.PagedKernelProfile,
  query_heads : Int,
  kv_heads : Int,
  head_dimension : Int,
  tokens_per_page : Int,
) -> (Int64, Int64) raise PagedAttentionAotError {
  let max_rows = profile.max_query_rows().to_int64()
  let max_tokens = profile.max_query_tokens().to_int64()
  let max_pages = profile.max_page_table_entries().to_int64()
  let exact : Array[Int64] = [
    20L,
    paged_attention_checked_product(max_tokens, 4L, OperandBytes),
    paged_attention_checked_product(max_rows + 1L, 4L, OperandBytes),
    paged_attention_checked_product(max_rows, 4L, OperandBytes),
    paged_attention_checked_product(max_rows + 1L, 4L, OperandBytes),
    paged_attention_checked_product(max_pages, 4L, OperandBytes),
  ]
  for index in 0..<exact.length() {
    if operands[index].byte_count() != exact[index] ||
      operands[index].alignment() != 4L {
      raise Mismatch(OperandBytes)
    }
  }
  let qkv_width = paged_attention_checked_product(
    (query_heads + 2 * kv_heads).to_int64(),
    head_dimension.to_int64(),
    Shape,
  )
  let output_width = paged_attention_checked_product(
    query_heads.to_int64(),
    head_dimension.to_int64(),
    Shape,
  )
  let minimum_qkv = paged_attention_checked_product(
    paged_attention_checked_product(max_tokens, qkv_width, OperandBytes),
    PAGED_ATTENTION_BF16_BYTES,
    OperandBytes,
  )
  let minimum_output = paged_attention_checked_product(
    paged_attention_checked_product(max_tokens, output_width, OperandBytes),
    PAGED_ATTENTION_BF16_BYTES,
    OperandBytes,
  )
  if operands[6].byte_count() < minimum_qkv ||
    operands[7].byte_count() < minimum_output ||
    operands[6].alignment() < 2L ||
    operands[7].alignment() < 2L {
    raise Mismatch(OperandBytes)
  }
  let page_values = paged_attention_checked_product(
    paged_attention_checked_product(
      tokens_per_page.to_int64(),
      kv_heads.to_int64(),
      KvLayout,
    ),
    head_dimension.to_int64(),
    KvLayout,
  )
  let page_stride = paged_attention_align_page(
    paged_attention_checked_product(
      page_values,
      PAGED_ATTENTION_BF16_BYTES,
      KvLayout,
    ),
  )
  let component_bytes = operands[8].byte_count()
  if operands[9].byte_count() != component_bytes ||
    operands[8].alignment() != PAGED_ATTENTION_PAGE_ALIGNMENT ||
    operands[9].alignment() != PAGED_ATTENTION_PAGE_ALIGNMENT ||
    component_bytes <= 0L ||
    component_bytes % page_stride != 0L {
    raise Mismatch(KvLayout)
  }
  (page_stride, component_bytes / page_stride)
}

///|
fn paged_attention_validate_candidate(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  operands : ArrayView[@launch_contract.AotOperand],
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
) -> (Int, Int, Int, Int64, Int64) raise PagedAttentionAotError {
  if operation.kind() != CausalAttention {
    raise Unsupported(Operation)
  }
  if operation.required_capability() !=
    @plan.KernelCapabilityId::paged_causal_attention() {
    raise Mismatch(Capability)
  }
  guard operation.kv_layer() is Some(layer) &&
    operation.persistent_state() == KvCacheReadWrite(layer) &&
    paged_attention_exact_runtime_inputs(operation.runtime_inputs()) else {
    raise Mismatch(ExecutionSemantics)
  }
  guard operation.shape()
    is AttentionShape(query_heads~, kv_heads~, head_dimension~) else {
    raise Mismatch(Shape)
  }
  if query_heads <= 0 ||
    kv_heads <= 0 ||
    head_dimension <= 0 ||
    query_heads % kv_heads != 0 {
    raise Unsupported(Shape)
  }
  if profile.max_query_rows() <= 0 ||
    profile.max_query_tokens() < profile.max_query_rows() ||
    profile.max_page_table_entries() <= 0 {
    raise Unsupported(Profile)
  }
  if layout.stable_layout_version() != 1 || layout.tokens_per_page() <= 0 {
    raise Unsupported(KvLayout)
  }
  if !target.supports_bf16() {
    raise Unsupported(DeviceTarget)
  }
  if compiler.reassociate() {
    raise Unsupported(CompilerPolicy)
  }
  if operation.workspace_bytes() != 0L || operands.length() != 10 {
    raise Unsupported(Contract)
  }
  for index, operand in operands {
    if !paged_attention_operand_role_matches(operand.role(), index, layer) {
      raise Mismatch(OperandOrder)
    }
  }
  let (page_stride, page_capacity) = paged_attention_validate_operand_bytes(
    operands,
    profile,
    query_heads,
    kv_heads,
    head_dimension,
    layout.tokens_per_page(),
  )
  if page_stride != layout.page_stride_bytes() ||
    operands[8].byte_count() != layout.component_stride_bytes() ||
    page_capacity != layout.total_page_count().to_int64() {
    raise Mismatch(KvLayout)
  }
  (query_heads, kv_heads, head_dimension, page_stride, page_capacity)
}
