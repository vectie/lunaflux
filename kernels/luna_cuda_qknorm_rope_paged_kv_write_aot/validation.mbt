///|
fn postprocess_checked_mul(
  left : Int64,
  right : Int64,
  rule : QkNormRopePagedKvWriteAotRule,
) -> Int64 raise QkNormRopePagedKvWriteAotError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise SizeOverflow(rule)
  }
  left * right
}

///|
fn postprocess_operation(
  plan : @plan.ModelPlan,
  index : Int,
) -> @plan.PlanOperation raise QkNormRopePagedKvWriteAotError {
  match plan.operations().get(index) {
    Some(operation) => operation
    None => raise Mismatch(OperationSpan)
  }
}

///|
fn postprocess_semantic_span(
  plan : @plan.ModelPlan,
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
) -> (@plan.TensorRef, @plan.TensorRef) raise QkNormRopePagedKvWriteAotError {
  let (qkv_index, norm_index, rope_index, attention_index) = semantic.operation_indices()
  if norm_index != qkv_index + 1 ||
    rope_index != norm_index + 1 ||
    attention_index != rope_index + 1 {
    raise Mismatch(OperationSpan)
  }
  let qkv = postprocess_operation(plan, qkv_index)
  let norm = postprocess_operation(plan, norm_index)
  let rope = postprocess_operation(plan, rope_index)
  let attention = postprocess_operation(plan, attention_index)
  guard qkv.kind() == QkvProjection &&
    norm.kind() == QkRmsNorm &&
    rope.kind() == RotaryEmbedding &&
    attention.kind() == CausalAttention &&
    qkv.outputs().length() == 1 &&
    qkv.outputs()[0] == semantic.projected_qkv() &&
    norm.inputs().length() == 3 &&
    norm.inputs()[0] == Activation(semantic.projected_qkv()) &&
    norm.inputs()[1] is Tensor(query_norm_weight) &&
    norm.inputs()[2] is Tensor(key_norm_weight) &&
    norm.outputs().length() == 1 &&
    rope.inputs().length() == 1 &&
    rope.inputs()[0] == Activation(norm.outputs()[0]) &&
    rope.outputs().length() == 1 &&
    rope.outputs()[0] == semantic.positioned_qkv() &&
    attention.inputs().length() == 1 &&
    attention.inputs()[0] == Activation(semantic.positioned_qkv()) &&
    attention.kv_layer() == Some(semantic.kv_layer()) &&
    attention.persistent_state() == KvCacheReadWrite(semantic.kv_layer()) else {
    raise Mismatch(SemanticPlan)
  }
  (query_norm_weight, key_norm_weight)
}

///|
fn postprocess_expected_roles(
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  query_norm_weight : @plan.TensorRef,
  key_norm_weight : @plan.TensorRef,
) -> Array[@launch_contract.AotOperandRole] {
  [
    RuntimeInput(StepCounts),
    RuntimeInput(QueryPositions),
    RuntimeInput(QueryRowOffsets),
    RuntimeInput(SequenceLengths),
    RuntimeInput(PageTableOffsets),
    RuntimeInput(PageTablePageIndices),
    ActivationInput(semantic.projected_qkv()),
    WeightInput(query_norm_weight),
    WeightInput(key_norm_weight),
    ActivationOutput(semantic.positioned_qkv()),
    PersistentKvComponent(semantic.kv_layer(), Key),
    PersistentKvComponent(semantic.kv_layer(), Value),
  ]
}

///|
fn postprocess_validate_operands(
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  query_norm_weight : @plan.TensorRef,
  key_norm_weight : @plan.TensorRef,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  operands : ArrayView[@launch_contract.AotOperand],
) -> Unit raise QkNormRopePagedKvWriteAotError {
  let roles = postprocess_expected_roles(
    semantic, query_norm_weight, key_norm_weight,
  )
  if operands.length() != roles.length() {
    raise Mismatch(OperandOrder)
  }
  let (_, qkv_width) = semantic.projection_shape()
  let (_, _, head_dimension) = semantic.attention_shape()
  let rows = profile.max_query_rows().to_int64()
  let tokens = profile.max_query_tokens().to_int64()
  let pages = profile.max_page_table_entries().to_int64()
  let qkv_bytes = postprocess_checked_mul(
    postprocess_checked_mul(tokens, qkv_width.to_int64(), OperandSize),
    2L,
    OperandSize,
  )
  let norm_bytes = postprocess_checked_mul(
    head_dimension.to_int64(),
    2L,
    OperandSize,
  )
  let sizes : Array[Int64] = [
    20L,
    postprocess_checked_mul(tokens, 4L, OperandSize),
    postprocess_checked_mul(rows + 1L, 4L, OperandSize),
    postprocess_checked_mul(rows, 4L, OperandSize),
    postprocess_checked_mul(rows + 1L, 4L, OperandSize),
    postprocess_checked_mul(pages, 4L, OperandSize),
    qkv_bytes,
    norm_bytes,
    norm_bytes,
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
    let required = if index < 6 {
      4L
    } else if index < 10 {
      2L
    } else {
      layout.segment_alignment_bytes()
    }
    if operand.alignment() < required || operand.alignment() % required != 0L {
      raise Mismatch(OperandAlignment)
    }
  }
}

///|
fn postprocess_validate(
  plan : @plan.ModelPlan,
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
) -> (@plan.TensorRef, @plan.TensorRef) raise QkNormRopePagedKvWriteAotError {
  let (query_norm_weight, key_norm_weight) = postprocess_semantic_span(
    plan, semantic,
  )
  let (_, output_width) = semantic.projection_shape()
  let (query_heads, kv_heads, head_dimension) = semantic.attention_shape()
  if query_heads <= 0 ||
    kv_heads <= 0 ||
    query_heads % kv_heads != 0 ||
    head_dimension != 128 ||
    output_width != (query_heads + 2 * kv_heads) * head_dimension ||
    query_heads + 2 * kv_heads > 65535 ||
    semantic.norm_epsilon().is_nan() ||
    semantic.norm_epsilon().is_inf() ||
    semantic.norm_epsilon() <= 0.0 ||
    semantic.norm_epsilon() > 1.0 ||
    semantic.rope_theta().is_nan() ||
    semantic.rope_theta().is_inf() ||
    semantic.rope_theta() <= 0.0 {
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
  postprocess_validate_operands(
    semantic, query_norm_weight, key_norm_weight, profile, layout, operands,
  )
  (query_norm_weight, key_norm_weight)
}
