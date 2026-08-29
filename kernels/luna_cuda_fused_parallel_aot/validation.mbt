///|
const FUSED_PARALLEL_THREADS : Int = 128

///|
const FUSED_PARALLEL_MAX_QUERY_TOKENS : Int = 1024

///|
const FUSED_PARALLEL_MAX_WIDTH : Int = 16384

///|
fn fused_checked_mul(
  left : Int64,
  right : Int64,
) -> Int64 raise FusedParallelAotError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise FusedParallelOverflow(StaticResources)
  }
  left * right
}

///|
fn fused_is_power_of_two(value : Int) -> Bool {
  value > 0 && (value & (value - 1)) == 0
}

///|
fn fused_find_operation(
  plan : @plan.ModelPlan,
  id : @plan.OperationId,
) -> @plan.PlanOperation raise FusedParallelAotError {
  for operation in plan.operations() {
    if operation.id() == id {
      return operation
    }
  }
  raise FusedParallelMismatch(Operation)
}

///|
fn fused_validate_common(
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
) -> Unit raise FusedParallelAotError {
  if target.compute_major() < 8 ||
    target.compute_major() > 12 ||
    target.compute_minor() < 0 ||
    target.compute_minor() > 9 ||
    !target.supports_bf16() {
    raise FusedParallelUnsupported(DeviceTarget)
  }
  let compiler_identity = compiler.identity()
  if compiler_identity.major() != 13 ||
    compiler_identity.minor() != 1 ||
    compiler_identity.patch() != 115 ||
    compiler.optimization_level() != 3 ||
    compiler.fused_multiply_add() ||
    compiler.reassociate() ||
    compiler.max_registers() < 64 ||
    compiler.max_registers() > 255 {
    raise FusedParallelUnsupported(CompilerPolicy)
  }
  if profile.max_query_rows() <= 0 ||
    profile.max_query_rows() > 256 ||
    profile.max_query_tokens() < profile.max_query_rows() ||
    profile.max_query_tokens() > FUSED_PARALLEL_MAX_QUERY_TOKENS ||
    profile.max_page_table_entries() < profile.max_query_rows() ||
    profile.max_page_table_entries() > 4096 {
    raise FusedParallelUnsupported(Profile)
  }
}

///|
fn fused_minimum_alignment(
  role : @launch_contract.AotOperandRole,
  kv_alignment : Int64,
) -> Int64 {
  match role {
    RuntimeInput(_) => 4L
    PersistentKvComponent(_, _) => kv_alignment
    _ => 16L
  }
}

///|
fn fused_validate_operand_alignments(
  operands : ReadOnlyArray[@launch_contract.AotOperand],
  kv_alignment : Int64,
) -> Unit raise FusedParallelAotError {
  if operands.is_empty() {
    raise FusedParallelMismatch(OperandOrder)
  }
  for operand in operands {
    let required = fused_minimum_alignment(operand.role(), kv_alignment)
    if operand.byte_count() <= 0L ||
      operand.alignment() < required ||
      operand.alignment() % required != 0L {
      raise FusedParallelMismatch(OperandAlignment)
    }
  }
}

///|
fn fused_operation_roles(
  operation : @plan.PlanOperation,
) -> Array[@launch_contract.AotOperandRole] {
  let roles : Array[@launch_contract.AotOperandRole] = []
  for runtime in operation.runtime_inputs() {
    roles.push(RuntimeInput(runtime))
  }
  for input in operation.inputs() {
    roles.push(
      match input {
        ModelInput(reference) => ExternalInput(reference)
        Tensor(reference) => WeightInput(reference)
        Activation(reference) => ActivationInput(reference)
      },
    )
  }
  for output in operation.outputs() {
    roles.push(ActivationOutput(output))
  }
  roles
}

///|
fn fused_exact_roles(
  operands : ReadOnlyArray[@launch_contract.AotOperand],
  roles : ArrayView[@launch_contract.AotOperandRole],
) -> Bool {
  if operands.length() != roles.length() {
    return false
  }
  for index, operand in operands {
    if operand.role() != roles[index] {
      return false
    }
  }
  true
}

///|
fn fused_validate_qkv_fallback_abis(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  parameters : FusedQkvParameters,
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering,
  rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  kv_write : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
) -> Unit raise FusedParallelAotError {
  let qkv_operation = fused_find_operation(plan, parameters.qkv_operation)
  let rope_operation = fused_find_operation(plan, parameters.rope_operation)
  let attention = fused_find_operation(plan, parameters.attention_operation)
  if !fused_exact_roles(qkv.operands(), fused_operation_roles(qkv_operation)) ||
    !fused_exact_roles(rope.operands(), fused_operation_roles(rope_operation)) {
    raise FusedParallelMismatch(OperandOrder)
  }
  let kv_roles : Array[@launch_contract.AotOperandRole] = []
  for runtime in attention.runtime_inputs() {
    kv_roles.push(RuntimeInput(runtime))
  }
  guard attention.inputs()[0] is Activation(rotated) else {
    raise FusedParallelMismatch(Operation)
  }
  kv_roles.push(ActivationInput(rotated))
  kv_roles.push(PersistentKvComponent(parameters.kv_layer, Key))
  kv_roles.push(PersistentKvComponent(parameters.kv_layer, Value))
  if !fused_exact_roles(kv_write.operands(), kv_roles) {
    raise FusedParallelMismatch(OperandOrder)
  }
  for operand in kv_write.operands() {
    match operand.role() {
      PersistentKvComponent(_, _) =>
        if operand.byte_count() != layout.component_stride_bytes() {
          raise FusedParallelMismatch(OperandSize)
        }
      ActivationInput(_) => {
        let expected = fused_checked_mul(
          fused_checked_mul(
            qkv.profile().max_query_tokens().to_int64(),
            parameters.output_width.to_int64(),
          ),
          2L,
        )
        if operand.byte_count() != expected {
          raise FusedParallelMismatch(OperandSize)
        }
      }
      _ => ()
    }
  }
}

///|
fn fused_validate_residual_norm_fallback_abis(
  plan : @plan.ModelPlan,
  parameters : FusedResidualNormParameters,
  residual : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
) -> Unit raise FusedParallelAotError {
  let residual_operation = fused_find_operation(
    plan,
    parameters.residual_operation,
  )
  let norm_operation = fused_find_operation(plan, parameters.norm_operation)
  if !fused_exact_roles(
      residual.operands(),
      fused_operation_roles(residual_operation),
    ) ||
    !fused_exact_roles(norm.operands(), fused_operation_roles(norm_operation)) {
    raise FusedParallelMismatch(OperandOrder)
  }
}

///|
fn fused_qkv_parameters(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  qkv_id : @plan.OperationId,
  rope_id : @plan.OperationId,
  attention_id : @plan.OperationId,
) -> FusedQkvParameters raise FusedParallelAotError {
  let qkv = fused_find_operation(plan, qkv_id)
  let rope = fused_find_operation(plan, rope_id)
  let attention = fused_find_operation(plan, attention_id)
  guard qkv.shape() is ProjectionShape(input_width~, output_width~) &&
    qkv.kind() == QkvProjection else {
    raise FusedParallelUnsupported(Operation)
  }
  guard rope.shape()
    is RotaryShape(query_heads~, kv_heads~, head_dimension~, rope_theta~) &&
    rope.kind() == RotaryEmbedding else {
    raise FusedParallelUnsupported(Operation)
  }
  guard attention.shape()
    is AttentionShape(
      query_heads=attention_query_heads,
      kv_heads=attention_kv_heads,
      head_dimension=attention_head_dimension
    ) &&
    attention.kind() == CausalAttention else {
    raise FusedParallelUnsupported(Operation)
  }
  if qkv.outputs().length() != 1 ||
    rope.inputs().length() != 1 ||
    rope.inputs()[0] != Activation(qkv.outputs()[0]) ||
    rope.outputs().length() != 1 ||
    attention.inputs().length() != 1 ||
    attention.inputs()[0] != Activation(rope.outputs()[0]) ||
    qkv.id().as_int() + 1 != rope.id().as_int() ||
    rope.id().as_int() + 1 != attention.id().as_int() {
    raise FusedParallelMismatch(OperationOrder)
  }
  let expected_output = fused_checked_mul(
    (query_heads + 2 * kv_heads).to_int64(),
    head_dimension.to_int64(),
  )
  if input_width < 64 ||
    input_width > FUSED_PARALLEL_MAX_WIDTH ||
    input_width % 16 != 0 ||
    output_width.to_int64() != expected_output ||
    output_width % 16 != 0 ||
    query_heads <= 0 ||
    kv_heads <= 0 ||
    query_heads % kv_heads != 0 ||
    head_dimension < 32 ||
    head_dimension > 128 ||
    head_dimension % 16 != 0 ||
    attention_query_heads != query_heads ||
    attention_kv_heads != kv_heads ||
    attention_head_dimension != head_dimension ||
    rope_theta.is_nan() ||
    rope_theta.is_inf() ||
    rope_theta <= 0.0 {
    raise FusedParallelUnsupported(Shape)
  }
  guard attention.kv_layer() is Some(layer) &&
    attention.persistent_state() is KvCacheReadWrite(persistent_layer) &&
    layer == persistent_layer else {
    raise FusedParallelMismatch(KvLayout)
  }
  if !layout.is_canonical_for(plan.identity(), plan.kv_cache_geometry()) ||
    layout.stable_layout_version() != 1 ||
    layout.geometry().dtype() != BFloat16 ||
    layout.geometry().layout() != LayerMajorSplitKeyValueV1 ||
    layout.geometry().kv_head_count() != kv_heads ||
    layout.geometry().head_dimension() != head_dimension ||
    !fused_is_power_of_two(layout.tokens_per_page()) ||
    layout.tokens_per_page() < 8 ||
    layout.tokens_per_page() > 256 ||
    layout.segment_alignment_bytes() < 256L ||
    layout.page_stride_bytes() % layout.segment_alignment_bytes() != 0L {
    raise FusedParallelMismatch(KvLayout)
  }
  {
    model_content: plan.identity().content().as_hex(),
    model_plan: plan.identity().plan().as_hex(),
    qkv_operation: qkv.id(),
    rope_operation: rope.id(),
    attention_operation: attention.id(),
    kv_layer: layer,
    input_width,
    output_width,
    query_heads,
    kv_heads,
    head_dimension,
    rope_theta,
    tokens_per_page: layout.tokens_per_page(),
    total_page_count: layout.total_page_count(),
    page_stride_bytes: layout.page_stride_bytes(),
    component_stride_bytes: layout.component_stride_bytes(),
  }
}

///|
fn fused_residual_norm_parameters(
  plan : @plan.ModelPlan,
  residual_id : @plan.OperationId,
  norm_id : @plan.OperationId,
) -> FusedResidualNormParameters raise FusedParallelAotError {
  let residual = fused_find_operation(plan, residual_id)
  let norm = fused_find_operation(plan, norm_id)
  guard residual.shape() is ElementwiseShape(width~) &&
    residual.kind() == ResidualAdd else {
    raise FusedParallelUnsupported(Operation)
  }
  guard norm.shape() is NormalizationShape(width=norm_width, epsilon~) &&
    norm.kind() == RmsNorm else {
    raise FusedParallelUnsupported(Operation)
  }
  if residual.outputs().length() != 1 ||
    norm.inputs().length() != 2 ||
    norm.inputs()[0] != Activation(residual.outputs()[0]) ||
    residual.id().as_int() + 1 != norm.id().as_int() {
    raise FusedParallelMismatch(OperationOrder)
  }
  if width != norm_width ||
    width < 64 ||
    width > FUSED_PARALLEL_MAX_WIDTH ||
    width % 16 != 0 ||
    epsilon.is_nan() ||
    epsilon.is_inf() ||
    epsilon <= 0.0 ||
    epsilon > 0.01 {
    raise FusedParallelUnsupported(Shape)
  }
  {
    model_content: plan.identity().content().as_hex(),
    model_plan: plan.identity().plan().as_hex(),
    residual_operation: residual.id(),
    norm_operation: norm.id(),
    width,
    epsilon,
  }
}
