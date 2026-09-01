///|
fn fusion_operation(
  plan : @plan.ModelPlan,
  index : Int,
) -> @plan.PlanOperation raise FusionPlanError {
  match plan.operations().get(index) {
    Some(operation) => operation
    None => raise InvalidFusionPlan(OperationOrder)
  }
}

///|
/// Matches the real Qwen dataflow beginning at one QKV projection. The result
/// is a backend-neutral semantic plan suitable for LunaTile/device lowering.
pub fn match_qkv_qknorm_positioned_rope_paged_kv_write(
  plan : @plan.ModelPlan,
  qkv_operation : @plan.OperationId,
) -> QkvQkNormPositionedRopePagedKvWritePlan raise FusionPlanError {
  let qkv_index = qkv_operation.as_int()
  let qkv = fusion_operation(plan, qkv_index)
  let qk_norm = fusion_operation(plan, qkv_index + 1)
  let rope = fusion_operation(plan, qkv_index + 2)
  let attention = fusion_operation(plan, qkv_index + 3)
  if qkv.id() != qkv_operation ||
    qkv.kind() != QkvProjection ||
    qk_norm.kind() != QkRmsNorm ||
    rope.kind() != RotaryEmbedding ||
    attention.kind() != CausalAttention {
    raise InvalidFusionPlan(OperationOrder)
  }
  guard qkv.shape() is ProjectionShape(input_width~, output_width~) &&
    qk_norm.shape()
    is QkNormalizationShape(
      query_heads~,
      kv_heads~,
      head_dimension~,
      epsilon=norm_epsilon
    ) &&
    rope.shape()
    is RotaryShape(
      query_heads=rope_query_heads,
      kv_heads=rope_kv_heads,
      head_dimension=rope_head_dimension,
      rope_theta~
    ) &&
    attention.shape()
    is AttentionShape(
      query_heads=attention_query_heads,
      kv_heads=attention_kv_heads,
      head_dimension=attention_head_dimension
    ) else {
    raise InvalidFusionPlan(Shape)
  }
  if input_width <= 0 ||
    query_heads <= 0 ||
    kv_heads <= 0 ||
    head_dimension <= 0 ||
    output_width != (query_heads + 2 * kv_heads) * head_dimension ||
    rope_query_heads != query_heads ||
    rope_kv_heads != kv_heads ||
    rope_head_dimension != head_dimension ||
    attention_query_heads != query_heads ||
    attention_kv_heads != kv_heads ||
    attention_head_dimension != head_dimension ||
    norm_epsilon.is_nan() ||
    norm_epsilon.is_inf() ||
    norm_epsilon <= 0.0 ||
    rope_theta.is_nan() ||
    rope_theta.is_inf() ||
    rope_theta <= 0.0 {
    raise InvalidFusionPlan(Shape)
  }
  guard qkv.outputs().get(0) is Some(projected) &&
    qk_norm.inputs().get(0) is Some(Activation(qk_input)) &&
    qk_norm.outputs().get(0) is Some(normalized) &&
    rope.inputs().get(0) is Some(Activation(rope_input)) &&
    rope.outputs().get(0) is Some(positioned) &&
    attention.inputs().get(0) is Some(Activation(attention_input)) &&
    qk_input == projected &&
    rope_input == normalized &&
    attention_input == positioned &&
    qkv.outputs().length() == 1 &&
    qk_norm.inputs().length() == 3 &&
    qk_norm.outputs().length() == 1 &&
    rope.inputs().length() == 1 &&
    rope.outputs().length() == 1 &&
    attention.inputs().length() == 1 else {
    raise InvalidFusionPlan(Dataflow)
  }
  guard attention.kv_layer() is Some(kv_layer) &&
    attention.persistent_state() is KvCacheReadWrite(persistent_layer) &&
    kv_layer == persistent_layer else {
    raise InvalidFusionPlan(PersistentState)
  }
  {
    qkv_operation: qkv.id(),
    qk_norm_operation: qk_norm.id(),
    rope_operation: rope.id(),
    attention_operation: attention.id(),
    kv_layer,
    input_width,
    output_width,
    query_heads,
    kv_heads,
    head_dimension,
    norm_epsilon,
    rope_theta,
    projected_qkv: projected,
    positioned_qkv: positioned,
  }
}

///|
pub fn find_qkv_qknorm_positioned_rope_paged_kv_write_plans(
  plan : @plan.ModelPlan,
) -> ReadOnlyArray[QkvQkNormPositionedRopePagedKvWritePlan] {
  let matches : Array[QkvQkNormPositionedRopePagedKvWritePlan] = []
  for operation in plan.operations() {
    if operation.kind() != QkvProjection {
      continue
    }
    let candidate = try
      match_qkv_qknorm_positioned_rope_paged_kv_write(plan, operation.id())
    catch {
      _ => None
    } noraise {
      value => Some(value)
    }
    match candidate {
      Some(value) => matches.push(value)
      None => ()
    }
  }
  ReadOnlyArray::from_array(matches)
}
