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
    output_width <= 0 ||
    output_width % head_dimension != 0 ||
    (output_width / head_dimension).to_int64() !=
    query_heads.to_int64() + 2L * kv_heads.to_int64() ||
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
    qkv.persistent_state() == NoPersistentState &&
    qk_norm.persistent_state() == NoPersistentState &&
    rope.persistent_state() == NoPersistentState &&
    attention.persistent_state() is KvCacheReadWrite(persistent_layer) &&
    kv_layer == persistent_layer else {
    raise InvalidFusionPlan(PersistentState)
  }
  // Record demand before choosing a fusion cut. A separated producer keeps
  // projected QKV observable even when a fully fused cut would be illegal.
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
    projected_exclusive: fusion_exclusive_activation(
      plan.operations(),
      plan.final_output(),
      projected,
      qk_norm.id(),
    ),
    normalized_exclusive: fusion_exclusive_activation(
      plan.operations(),
      plan.final_output(),
      normalized,
      rope.id(),
    ),
  }
}

///|
fn fusion_exclusive_activation(
  operations : ReadOnlyArray[@plan.PlanOperation],
  final_output : @plan.ActivationRef,
  activation : @plan.ActivationRef,
  consumer : @plan.OperationId,
) -> Bool {
  if final_output == activation {
    return false
  }
  for operation in operations; uses = 0 {
    let count = operation
      .inputs()
      .fold(init=0, (count, input) => {
        count + (if input == Activation(activation) { 1 } else { 0 })
      })
    if count > 0 && operation.id() != consumer {
      return false
    }
    continue uses + count
  } nobreak {
    uses == 1
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
