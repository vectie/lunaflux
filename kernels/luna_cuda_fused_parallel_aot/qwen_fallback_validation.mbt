///|
fn qwen_fused_validate_fallbacks(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering,
  qk_norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  kv_write : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
) -> Unit raise FusedParallelAotError {
  let (qkv_index, qk_norm_index, rope_index, attention_index) = semantic.operation_indices()
  let qkv_operation = fused_find_operation(
    plan,
    @plan.OperationId::new(qkv_index) catch {
      _ => raise FusedParallelMismatch(Operation)
    },
  )
  let qk_norm_operation = fused_find_operation(
    plan,
    @plan.OperationId::new(qk_norm_index) catch {
      _ => raise FusedParallelMismatch(Operation)
    },
  )
  let rope_operation = fused_find_operation(
    plan,
    @plan.OperationId::new(rope_index) catch {
      _ => raise FusedParallelMismatch(Operation)
    },
  )
  let attention = fused_find_operation(
    plan,
    @plan.OperationId::new(attention_index) catch {
      _ => raise FusedParallelMismatch(Operation)
    },
  )
  if qkv.operation_id() != qkv_operation.id() ||
    qk_norm.operation_id() != qk_norm_operation.id() ||
    rope.operation_id() != rope_operation.id() ||
    kv_write.operation_id() != attention.id() ||
    !fused_exact_roles(qkv.operands(), fused_operation_roles(qkv_operation)) ||
    !fused_exact_roles(
      qk_norm.operands(),
      fused_operation_roles(qk_norm_operation),
    ) ||
    !fused_exact_roles(rope.operands(), fused_operation_roles(rope_operation)) {
    raise FusedParallelMismatch(OperandOrder)
  }
  let kv_roles : Array[@launch_contract.AotOperandRole] = []
  for runtime in attention.runtime_inputs() {
    kv_roles.push(RuntimeInput(runtime))
  }
  kv_roles.push(ActivationInput(semantic.positioned_qkv()))
  kv_roles.push(PersistentKvComponent(semantic.kv_layer(), Key))
  kv_roles.push(PersistentKvComponent(semantic.kv_layer(), Value))
  if !fused_exact_roles(kv_write.operands(), kv_roles) {
    raise FusedParallelMismatch(OperandOrder)
  }
  for operand in kv_write.operands() {
    match operand.role() {
      PersistentKvComponent(_, _) =>
        if operand.byte_count() != layout.component_stride_bytes() {
          raise FusedParallelMismatch(OperandSize)
        }
      _ => ()
    }
  }
}
