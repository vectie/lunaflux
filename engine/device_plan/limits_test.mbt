///|
fn repeated_plan(
  kind : @plan.OperationKind,
  repeat_count : Int,
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  let operations : Array[@plan.PlanOperation] = [try! device_plan_embedding(0)]
  let mut first_repeated_id = 1
  if kind == QkvProjection {
    operations.push(
      @plan.PlanOperation::new_stateless(
        id=@plan.OperationId::new(1),
        kind=RmsNorm,
        inputs=[
          Activation(@plan.ActivationRef::new(0)),
          Tensor(@plan.TensorRef::new(1)),
        ],
        outputs=[@plan.ActivationRef::new(1)],
        shape=NormalizationShape(width=4, epsilon=0.000001),
        required_capability=@plan.KernelCapabilityId::rms_norm(),
        workspace_bytes=0L,
      ),
    )
    first_repeated_id = 2
  }
  for repeated_index in 0..<repeat_count {
    let index = first_repeated_id + repeated_index
    let inputs : Array[@plan.ValueRef] = match kind {
      RmsNorm =>
        [
          Activation(@plan.ActivationRef::new(index - 1)),
          Tensor(@plan.TensorRef::new(1)),
        ]
      QkvProjection =>
        [
          Activation(@plan.ActivationRef::new(index - 1)),
          Tensor(@plan.TensorRef::new(2)),
          Tensor(@plan.TensorRef::new(3)),
          Tensor(@plan.TensorRef::new(4)),
        ]
      _ => abort("test helper supports only bounded repeated kinds")
    }
    let (shape, capability) : (@plan.OperationShape, @plan.KernelCapabilityId) = match
      kind {
      RmsNorm =>
        (
          NormalizationShape(width=4, epsilon=0.000001),
          @plan.KernelCapabilityId::rms_norm(),
        )
      QkvProjection =>
        (
          ProjectionShape(input_width=4, output_width=8),
          @plan.KernelCapabilityId::qkv_projection(),
        )
      _ => abort("test helper supports only bounded repeated kinds")
    }
    operations.push(
      @plan.PlanOperation::new_stateless(
        id=@plan.OperationId::new(index),
        kind~,
        inputs~,
        outputs=[@plan.ActivationRef::new(index)],
        shape~,
        required_capability=capability,
        workspace_bytes=0L,
      ),
    )
  }
  if operations.length() > 4096 {
    raise LimitExceeded(Operations)
  }
  @plan.ModelPlan::new_stateless(
    content_digest=(try! device_plan_test_model()).content_digest(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=try! device_plan_base_constraints(),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=64L),
    operations~,
    final_output=@plan.ActivationRef::new(operations.length() - 1),
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema(operations),
  )
}

///|
test "semantic plan preflight rejects excessive operations and admits expanded input envelope" {
  try repeated_plan(RmsNorm, 4096) catch {
    LimitExceeded(Operations) => ()
    _ => fail("unexpected semantic operation-bound error")
  } noraise {
    _ => fail("expected semantic operation-count bound")
  }
  let expanded = try! repeated_plan(QkvProjection, 2048)
  assert_eq(expanded.operations().length(), 2050)
}
