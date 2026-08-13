///|
fn repeated_plan(
  kind : @plan.OperationKind,
  repeat_count : Int,
) -> @plan.ModelPlan raise {
  let operations : Array[@plan.PlanOperation] = [device_plan_embedding(0)]
  for index in 1..<=repeat_count {
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
  @plan.ModelPlan::new_stateless(
    identity=device_plan_test_model().identity(),
    shape_constraints=device_plan_base_constraints(),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=64L),
    operations~,
    final_output=@plan.ActivationRef::new(repeat_count),
  )
}

///|
test "startup planning rejects excessive operation and input totals" {
  let too_many_operations = repeated_plan(RmsNorm, 4096)
  try
    @device_plan.build(
      too_many_operations,
      device_plan_test_layout(),
      device_plan_catalog_for(too_many_operations),
    )
  catch {
    @device_plan.DevicePlanError::BoundExceeded(OperationCount) => ()
    _ => fail("unexpected operation-bound error")
  } noraise {
    _ => fail("expected operation-count bound")
  }
  let too_many_inputs = repeated_plan(QkvProjection, 2048)
  try
    @device_plan.build(
      too_many_inputs,
      device_plan_test_layout(),
      device_plan_catalog_for(too_many_inputs),
    )
  catch {
    @device_plan.DevicePlanError::BoundExceeded(TotalInputCount) => ()
    _ => fail("unexpected input-bound error")
  } noraise {
    _ => fail("expected total-input bound")
  }
}
