///|
fn device_plan_one_operation(
  identity : @spec.ModelIdentity,
  tensor_reference? : Int = 0,
) -> @plan.ModelPlan raise {
  @plan.ModelPlan::new(
    identity~,
    shape_constraints=device_plan_base_constraints(),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=64L),
    operations=[device_plan_embedding(tensor_reference)],
    final_output=@plan.ActivationRef::new(0),
  )
}

///|
fn device_plan_two_operations(
  second_kind : @plan.OperationKind,
  second_workspace? : Int64 = 0L,
) -> @plan.ModelPlan raise {
  let (shape, capability) : (@plan.OperationShape, @plan.KernelCapabilityId) = match
    second_kind {
    RmsNorm =>
      (
        NormalizationShape(width=4, epsilon=0.000001),
        @plan.KernelCapabilityId::rms_norm(),
      )
    OutputProjection =>
      (
        ProjectionShape(input_width=4, output_width=4),
        @plan.KernelCapabilityId::dense_projection(),
      )
    _ => abort("test helper supports only two operation kinds")
  }
  let second = @plan.PlanOperation::new(
    id=@plan.OperationId::new(1),
    kind=second_kind,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(1)),
    ],
    outputs=[@plan.ActivationRef::new(1)],
    shape~,
    required_capability=capability,
    workspace_bytes=second_workspace,
  )
  @plan.ModelPlan::new(
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
    operations=[device_plan_embedding(0), second],
    final_output=@plan.ActivationRef::new(1),
  )
}

///|
test "identity mismatches fail before operation planning" {
  let original = @llama.build(device_plan_test_model())
  let other = @llama.build(
    device_plan_test_model(
      digest="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
  try
    @device_plan.build(
      other,
      device_plan_test_layout(),
      device_plan_catalog_for(other),
    )
  catch {
    @device_plan.DevicePlanError::IdentityMismatch(WeightLayout) => ()
    _ => fail("unexpected identity error")
  } noraise {
    _ => fail("expected weight-layout identity mismatch")
  }
  try
    @device_plan.build(
      original,
      device_plan_test_layout(),
      device_plan_catalog_for(other),
    )
  catch {
    @device_plan.DevicePlanError::IdentityMismatch(KernelCatalog) => ()
    _ => fail("unexpected identity error")
  } noraise {
    _ => fail("expected kernel-catalog identity mismatch")
  }
}

///|
test "missing semantic tensor region fails closed" {
  let semantic = device_plan_one_operation(
    device_plan_test_model().identity(),
    tensor_reference=12,
  )
  try
    @device_plan.build(
      semantic,
      device_plan_test_layout(),
      device_plan_catalog_for(semantic),
    )
  catch {
    @device_plan.DevicePlanError::MissingWeightRegion(operation, tensor) => {
      assert_eq(operation.as_int(), 0)
      assert_eq(tensor.as_int(), 12)
    }
    _ => fail("unexpected region error")
  } noraise {
    _ => fail("expected missing region")
  }
}

///|
test "binding count order and capability are revalidated" {
  let one = device_plan_one_operation(device_plan_test_model().identity())
  let two = device_plan_two_operations(RmsNorm)
  try
    @device_plan.build(
      two,
      device_plan_test_layout(),
      device_plan_catalog_for(one),
    )
  catch {
    @device_plan.DevicePlanError::BindingCountMismatch => ()
    _ => fail("unexpected binding-count error")
  } noraise {
    _ => fail("expected binding-count mismatch")
  }
  let different = device_plan_two_operations(OutputProjection)
  try
    @device_plan.build(
      different,
      device_plan_test_layout(),
      device_plan_catalog_for(two),
    )
  catch {
    @device_plan.DevicePlanError::BindingCapabilityMismatch(operation) =>
      assert_eq(operation.as_int(), 1)
    _ => fail("unexpected binding-capability error")
  } noraise {
    _ => fail("expected binding-capability mismatch")
  }
}

///|
test "kernel workspace must satisfy the current semantic operation" {
  let smaller = device_plan_two_operations(RmsNorm)
  let larger = device_plan_two_operations(RmsNorm, second_workspace=32L)
  try
    @device_plan.build(
      larger,
      device_plan_test_layout(),
      device_plan_catalog_for(smaller, workspace_bytes=16L),
    )
  catch {
    @device_plan.DevicePlanError::WorkspaceMismatch(
      operation,
      InsufficientForOperation
    ) => assert_eq(operation.as_int(), 1)
    _ => fail("unexpected workspace error")
  } noraise {
    _ => fail("expected insufficient workspace")
  }
}
