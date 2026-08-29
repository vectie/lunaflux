///|
fn device_plan_one_operation(
  content : @spec.ContentDigest,
) -> @plan.ModelPlan raise {
  @plan.ModelPlan::new_stateless(
    content_digest=content,
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=device_plan_base_constraints(),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=64L),
    operations=[device_plan_embedding(0)],
    final_output=@plan.ActivationRef::new(0),
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema([
      device_plan_embedding(0),
    ]),
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
  let second = @plan.PlanOperation::new_stateless(
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
  @plan.ModelPlan::new_stateless(
    content_digest=device_plan_test_model().content_digest(),
    limits=@plan.ModelPlanLimits::production(),
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
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema([
      device_plan_embedding(0),
      second,
    ]),
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
    @device_plan.DevicePlanError::IdentityMismatch(WeightContent) => ()
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
test "weight content is reusable across exact execution-plan identities" {
  let model = device_plan_test_model()
  let semantic = device_plan_one_operation(model.content_digest())
  let planned = @device_plan.build(
    semantic,
    device_plan_numeric_layout(semantic),
    device_plan_catalog_for(semantic),
  )
  assert_eq(planned.identity(), semantic.identity())
  assert_eq(planned.weight_tensor_count(), semantic.tensor_count())
}

///|
test "missing semantic tensor region fails closed" {
  let base = @llama.build(device_plan_test_model())
  let operations : Array[@plan.PlanOperation] = []
  for operation in base.operations() {
    operations.push(operation)
  }
  let extra_id = @plan.OperationId::new(operations.length())
  let extra_output = @plan.ActivationRef::new(operations.length())
  operations.push(
    @plan.PlanOperation::new_stateless(
      id=extra_id,
      kind=LanguageModelHead,
      inputs=[Activation(base.final_output()), Tensor(@plan.TensorRef::new(12))],
      outputs=[extra_output],
      shape=ProjectionShape(input_width=4, output_width=8),
      required_capability=@plan.KernelCapabilityId::dense_projection(),
      workspace_bytes=0L,
    ),
  )
  let semantic = @plan.ModelPlan::new_with_kv_execution(
    content_digest=base.identity().content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=base.shape_constraints(),
    kv_cache_geometry=base.kv_cache_geometry(),
    workspace_bounds=base.workspace_bounds(),
    operations~,
    final_output=extra_output,
    kv_execution=base.kv_execution(),
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema(operations),
  )
  try
    @device_plan.build(
      semantic,
      device_plan_test_layout(),
      device_plan_catalog_for(semantic),
    )
  catch {
    @device_plan.DevicePlanError::InvalidWeightArena(RegionCount) => ()
    _ => fail("unexpected region error")
  } noraise {
    _ => fail("expected missing region")
  }
}

///|
test "weight regions accept exact numeric-schema byte sizes" {
  let semantic = device_plan_two_operations(OutputProjection)
  let layout = device_plan_numeric_layout(semantic)
  let planned = try! @device_plan.build(
    semantic,
    layout,
    device_plan_catalog_for(semantic),
  )
  assert_eq(planned.weight_layout(), layout)
}

///|
test "binding count order and capability are revalidated" {
  let one = device_plan_one_operation(device_plan_test_model().content_digest())
  let two = device_plan_two_operations(RmsNorm)
  try
    @device_plan.build(
      two,
      device_plan_numeric_layout(two),
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
      device_plan_numeric_layout(different),
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
      device_plan_numeric_layout(larger),
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
