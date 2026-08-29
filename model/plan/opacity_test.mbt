///|
fn opacity_test_schema(
  operation : @plan.PlanOperation,
) -> @numeric_contract.ModelNumericSchema {
  let storage = try! numeric_test_plain(
    @numeric_contract.StorageDType::bfloat16(),
  )
  let execution = try! numeric_test_execution(
    @numeric_contract.ComputeDType::absent(),
    @numeric_contract.ComputeDType::bfloat16(),
  )
  try! @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[
      @numeric_contract.NumericTensorSchema::new(
        ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
        role=@numeric_contract.NumericTensorRole::parameter(),
        shape=@numeric_contract.NumericShape::new([8L, 4L]),
        storage~,
      ),
    ],
    operations=[
      @numeric_contract.NumericOperationSchema::new(
        ordinal=@numeric_contract.NumericOperationOrdinal::new(
          operation.id().as_int(),
        ),
        execution~,
      ),
    ],
  )
}

///|
test "opaque plan maps canonical numeric views and owns source arrays" {
  let tensor = try! @plan.TensorRef::new(0)
  let output = try! @plan.ActivationRef::new(0)
  let inputs : Array[@plan.ValueRef] = [ModelInput(TokenIds), Tensor(tensor)]
  let outputs : Array[@plan.ActivationRef] = [output]
  let operation = try! @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs~,
    outputs~,
    shape=EmbeddingShape(vocabulary_size=8, width=4),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  // Operation construction owns both source arrays.
  inputs[1] = ModelInput(TokenIds)
  outputs[0] = try! @plan.ActivationRef::new(7)
  assert_eq(operation.inputs()[1], Tensor(tensor))
  assert_eq(operation.outputs()[0], output)

  let schema = opacity_test_schema(operation)
  let source_operations : Array[@plan.PlanOperation] = [operation]
  let plan = try! @plan.ModelPlan::new_stateless(
    content_digest=numeric_test_content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=numeric_test_constraints(),
    kv_cache_geometry=numeric_test_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations=source_operations,
    final_output=output,
    numeric_schema=schema,
  )
  source_operations[0] = try! @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(tensor)],
    outputs=[output],
    shape=EmbeddingShape(vocabulary_size=8, width=4),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=1L,
  )
  assert_eq(plan.operations()[0].workspace_bytes(), 0L)
  assert_eq(plan.final_output(), output)
  assert_eq(plan.required_capabilities(), [
    @plan.KernelCapabilityId::embedding_lookup(),
  ])

  let numeric = plan.numeric_binding()
  assert_eq(numeric.schema(), schema)
  assert_eq(numeric.digest(), schema.digest())
  assert_eq(numeric.operations().length(), 1)
  assert_eq(numeric.operations()[0].operation(), operation.id())
  assert_eq(
    numeric.operations()[0].execution(),
    schema.operations()[0].execution(),
  )
  assert_eq(numeric.tensors().length(), 1)
  let mapped = numeric.tensors()[0]
  assert_eq(mapped.reference(), tensor)
  assert_eq(mapped.role(), ParameterTensor)
  assert_eq(mapped.shape(), schema.tensors()[0].shape())
  assert_eq(mapped.storage(), schema.tensors()[0].storage())
  assert_eq(mapped.scale(), None)
  assert_eq(mapped.zero_point(), None)
  assert_eq(mapped.codebook(), None)
  assert_eq(plan.tensor(tensor), Some(mapped))
}

///|
test "opaque validated identifiers preserve checked mappings" {
  let operation = try! @plan.OperationId::new(3)
  let tensor = try! @plan.TensorRef::new(4)
  let activation = try! @plan.ActivationRef::new(5)
  let layer = try! @plan.KvLayerId::new(6)
  let capability = try! @plan.KernelCapabilityId::from_stable_int(7)
  assert_eq(operation.as_int(), 3)
  assert_eq(tensor.as_int(), 4)
  assert_eq(activation.as_int(), 5)
  assert_eq(layer.as_int(), 6)
  assert_eq(capability.as_int(), 7)
  assert_eq(@plan.ScaleTensorRef::new(tensor).tensor(), tensor)
  assert_eq(@plan.ZeroPointTensorRef::new(tensor).tensor(), tensor)
  assert_eq(@plan.CodebookTensorRef::new(tensor).tensor(), tensor)
}

///|
test "opaque identifier constructors retain hostile validation" {
  try @plan.OperationId::new(-1) catch {
    InvalidIdentifier(Operation) => ()
    _ => fail("unexpected hostile operation ID error")
  } noraise {
    _ => fail("negative operation ID was admitted")
  }
  try @plan.TensorRef::new(-1) catch {
    InvalidIdentifier(Tensor) => ()
    _ => fail("unexpected hostile tensor reference error")
  } noraise {
    _ => fail("negative tensor reference was admitted")
  }
  try @plan.ActivationRef::new(-1) catch {
    InvalidIdentifier(Activation) => ()
    _ => fail("unexpected hostile activation reference error")
  } noraise {
    _ => fail("negative activation reference was admitted")
  }
  try @plan.KvLayerId::new(-1) catch {
    InvalidIdentifier(KvLayer) => ()
    _ => fail("unexpected hostile KV layer error")
  } noraise {
    _ => fail("negative KV layer was admitted")
  }
  try @plan.KernelCapabilityId::from_stable_int(0) catch {
    InvalidIdentifier(KernelCapability) => ()
    _ => fail("unexpected hostile capability error")
  } noraise {
    _ => fail("nonpositive kernel capability was admitted")
  }
}
