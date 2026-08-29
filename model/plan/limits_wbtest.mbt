///|
fn limits_wb_embedding(
  id : Int,
  inputs : ArrayView[ValueRef],
  outputs? : Array[ActivationRef] = [{ value: 0 }],
  runtime_inputs? : Array[OperationRuntimeInput] = [],
) -> PlanOperation raise {
  if runtime_inputs.length() == 0 {
    PlanOperation::new_stateless(
      id=OperationId::new(id),
      kind=Embedding,
      inputs~,
      outputs~,
      shape=EmbeddingShape(vocabulary_size=128, width=16),
      required_capability=KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
    )
  } else {
    PlanOperation::new_with_runtime_inputs(
      id=OperationId::new(id),
      kind=Embedding,
      inputs~,
      outputs~,
      shape=EmbeddingShape(vocabulary_size=128, width=16),
      required_capability=KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
      runtime_inputs~,
    )
  }
}

///|
fn limits_wb_limits(
  numeric? : Int = 16,
  total? : Int = 16,
  per_operation? : Int = 8,
  runtime? : Int = 8,
) -> ModelPlanLimits raise {
  ModelPlanLimits::new(
    max_operations=8,
    max_numeric_tensors=numeric,
    max_total_value_inputs=total,
    max_value_inputs_per_operation=per_operation,
    max_runtime_inputs_per_operation=runtime,
    max_outputs_per_operation=1,
    max_canonical_bytes=8192,
  )
}

///|
fn limits_wb_preflight(
  operations : ArrayView[PlanOperation],
  schema : @numeric_contract.ModelNumericSchema,
  limits : ModelPlanLimits,
) -> Unit raise PlanValidationError {
  ignore(
    preflight_model_plan(
      limits,
      canonical_wb_constraints(),
      canonical_wb_geometry(),
      WorkspaceBounds::new(max_temporary_bytes=0L),
      operations,
      operations[operations.length() - 1].outputs()[0],
      PagedKeyValue,
      schema,
    ),
  )
}

///|
test "preflight enforces numeric, aggregate, per-operation, and runtime limits" {
  let embedding = limits_wb_embedding(0, [
    ModelInput(TokenIds),
    Tensor(TensorRef::new(0)),
  ])
  let projection = PlanOperation::new_stateless(
    id=OperationId::new(1),
    kind=QkvProjection,
    inputs=[
      Activation(ActivationRef::new(0)),
      Tensor(TensorRef::new(1)),
      Tensor(TensorRef::new(2)),
      Tensor(TensorRef::new(3)),
    ],
    outputs=[ActivationRef::new(1)],
    shape=ProjectionShape(input_width=16, output_width=32),
    required_capability=KernelCapabilityId::qkv_projection(),
    workspace_bytes=0L,
  )
  let numeric_operations = [embedding, projection]
  try
    limits_wb_preflight(
      numeric_operations,
      @plan_test_fixture.synthetic_bf16_numeric_schema(numeric_operations),
      limits_wb_limits(numeric=3),
    )
  catch {
    LimitExceeded(NumericTensors) => ()
    _ => fail("unexpected numeric tensor limit error")
  } noraise {
    _ => fail("numeric tensor limit was ignored")
  }

  let too_many_inputs = limits_wb_embedding(0, [
    ModelInput(TokenIds),
    Tensor(TensorRef::new(0)),
    Tensor(TensorRef::new(1)),
  ])
  try
    limits_wb_preflight(
      [too_many_inputs],
      @plan_test_fixture.synthetic_bf16_numeric_schema([too_many_inputs]),
      limits_wb_limits(per_operation=2),
    )
  catch {
    LimitExceeded(ValueInputsPerOperation) => ()
    _ => fail("unexpected per-operation input limit error")
  } noraise {
    _ => fail("per-operation input limit was ignored")
  }

  let second = limits_wb_embedding(
    1,
    [ModelInput(TokenIds), Tensor(TensorRef::new(0))],
    outputs=[ActivationRef::new(1)],
  )
  let aggregate = [embedding, second]
  try
    limits_wb_preflight(
      aggregate,
      @plan_test_fixture.synthetic_bf16_numeric_schema(aggregate),
      limits_wb_limits(total=3, per_operation=2),
    )
  catch {
    LimitExceeded(TotalValueInputs) => ()
    _ => fail("unexpected aggregate input limit error")
  } noraise {
    _ => fail("aggregate input limit was ignored")
  }

  let runtime = limits_wb_embedding(
    0,
    [ModelInput(TokenIds), Tensor(TensorRef::new(0))],
    runtime_inputs=[StepCounts, QueryPositions],
  )
  try
    limits_wb_preflight(
      [runtime],
      @plan_test_fixture.synthetic_bf16_numeric_schema([runtime]),
      limits_wb_limits(runtime=1),
    )
  catch {
    LimitExceeded(RuntimeInputsPerOperation) => ()
    _ => fail("unexpected runtime input limit error")
  } noraise {
    _ => fail("runtime input limit was ignored")
  }
}

///|
test "preflight rejects hostile output arity before canonical traversal" {
  let operation = limits_wb_embedding(
    0,
    [ModelInput(TokenIds), Tensor(TensorRef::new(0))],
    outputs=[ActivationRef::new(0), ActivationRef::new(1)],
  )
  try
    limits_wb_preflight(
      [operation],
      @plan_test_fixture.synthetic_bf16_numeric_schema([operation]),
      limits_wb_limits(),
    )
  catch {
    LimitExceeded(OutputsPerOperation) => ()
    _ => fail("unexpected hostile output arity error")
  } noraise {
    _ => fail("hostile output arity reached canonical traversal")
  }
}

///|
test "resource output envelope intentionally precedes ordered graph errors" {
  let earlier_bad_order = PlanOperation::new_stateless(
    id=OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(TensorRef::new(0))],
    outputs=[ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=KernelCapabilityId::rms_norm(),
    workspace_bytes=0L,
  )
  let hostile_outputs = limits_wb_embedding(
    1,
    [ModelInput(TokenIds), Tensor(TensorRef::new(0))],
    outputs=[ActivationRef::new(1), ActivationRef::new(2)],
  )
  let operations = [earlier_bad_order, hostile_outputs]
  try
    limits_wb_preflight(
      operations,
      @plan_test_fixture.synthetic_bf16_numeric_schema(operations),
      limits_wb_limits(),
    )
  catch {
    LimitExceeded(OutputsPerOperation) => ()
    _ => fail("resource-first output ordering changed")
  } noraise {
    _ => fail("hostile output list bypassed the resource envelope")
  }
}

///|
test "preflight rejects numeric operation count before schema mapping" {
  let first = limits_wb_embedding(0, [
    ModelInput(TokenIds),
    Tensor(TensorRef::new(0)),
  ])
  let second = limits_wb_embedding(
    1,
    [ModelInput(TokenIds), Tensor(TensorRef::new(0))],
    outputs=[ActivationRef::new(1)],
  )
  let schema = @plan_test_fixture.synthetic_bf16_numeric_schema([first, second])
  try limits_wb_preflight([first], schema, limits_wb_limits()) catch {
    InvalidGraph(NumericOperationBinding) => ()
    _ => fail("unexpected numeric operation count error")
  } noraise {
    _ => fail("numeric operation mismatch reached schema mapping")
  }
}

///|
fn limits_wb_invalid_kind(
  numeric : Int,
  total : Int,
  per_operation : Int,
  runtime : Int,
  outputs : Int,
  canonical : Int,
) -> ModelPlanLimitKind? {
  try
    ignore(
      ModelPlanLimits::new(
        max_operations=1,
        max_numeric_tensors=numeric,
        max_total_value_inputs=total,
        max_value_inputs_per_operation=per_operation,
        max_runtime_inputs_per_operation=runtime,
        max_outputs_per_operation=outputs,
        max_canonical_bytes=canonical,
      ),
    )
  catch {
    InvalidLimit(kind) => Some(kind)
    _ => None
  } noraise {
    _ => None
  }
}

///|
test "every model plan limit field has a hard factory ceiling" {
  assert_eq(limits_wb_invalid_kind(65537, 1, 1, 1, 1, 1), Some(NumericTensors))
  assert_eq(
    limits_wb_invalid_kind(1, 16385, 1, 1, 1, 1),
    Some(TotalValueInputs),
  )
  assert_eq(
    limits_wb_invalid_kind(1, 16384, 65, 1, 1, 1),
    Some(ValueInputsPerOperation),
  )
  assert_eq(
    limits_wb_invalid_kind(1, 1, 1, 17, 1, 1),
    Some(RuntimeInputsPerOperation),
  )
  assert_eq(limits_wb_invalid_kind(1, 1, 1, 1, 2, 1), Some(OutputsPerOperation))
  assert_eq(
    limits_wb_invalid_kind(1, 1, 1, 1, 1, 16 * 1024 * 1024 + 1),
    Some(CanonicalBytes),
  )
}
