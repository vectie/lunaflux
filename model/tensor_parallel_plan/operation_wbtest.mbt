///|
fn operation_test_plan(
  operations : ArrayView[TensorParallelOperationPlan],
  collectives : ArrayView[TensorCollectiveSite],
) -> TensorParallelRankPlan raise {
  TensorParallelRankPlan::new(
    model=generic_test_model(),
    topology=TensorParallelTopology::new(rank=0, world_size=2),
    alignment_bytes=16L,
    arena_bytes=16L,
    materialized_bytes=16L,
    tensors=[generic_row_tensor(0, 8L, 0L, 4L, 0L, 16L)],
    operations~,
    collectives~,
  )
}

///|
test "generic operation plans require exact contiguous semantic order" {
  let stale = generic_operation(index=1)
  try ignore(operation_test_plan([stale], [])) catch {
    InvalidTensorParallelPlan(rule=OperationOrder, index=0) => ()
    _ => fail("unexpected operation-order rejection")
  } noraise {
    _ => fail("stale operation reference was admitted")
  }
}

///|
test "generic operation plans require complete semantic coverage" {
  try ignore(operation_test_plan([], [])) catch {
    InvalidTensorParallelPlan(rule=OperationOrder, index=0) => ()
    _ => fail("unexpected operation-coverage rejection")
  } noraise {
    _ => fail("missing operation coverage was admitted")
  }
}

///|
test "generic operation plans reject scalar substitution from semantic shape" {
  let substituted = generic_operation(output_width=8)
  try ignore(operation_test_plan([substituted], [])) catch {
    InvalidTensorParallelPlan(rule=OperationGeometry, index=0) => ()
    _ => fail("unexpected semantic-geometry rejection")
  } noraise {
    _ => fail("substituted operation geometry was admitted")
  }
}

///|
test "generic operation plans reject invalid local head geometry" {
  try
    ignore(
      TensorParallelOperationPlan::new(
        operation=@plan.OperationId::new(0),
        kind=QkvProjection,
        capability=@plan.KernelCapabilityId::qkv_projection(),
        input_width=8,
        output_width=8,
        inner_width=0,
        query_head_count=2,
        kv_head_count=3,
        head_dimension=4,
      ),
    )
  catch {
    InvalidTensorParallelPlan(rule=OperationGeometry, index=0) => ()
    _ => fail("unexpected operation-geometry rejection")
  } noraise {
    _ => fail("invalid head geometry was admitted")
  }
}

///|
test "generic operation plans fail closed on QK RMS normalization" {
  try
    ignore(
      TensorParallelOperationPlan::new(
        operation=@plan.OperationId::new(0),
        kind=QkRmsNorm,
        capability=@plan.KernelCapabilityId::qk_rms_norm(),
        input_width=8,
        output_width=8,
        inner_width=0,
        query_head_count=0,
        kv_head_count=0,
        head_dimension=0,
      ),
    )
  catch {
    InvalidTensorParallelPlan(rule=UnsupportedSemanticOperation, index=0) => ()
    _ => fail("unexpected unsupported-semantic rejection")
  } noraise {
    _ => fail("QK RMS normalization was admitted for tensor parallelism")
  }
}

///|
test "generic collective binds an exact covered operation" {
  let operation = generic_operation(output_width=4)
  let gather = TensorCollectiveSite::new(
    sequence=1,
    operation=@plan.OperationId::new(1),
    kind=LastAxisAllGather,
    full_width=8,
    local_width=4,
    world_size=2,
  )
  try ignore(operation_test_plan([operation], [gather])) catch {
    InvalidTensorParallelPlan(rule=CollectiveOrder, index=0) => ()
    _ => fail("unexpected collective-operation rejection")
  } noraise {
    _ => fail("collective bound to the wrong operation width was admitted")
  }
}
