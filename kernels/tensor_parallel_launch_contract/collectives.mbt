///|
priv struct TensorParallelCollectiveFact {
  sequence : Int
  operation : Int
  kind : @tensor_parallel_plan.TensorCollectiveKind
  full_width : Int
  local_width : Int
}

///|
fn rank_collective_facts(
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
) -> Array[TensorParallelCollectiveFact] {
  let facts : Array[TensorParallelCollectiveFact] = []
  for index in 0..<rank_plan.collective_count() {
    guard rank_plan.collective(index) is Some(site) else { continue }
    facts.push({
      sequence: site.sequence().as_int(),
      operation: site.operation().as_int(),
      kind: site.kind(),
      full_width: site.full_width(),
      local_width: site.local_width(),
    })
  }
  facts
}

///|
fn validate_collective_facts(
  facts : ArrayView[TensorParallelCollectiveFact],
  operations : ReadOnlyArray[@tensor_parallel_plan.TensorParallelOperationPlan],
) -> ReadOnlyArray[TensorParallelCollectiveAfter] raise TensorParallelLaunchContractError {
  let output : Array[TensorParallelCollectiveAfter] = []
  for _ in operations {
    output.push(NoCollective)
  }
  let mut previous_operation = -1
  for index, fact in facts {
    if fact.sequence != index + 1 ||
      fact.operation <= previous_operation ||
      fact.operation < 0 ||
      fact.operation >= operations.length() {
      raise invalid(CollectiveSequence, operation=index)
    }
    let geometry = operations[fact.operation]
    if geometry.operation().as_int() != fact.operation {
      raise invalid(CollectiveOperation, operation=index)
    }
    match fact.kind {
      SumAllReduce => {
        if fact.full_width != geometry.output_width() ||
          fact.local_width != geometry.output_width() {
          raise invalid(CollectiveWidth, operation=index)
        }
        output[fact.operation] = CollectiveSumAllReduce(
          sequence=index + 1,
          width=fact.full_width,
        )
      }
      LastAxisAllGather => {
        if fact.local_width != geometry.output_width() ||
          fact.full_width <= fact.local_width {
          raise invalid(CollectiveWidth, operation=index)
        }
        output[fact.operation] = CollectiveLastAxisAllGather(
          sequence=index + 1,
          full_width=fact.full_width,
          local_width=fact.local_width,
        )
      }
    }
    previous_operation = fact.operation
  }
  ReadOnlyArray::from_array(output)
}

///|
fn collective_test_operation(
  index : Int,
  output_width : Int,
) -> @tensor_parallel_plan.TensorParallelOperationPlan raise {
  @tensor_parallel_plan.TensorParallelOperationPlan::new(
    operation=@plan.OperationId::new(index),
    kind=Embedding,
    capability=@plan.KernelCapabilityId::embedding_lookup(),
    input_width=1,
    output_width~,
    inner_width=0,
    query_head_count=0,
    kv_head_count=0,
    head_dimension=0,
  )
}

///|
test "collective sites bind exact generic operation geometry" {
  let operations = ReadOnlyArray::from_array([
    collective_test_operation(0, 8),
    collective_test_operation(1, 4),
  ])
  let exact : Array[TensorParallelCollectiveFact] = [
    {
      sequence: 1,
      operation: 0,
      kind: SumAllReduce,
      full_width: 8,
      local_width: 8,
    },
    {
      sequence: 2,
      operation: 1,
      kind: LastAxisAllGather,
      full_width: 8,
      local_width: 4,
    },
  ]
  assert_eq(validate_collective_facts(exact, operations).length(), 2)
  let width = exact.copy()
  width[1] = { ..width[1], local_width: 3 }
  try ignore(validate_collective_facts(width, operations)) catch {
    InvalidTensorParallelLaunchContract(rule=CollectiveWidth, ..) => ()
    _ => fail("unexpected collective-width rejection")
  } noraise {
    _ => fail("wrong operation-local collective width was admitted")
  }
  let reordered = exact.copy()
  reordered[1] = { ..reordered[1], operation: 0 }
  try ignore(validate_collective_facts(reordered, operations)) catch {
    InvalidTensorParallelLaunchContract(rule=CollectiveSequence, ..) => ()
    _ => fail("unexpected collective-order rejection")
  } noraise {
    _ => fail("reordered collective operation was admitted")
  }
}
