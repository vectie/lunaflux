///|
test "rank plan derives exact row column and replicated placements" {
  let (model, bindings) = tp_fixture()
  let rank = tp_plan(model, bindings, 1, 2)
  assert_eq(rank.topology().rank(), 1)
  assert_eq(rank.topology().world_size(), 2)
  assert_eq(rank.tensor_count(), 12)
  assert_eq(rank.operation_count(), model.operations().length())
  let mlp_operation = try! @plan.OperationId::new(8)
  assert_eq(rank.operation(mlp_operation).unwrap().inner_width(), 6)
  for reference in 0..<rank.tensor_count() {
    let tensor_reference = try! @plan.TensorRef::new(reference)
    let rank_tensor = rank.tensor(tensor_reference).unwrap()
    let full_bytes = rank_tensor.extent().full_rows() *
      rank_tensor.extent().full_columns() *
      2L
    if rank_tensor.placement() == Replicated {
      assert_eq(rank_tensor.extent().byte_length(), full_bytes)
    } else {
      assert_true(rank_tensor.extent().byte_length() < full_bytes)
    }
  }
  let embedding_reference = try! @plan.TensorRef::new(0)
  let embedding = rank.tensor(embedding_reference).unwrap()
  assert_eq(embedding.placement(), ShardRows)
  assert_eq(embedding.extent().full_rows(), 16L)
  assert_eq(embedding.extent().row_start(), 8L)
  assert_eq(embedding.extent().row_count(), 8L)
  assert_eq(embedding.extent().column_count(), 8L)
  assert_eq(embedding.extent().byte_length(), 128L)
  assert_eq(embedding.transfer_recipe().segment_count(), 1L)
  assert_eq(
    (try! embedding.transfer_recipe().segment(0L)).source_payload_offset_bytes(),
    128L,
  )
  let norm_reference = try! @plan.TensorRef::new(1)
  let norm = rank.tensor(norm_reference).unwrap()
  assert_eq(norm.placement(), Replicated)
  assert_eq(norm.extent().byte_length(), 16L)
  let output_reference = try! @plan.TensorRef::new(5)
  let output = rank.tensor(output_reference).unwrap()
  assert_eq(output.placement(), ShardColumns)
  assert_eq(output.extent().column_start(), 4L)
  assert_eq(output.extent().column_count(), 4L)
  assert_eq(output.transfer_recipe().segment_count(), 8L)
  let first = try! output.transfer_recipe().segment(0L)
  let second = try! output.transfer_recipe().segment(1L)
  assert_eq(first.byte_length(), 8L)
  assert_eq(
    second.source_payload_offset_bytes() - first.source_payload_offset_bytes(),
    16L,
  )
  assert_eq(
    second.destination_offset_bytes() - first.destination_offset_bytes(),
    8L,
  )
}

///|
test "collective identities are plan ordered and rank invariant" {
  let (model, bindings) = tp_fixture(layers=2)
  let left = tp_plan(model, bindings, 0, 2)
  let right = tp_plan(model, bindings, 1, 2)
  assert_eq(left.collective_count(), 6)
  assert_eq(left.collective_count(), right.collective_count())
  for index in 0..<left.collective_count() {
    let a = left.collective(index).unwrap()
    let b = right.collective(index).unwrap()
    assert_eq(a.sequence().as_int(), index + 1)
    assert_eq(a.sequence(), b.sequence())
    assert_eq(a.operation(), b.operation())
    assert_eq(a.kind(), b.kind())
    assert_eq(a.full_width(), b.full_width())
    assert_eq(a.local_width(), b.local_width())
  }
  assert_eq(left.collective(0).unwrap().sequence().as_int(), 1)
  assert_eq(left.collective(0).unwrap().operation().as_int(), 0)
  assert_eq(left.collective(1).unwrap().operation().as_int(), 5)
  assert_eq(left.collective(2).unwrap().operation().as_int(), 8)
  assert_eq(left.collective(5).unwrap().kind(), LastAxisAllGather)
  assert_eq(left.collective(5).unwrap().operation().as_int(), 20)
  assert_eq(left.collective(5).unwrap().sequence().as_int(), 6)
}

///|
test "world size one is the deterministic no-collective oracle" {
  let (model, bindings) = tp_fixture()
  let first = tp_plan(model, bindings, 0, 1)
  let replay = tp_plan(model, bindings, 0, 1)
  assert_eq(first, replay)
  assert_eq(first.collective_count(), 0)
  for reference in 0..<first.tensor_count() {
    let tensor_reference = try! @plan.TensorRef::new(reference)
    let tensor = first.tensor(tensor_reference).unwrap()
    assert_eq(
      tensor.extent().byte_length(),
      tensor.extent().full_rows() * tensor.extent().full_columns() * 2L,
    )
  }
}

///|
test "rank plan rejects topology divisibility arena and recipe hostiles" {
  let (model, bindings) = tp_fixture()
  let (ceiling_model, ceiling_bindings) = tp_fixture(
    vocabulary=32,
    hidden=32,
    intermediate=32,
    attention_heads=16,
    kv_heads=16,
  )
  assert_eq(
    tp_plan(ceiling_model, ceiling_bindings, 15, 16).topology().world_size(),
    16,
  )
  for topology in [(2, 2), (0, 17)] {
    try tp_plan(model, bindings, topology.0, topology.1) catch {
      InvalidTopology => ()
      _ => fail("unexpected topology error")
    } noraise {
      _ => fail("unsupported topology accepted")
    }
  }
  try tp_plan(model, bindings, 0, 4) catch {
    NonDivisible(KeyValueHeads) => ()
    _ => fail("unexpected divisibility error")
  } noraise {
    _ => fail("non-divisible KV heads accepted")
  }
  try
    build_rank_plan(
      model,
      bindings,
      rank=0,
      world_size=2,
      alignment_bytes=3L,
      max_arena_bytes=1048576L,
    )
  catch {
    InvalidAlignment => ()
    _ => fail("unexpected alignment error")
  } noraise {
    _ => fail("invalid alignment accepted")
  }
  try
    build_rank_plan(
      model,
      bindings,
      rank=0,
      world_size=2,
      alignment_bytes=16L,
      max_arena_bytes=127L,
    )
  catch {
    ArenaTooLarge => ()
    _ => fail("unexpected arena error")
  } noraise {
    _ => fail("undersized arena accepted")
  }
  let recipe_reference = try! @plan.TensorRef::new(5)
  let recipe = tp_plan(model, bindings, 0, 2)
    .tensor(recipe_reference)
    .unwrap()
    .transfer_recipe()
  try recipe.segment(recipe.segment_count()) catch {
    InvalidTensorParallelPlan(rule=TransferSegment, index=5) => ()
    _ => fail("unexpected segment error")
  } noraise {
    _ => fail("out-of-range segment accepted")
  }
}

///|
test "rank plan rejects weight content substitution" {
  let (model, _) = tp_fixture()
  let (_, foreign) = tp_fixture(
    content="1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  try tp_plan(model, foreign, 0, 2) catch {
    IdentityMismatch => ()
    _ => fail("unexpected identity error")
  } noraise {
    _ => fail("foreign weight identity accepted")
  }
}

///|
test "generic rank plan rejects shard topology substitution" {
  let (model, bindings) = tp_fixture()
  let source = tp_plan(model, bindings, 0, 2)
  let tensors : Array[@tensor_parallel_plan.TensorRankPlacement] = []
  for index in 0..<source.tensor_count() {
    let reference = try! @plan.TensorRef::new(index)
    tensors.push(source.tensor(reference).unwrap())
  }
  let operations : Array[@tensor_parallel_plan.TensorParallelOperationPlan] = []
  for index in 0..<source.operation_count() {
    let operation = try! @plan.OperationId::new(index)
    operations.push(source.operation(operation).unwrap())
  }
  let topology = try! @tensor_parallel_plan.TensorParallelTopology::new(
    rank=0,
    world_size=4,
  )
  try
    @tensor_parallel_plan.TensorParallelRankPlan::new(
      model~,
      topology~,
      alignment_bytes=source.alignment_bytes(),
      arena_bytes=source.arena_bytes(),
      materialized_bytes=source.materialized_bytes(),
      tensors~,
      operations~,
      collectives=[],
    )
  catch {
    InvalidTensorParallelPlan(rule=Extent, ..) => ()
    _ => fail("unexpected substituted shard topology error")
  } noraise {
    _ => fail("rank plan accepted shard geometry from another topology")
  }
}

///|
test "generic rank plan rejects all gather world substitution" {
  let (model, bindings) = tp_fixture(
    vocabulary=32,
    hidden=8,
    intermediate=12,
    attention_heads=4,
    kv_heads=4,
  )
  let source = tp_plan(model, bindings, 0, 4)
  let tensors : Array[@tensor_parallel_plan.TensorRankPlacement] = []
  for index in 0..<source.tensor_count() {
    let reference = try! @plan.TensorRef::new(index)
    tensors.push(source.tensor(reference).unwrap())
  }
  let collectives : Array[@tensor_parallel_plan.TensorCollectiveSite] = []
  for index in 0..<source.collective_count() {
    let site = source.collective(index).unwrap()
    if site.kind() == LastAxisAllGather {
      collectives.push(
        try! @tensor_parallel_plan.TensorCollectiveSite::new(
          sequence=site.sequence().as_int(),
          operation=site.operation(),
          kind=LastAxisAllGather,
          full_width=32,
          local_width=16,
          world_size=2,
        ),
      )
    } else {
      collectives.push(site)
    }
  }
  let operations : Array[@tensor_parallel_plan.TensorParallelOperationPlan] = []
  for index in 0..<source.operation_count() {
    let operation = try! @plan.OperationId::new(index)
    operations.push(source.operation(operation).unwrap())
  }
  try
    @tensor_parallel_plan.TensorParallelRankPlan::new(
      model~,
      topology=source.topology(),
      alignment_bytes=source.alignment_bytes(),
      arena_bytes=source.arena_bytes(),
      materialized_bytes=source.materialized_bytes(),
      tensors~,
      operations~,
      collectives~,
    )
  catch {
    InvalidTensorParallelPlan(rule=CollectiveGeometry, ..) => ()
    _ => fail("unexpected substituted all-gather topology error")
  } noraise {
    _ => fail("rank plan accepted all-gather geometry from another world")
  }
}
