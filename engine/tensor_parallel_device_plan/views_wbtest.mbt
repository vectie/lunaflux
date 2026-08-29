///|
fn tpdp_assert_local_shard(view : TensorParallelMatrixView) -> Unit raise {
  assert_true(
    view.local_rows() * view.local_columns() <
    view.full_rows() * view.full_columns(),
  )
  assert_true(view.region_byte_length() > 0L)
}

///|
fn tpdp_tensor(
  plan : TensorParallelDevicePlan,
  index : Int,
) -> TensorParallelMatrixView raise {
  guard plan.tensor(@plan.TensorRef::new(index)) is Some(view) else {
    fail("ordered tensor view is missing")
  }
  view
}

///|
test "world one device plan is the full-shape oracle without collectives" {
  let (model_plan, bindings) = tpdp_fixture()
  let rank_plan = tpdp_rank_plan(model_plan, bindings, 0, 1)
  let target = tpdp_target()
  let result = build_authenticated(
    model_plan,
    rank_plan,
    rank_plan,
    TensorParallelRankDevice::single(device_ordinal=3, target~),
    target,
  )
  assert_eq(result.identity(), model_plan.identity())
  assert_eq(result.rank(), 0)
  assert_eq(result.world_size(), 1)
  assert_eq(result.device_ordinal(), 3)
  assert_eq(result.target(), target)
  assert_eq(result.collective_count(), 0)
  assert_eq(result.materialized_bytes(), 424L)
  assert_eq(result.operation_count(), model_plan.operations().length())
  for index in 0..<result.tensor_count() {
    let view = tpdp_tensor(result, index)
    assert_eq(view.local_rows(), view.full_rows())
    assert_eq(view.local_columns(), view.full_columns())
  }
}

///|
test "multi rank views expose exact local projection and logit shapes" {
  let (model_plan, bindings) = tpdp_fixture()
  let rank_plan = tpdp_rank_plan(model_plan, bindings, 1, 2)
  let target = tpdp_target()
  let topology = tpdp_two_rank_topology(target)
  let device = TensorParallelRankDevice::from_local_topology(topology, rank=1)
  let result = build_authenticated(
    model_plan, rank_plan, rank_plan, device, target,
  )
  assert_eq(result.rank(), 1)
  assert_eq(result.world_size(), 2)
  assert_eq(result.device_ordinal(), 5)
  assert_eq(tpdp_tensor(result, 0).local_rows(), 4L)
  assert_eq(tpdp_tensor(result, 0).local_row_start(), 4L)
  assert_eq(tpdp_tensor(result, 11).local_rows(), 4L)
  assert_eq(tpdp_tensor(result, 11).local_row_start(), 4L)
  assert_eq(tpdp_tensor(result, 2).local_rows(), 2L)
  assert_eq(tpdp_tensor(result, 2).local_columns(), 4L)
  assert_eq(tpdp_tensor(result, 3).local_rows(), 2L)
  assert_eq(tpdp_tensor(result, 4).local_rows(), 2L)
  assert_eq(tpdp_tensor(result, 5).local_rows(), 4L)
  assert_eq(tpdp_tensor(result, 5).local_columns(), 2L)
  assert_eq(tpdp_tensor(result, 5).local_column_start(), 2L)
  assert_eq(tpdp_tensor(result, 7).local_rows(), 3L)
  assert_eq(tpdp_tensor(result, 8).local_rows(), 3L)
  assert_eq(tpdp_tensor(result, 9).local_rows(), 4L)
  assert_eq(tpdp_tensor(result, 9).local_columns(), 3L)
  assert_eq(tpdp_tensor(result, 9).local_column_start(), 3L)
}

///|
test "multi rank plan never references a complete sharded tensor region" {
  let (model_plan, bindings) = tpdp_fixture()
  let target = tpdp_target()
  let topology = tpdp_two_rank_topology(target)
  for rank in 0..<2 {
    let rank_plan = tpdp_rank_plan(model_plan, bindings, rank, 2)
    let result = build_authenticated(
      model_plan,
      rank_plan,
      rank_plan,
      TensorParallelRankDevice::from_local_topology(topology, rank~),
      target,
    )
    for index in 0..<result.tensor_count() {
      let view = tpdp_tensor(result, index)
      if view.placement() != Replicated {
        tpdp_assert_local_shard(view)
      }
    }
    assert_true(result.materialized_bytes() < 424L)
  }
}

///|
test "generic tensor and operation views are exact ordered sources" {
  let (model_plan, bindings) = tpdp_fixture()
  let rank_plan = tpdp_rank_plan(model_plan, bindings, 1, 2)
  let target = tpdp_target()
  let result = build_authenticated(
    model_plan,
    rank_plan,
    rank_plan,
    TensorParallelRankDevice::from_local_topology(
      tpdp_two_rank_topology(target),
      rank=1,
    ),
    target,
  )
  assert_eq(result.tensor_count(), rank_plan.tensor_count())
  for index in 0..<result.tensor_count() {
    guard result.tensor(@plan.TensorRef::new(index)) is Some(view) else {
      fail("ordered tensor view is missing")
    }
    assert_eq(view.reference().as_int(), index)
  }
  assert_eq(result.operation_count(), model_plan.operations().length())
  for index, semantic in model_plan.operations() {
    guard result.operation(semantic.id()) is Some(geometry) else {
      fail("ordered operation geometry is missing")
    }
    assert_eq(geometry.operation().as_int(), index)
    assert_eq(geometry.kind(), semantic.kind())
    assert_eq(geometry.capability(), semantic.required_capability())
  }
}

///|
test "collective sites preserve first and last semantic identities" {
  let (model_plan, bindings) = tpdp_fixture()
  let rank_plan = tpdp_rank_plan(model_plan, bindings, 0, 2)
  let target = tpdp_target()
  let result = build_authenticated(
    model_plan,
    rank_plan,
    rank_plan,
    TensorParallelRankDevice::from_local_topology(
      tpdp_two_rank_topology(target),
      rank=0,
    ),
    target,
  )
  assert_eq(result.collective_count(), 4)
  guard result.collective(0) is Some(first) &&
    result.collective(3) is Some(last) else {
    fail("stable collective endpoints are missing")
  }
  assert_eq(first.sequence().as_int(), 1)
  assert_eq(first.operation().as_int(), 0)
  assert_eq(first.kind(), SumAllReduce)
  assert_eq(last.sequence().as_int(), 4)
  assert_eq(last.operation().as_int(), 11)
  assert_eq(last.kind(), LastAxisAllGather)
  assert_eq(last.full_width(), 8)
  assert_eq(last.local_width(), 4)
}
