///|
const TP_MAX_INT64 : Int64 = 9223372036854775807L

///|
fn tensor_reference(
  index : Int,
) -> @plan.TensorRef raise TensorParallelLaunchContractError {
  @plan.TensorRef::new(index) catch {
    _ => raise invalid(TensorExtent, operation=index)
  }
}

///|
fn validate_rank_plan(
  model : @plan.ModelPlan,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  expected_rank : Int,
  expected_world_size : Int,
) -> Unit raise TensorParallelLaunchContractError {
  let topology = rank_plan.topology()
  if rank_plan.identity() != model.identity() {
    raise invalid(Identity)
  }
  if expected_world_size <= 1 || topology.world_size() != expected_world_size {
    raise invalid(WorldSize)
  }
  if expected_rank < 0 ||
    expected_rank >= expected_world_size ||
    topology.rank() != expected_rank {
    raise invalid(Rank)
  }
  if rank_plan.tensor_count() <= 0 {
    raise invalid(TensorCount)
  }
  let referenced = FixedArray::make(rank_plan.tensor_count(), false)
  for operation in model.operations() {
    for input in operation.inputs() {
      match input {
        Tensor(reference) => {
          let index = reference.as_int()
          guard rank_plan.tensor(reference) is Some(_) else {
            raise invalid(TensorExtent, operation=operation.id().as_int())
          }
          if index < 0 || index >= referenced.length() {
            raise invalid(TensorExtent, operation=operation.id().as_int())
          }
          referenced[index] = true
        }
        _ => ()
      }
    }
  }
  let mut previous_end = 0L
  let mut materialized = 0L
  for index in 0..<rank_plan.tensor_count() {
    if !referenced[index] {
      raise invalid(TensorCount, operation=index)
    }
    guard rank_plan.tensor(tensor_reference(index)) is Some(placement) else {
      raise invalid(TensorExtent, operation=index)
    }
    let extent = placement.extent()
    let region = placement.region()
    if placement.reference().as_int() != index ||
      region.offset_bytes() < previous_end ||
      region.byte_length() <= 0L ||
      region.offset_bytes() % rank_plan.alignment_bytes() != 0L ||
      region.offset_bytes() > TP_MAX_INT64 - region.byte_length() ||
      materialized > TP_MAX_INT64 - region.byte_length() {
      raise invalid(TensorRegion, operation=index)
    }
    match placement.placement() {
      Replicated => ()
      ShardRows =>
        if extent.row_count() >= extent.full_rows() {
          raise invalid(FullTensorReplication, operation=index)
        }
      ShardColumns =>
        if extent.column_count() >= extent.full_columns() {
          raise invalid(FullTensorReplication, operation=index)
        }
    }
    previous_end = region.end_offset_bytes()
    materialized += region.byte_length()
  }
  if materialized != rank_plan.materialized_bytes() ||
    previous_end != rank_plan.arena_bytes() {
    raise invalid(TensorRegion)
  }
}

///|
fn rank_weight_bytes(
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  reference : @plan.TensorRef,
) -> Int64 raise TensorParallelLaunchContractError {
  guard rank_plan.tensor(reference) is Some(placement) else {
    raise invalid(TensorExtent, operation=reference.as_int())
  }
  placement.region().byte_length()
}
