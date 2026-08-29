///|
fn build_authenticated(
  model_plan : @plan.ModelPlan,
  planned_rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  materialized_rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  device : TensorParallelRankDevice,
  expected_target : @catalog.DeviceTarget,
) -> TensorParallelDevicePlan raise TensorParallelDevicePlanError {
  if model_plan.identity() != planned_rank_plan.identity() {
    raise invalid(ModelIdentity, 0)
  }
  if planned_rank_plan != materialized_rank_plan {
    raise invalid(MaterializedRankPlan, 0)
  }
  let topology = planned_rank_plan.topology()
  if topology.rank() != device.rank {
    raise invalid(Rank, topology.rank())
  }
  if topology.world_size() != device.world_size {
    raise invalid(WorldSize, topology.world_size())
  }
  if device.target != expected_target {
    raise invalid(DeviceTarget, device.rank)
  }
  if planned_rank_plan.tensor_count() <= 0 {
    raise invalid(TensorCount, planned_rank_plan.tensor_count())
  }
  let operations = validate_operation_layout(model_plan, planned_rank_plan)
  validate_region_order(planned_rank_plan)
  let tensors : Array[TensorParallelMatrixView] = []
  for index in 0..<planned_rank_plan.tensor_count() {
    tensors.push(require_matrix(planned_rank_plan, index))
  }
  let collectives = validate_collectives(planned_rank_plan, operations)
  {
    identity: model_plan.identity(),
    device,
    arena_bytes: planned_rank_plan.arena_bytes(),
    materialized_bytes: planned_rank_plan.materialized_bytes(),
    tensors: ReadOnlyArray::from_array(tensors),
    operations,
    collectives,
  }
}

///|
/// Authenticates semantic, file-bound shard, rank, world, and target evidence
/// before deriving an immutable local execution view. No backend handle or
/// allocation authority is retained.
pub fn build(
  model_plan : @plan.ModelPlan,
  planned_rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  materialized : @device_materialize.DeviceShardedWeightFileInspection,
  device : TensorParallelRankDevice,
  expected_target : @catalog.DeviceTarget,
) -> TensorParallelDevicePlan raise TensorParallelDevicePlanError {
  build_authenticated(
    model_plan,
    planned_rank_plan,
    materialized.rank_plan(),
    device,
    expected_target,
  )
}
