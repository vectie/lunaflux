///|
/// Derives existing per-rank KV bindings from a compiled two-node plan.
/// Physical reservations remain separate even when both GPUs use ordinal zero.
pub fn admit_from_two_node_plan(
  model_plan : @plan.ModelPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  compiled : @two_node_plan.TwoNodePlan,
) -> TensorParallelKvPlan raise TensorParallelKvPlanError {
  if model_plan.identity() != compiled.model_identity() {
    raise rejected(ModelIdentity, -1)
  }
  let devices : Array[RankDeviceFacts] = []
  let ranks : Array[RankRuntimeFacts] = []
  for rank in 0..<2 {
    guard compiled.topology().rank(rank) is Some(device) &&
      compiled.rank_budget(rank) is Some(budget) else {
      raise rejected(BindingCount, rank)
    }
    devices.push({
      identity: model_plan.identity(),
      rank,
      world_size: 2,
      device_ordinal: device.device_ordinal(),
      target: device.target(),
    })
    ranks.push({
      rank,
      world_size: 2,
      process_ordinal: device.device_ordinal(),
      target: device.target(),
      kv_reserved_bytes: budget.kv_bytes(),
    })
  }
  let result = admit_facts(
    model_plan.identity(),
    model_plan.kv_cache_geometry(),
    generation.value(),
    devices,
    compiled.topology().target(),
    ranks,
    compiled.tokens_per_page(),
    compiled.page_count(),
  )
  if result.local_layout != compiled.kv_layout() {
    raise rejected(KvHeadDivisibility, -1)
  }
  result
}
