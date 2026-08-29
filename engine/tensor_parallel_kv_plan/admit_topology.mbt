///|
/// Derives the exact full rank-ordered KV plan from admitted topology and
/// runtime capacity evidence without constructing or inspecting other ranks'
/// weight/device plans.
pub fn admit_from_topology(
  model_plan : @plan.ModelPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  topology : @device_topology.AdmittedLocalTopology,
  runtime : @tensor_parallel_admission.TensorParallelRuntimeAdmission,
  tokens_per_page~ : Int,
  total_page_count~ : Int,
) -> TensorParallelKvPlan raise TensorParallelKvPlanError {
  let world_size = topology.world_size()
  if runtime.world_size() != world_size || runtime.target() != topology.target() {
    raise rejected(BindingCount, -1)
  }
  let devices : Array[RankDeviceFacts] = []
  let ranks : Array[RankRuntimeFacts] = []
  for rank in 0..<world_size {
    guard topology.rank(rank) is Some(device) &&
      runtime.rank_capacity(rank) is Some(capacity) else {
      raise rejected(BindingCount, rank)
    }
    if capacity.rank() != rank ||
      capacity.world_size() != world_size ||
      capacity.process_ordinal() != device.ordinal() ||
      capacity.target() != device.target() ||
      capacity.kv_reserved_bytes() <= 0L {
      raise rejected(DeviceOrdinal, rank)
    }
    devices.push({
      identity: model_plan.identity(),
      rank,
      world_size,
      device_ordinal: device.ordinal(),
      target: device.target(),
    })
    ranks.push({
      rank,
      world_size,
      process_ordinal: capacity.process_ordinal(),
      target: capacity.target(),
      kv_reserved_bytes: capacity.kv_reserved_bytes(),
    })
  }
  admit_facts(
    model_plan.identity(),
    model_plan.kv_cache_geometry(),
    generation.value(),
    devices,
    runtime.target(),
    ranks,
    tokens_per_page,
    total_page_count,
  )
}
