///|
fn same_rendezvous_id(
  left : @device.TensorCollectiveGroupId,
  right : @device.TensorCollectiveGroupId,
) -> Bool {
  let left_bytes = left.to_bytes()
  let right_bytes = right.to_bytes()
  if left_bytes.length() != 128 || right_bytes.length() != 128 {
    return false
  }
  for index in 0..<128 {
    if left_bytes[index] != right_bytes[index] {
      return false
    }
  }
  true
}

///|
/// Independently authenticates a decoded rank envelope against rank-local
/// startup evidence. The returned contract is still authority-free; opening a
/// communicator remains the private device owner's responsibility.
pub fn TensorParallelRankBootstrapEnvelope::authenticate_local(
  self : TensorParallelRankBootstrapEnvelope,
  model_plan : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  group_generation : @rank_group_protocol.RankGroupGeneration,
  topology : @device_topology.AdmittedLocalTopology,
  startup : @worker_wire.WorkerStartupContract,
  inspection : @device_materialize.DeviceShardedWeightFileInspection,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  execution_plan : @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  kv_binding : @tensor_parallel_kv_plan.TensorParallelKvRankBinding,
  expected_group_digest~ : TensorParallelGroupBootstrapDigest,
  expected_rendezvous_id~ : @device.TensorCollectiveGroupId,
  expected_collective_runtime_version_code~ : Int,
) -> @device.TensorCollectiveContract raise TensorParallelWorkerBootstrapError {
  let rank = self.rank
  if self.identity != model_plan.identity() {
    raise rejected(ModelIdentity, rank)
  }
  if self.model_generation_value != model_generation.value() {
    raise rejected(ModelGeneration, rank)
  }
  if self.group_generation_value != group_generation.value() {
    raise rejected(GroupGeneration, rank)
  }
  if self.group_digest != expected_group_digest {
    raise rejected(Digest, rank)
  }
  if expected_collective_runtime_version_code <= 0 ||
    self.collective_runtime_version_code !=
    expected_collective_runtime_version_code {
    raise rejected(CollectiveContract, rank)
  }
  if self.world_size != topology.world_size() {
    raise rejected(WorldSize, rank)
  }
  guard topology.rank(rank) is Some(topology_rank) else {
    raise rejected(Rank, rank)
  }
  if self.device_ordinal != topology_rank.ordinal() {
    raise rejected(DeviceOrdinal, rank)
  }
  if self.topology_digest != digest_topology(topology) {
    raise rejected(Topology, rank)
  }
  require_worker_contract(
    startup,
    self.identity,
    self.model_generation_value,
    rank,
    self.device_ordinal,
    startup,
  )
  if self.worker_digest != digest_worker_contract(startup) {
    raise rejected(WorkerContract, rank)
  }
  let rank_device = @tensor_parallel_device_plan.TensorParallelRankDevice::from_local_topology(
    topology,
    rank~,
  ) catch {
    _ => raise rejected(Topology, rank)
  }
  let expected_device_plan = @tensor_parallel_device_plan.build(
    model_plan,
    inspection.rank_plan(),
    inspection,
    rank_device,
    topology.target(),
  ) catch {
    _ => raise rejected(DevicePlan, rank)
  }
  if expected_device_plan != device_plan ||
    self.device_plan_digest != digest_device_plan(device_plan) {
    raise rejected(DevicePlan, rank)
  }
  require_execution_plan(
    execution_plan,
    self.identity,
    rank,
    self.world_size,
    self.device_ordinal,
    topology.target(),
  )
  if self.execution_plan_digest_sha256 != execution_plan.digest().as_sha256() {
    raise rejected(ExecutionPlan, rank)
  }
  let collective = @tensor_parallel_collective.authenticate_rank_plan(
    inspection.rank_plan(),
    expected_identity=self.identity,
    max_query_tokens=startup.worker_limits().max_plan_tokens(),
  ) catch {
    _ => raise rejected(CollectiveContract, rank)
  }
  if collective.rank() != rank ||
    collective.world_size() != self.world_size ||
    collective.digest() != self.collective_digest {
    raise rejected(CollectiveContract, rank)
  }
  require_kv_binding(
    kv_binding,
    self.identity,
    self.model_generation_value,
    rank,
    self.world_size,
    self.device_ordinal,
    topology.target(),
  )
  let expected_kv_digest = kv_binding_digest_from_hex(
    kv_binding.digest().as_sha256(),
  )
  if self.kv_digest != expected_kv_digest {
    raise rejected(KvContract, rank)
  }
  if !same_rendezvous_id(self.rendezvous_id, expected_rendezvous_id) {
    raise rejected(RendezvousId, rank)
  }
  collective
}
