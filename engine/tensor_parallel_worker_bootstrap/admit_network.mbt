///|
fn network_topology_digest(
  compiled : @two_node_plan.TwoNodePlan,
  remote_group_digest : String,
) -> TensorParallelTopologyDigest raise TensorParallelWorkerBootstrapError {
  let encoder : BootstrapDigestEncoder = { bytes: [], }
  encoder.append_string("lunaflux.network-execution-binding.v1")
  encoder.append_string(compiled.digest().as_sha256())
  encoder.append_string(remote_group_digest)
  topology_digest_from_hex(encoder.finish())
}

///|
/// Joins independently pinned remote startup contracts to the existing rank
/// execution evidence. Produces only immutable envelopes, never live readiness.
pub fn admit_network(
  model : @plan.ModelPlan,
  compiled : @two_node_plan.TwoNodePlan,
  remote : @remote_startup.RemoteGroupStartup,
  inspections : ArrayView[@device_materialize.DeviceShardedWeightFileInspection],
  devices : ArrayView[@tensor_parallel_device_plan.TensorParallelDevicePlan],
  executions : ArrayView[
    @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  ],
  kv : @tensor_parallel_kv_plan.TensorParallelKvPlan,
  collective_runtime_version_code~ : Int,
) -> TensorParallelGroupBootstrap raise TensorParallelWorkerBootstrapError {
  if inspections.length() != 2 ||
    devices.length() != 2 ||
    executions.length() != 2 {
    raise rejected(IncompleteRankSet, -1)
  }
  let claims : Array[RankBootstrapClaim] = []
  for rank in 0..<2 {
    guard remote.rank(rank) is Some(startup) else {
      raise rejected(IncompleteRankSet, rank)
    }
    claims.push(
      network_rank_claim(
        model,
        compiled,
        startup,
        inspections[rank],
        devices[rank],
        executions[rank],
        kv,
      ),
    )
  }
  guard remote.rank(0) is Some(first) else {
    raise rejected(IncompleteRankSet, 0)
  }
  admit_claims(
    model.identity(),
    first.worker().model_generation_value(),
    first.group_generation_value(),
    network_topology_digest(compiled, remote.digest_sha256()),
    collective_runtime_version_code,
    first.rendezvous(),
    2,
    claims,
    template_digest_sha256=remote.digest_sha256(),
  )
}

///|
/// Cross-checks an envelope against locally rebuilt file and AOT evidence.
/// `expected_group_digest` must come from the independently pinned full group.
pub fn TensorParallelRankBootstrapEnvelope::authenticate_network(
  self : TensorParallelRankBootstrapEnvelope,
  model : @plan.ModelPlan,
  compiled : @two_node_plan.TwoNodePlan,
  remote : @remote_startup.RemoteRankStartup,
  inspection : @device_materialize.DeviceShardedWeightFileInspection,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  execution : @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  kv : @tensor_parallel_kv_plan.TensorParallelKvPlan,
  expected_group_digest~ : TensorParallelGroupBootstrapDigest,
  expected_collective_runtime_version_code~ : Int,
) -> @device.TensorCollectiveContract raise TensorParallelWorkerBootstrapError {
  let claim = network_rank_claim(
    model, compiled, remote, inspection, device_plan, execution, kv,
  )
  let rank = claim.rank
  if self.identity != claim.identity ||
    self.model_generation_value != claim.model_generation_value {
    raise rejected(ModelIdentity, rank)
  }
  if self.group_generation_value != claim.group_generation_value ||
    self.group_digest != expected_group_digest {
    raise rejected(GroupGeneration, rank)
  }
  if self.rank != rank ||
    self.world_size != 2 ||
    self.device_ordinal != claim.device_ordinal ||
    self.topology_digest !=
    network_topology_digest(compiled, remote.group_digest_sha256()) {
    raise rejected(Topology, rank)
  }
  if expected_collective_runtime_version_code <= 0 ||
    self.collective_runtime_version_code !=
    expected_collective_runtime_version_code {
    raise rejected(CollectiveContract, rank)
  }
  if self.worker_digest != claim.worker_digest ||
    self.device_plan_digest != claim.device_plan_digest ||
    self.execution_plan_digest_sha256 != claim.execution_plan_digest_sha256 ||
    self.collective_digest != claim.collective_digest ||
    self.kv_digest != claim.kv_digest {
    raise rejected(Digest, rank)
  }
  if !same_rendezvous_id(self.rendezvous_id, remote.rendezvous()) {
    raise rejected(RendezvousId, rank)
  }
  @tensor_parallel_collective.authenticate_rank_plan(
    inspection.rank_plan(),
    expected_identity=model.identity(),
    max_query_tokens=remote.worker().worker_limits().max_plan_tokens(),
  ) catch {
    _ => raise rejected(CollectiveContract, rank)
  }
}
