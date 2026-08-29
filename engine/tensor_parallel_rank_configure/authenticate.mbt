///|
fn configure_bindings_equal(
  left : @rank_group_wire.RankGroupWireBinding,
  right : @rank_group_wire.RankGroupWireBinding,
) -> Bool {
  left.identity() == right.identity() &&
  left.model_generation_value() == right.model_generation_value() &&
  left.group_generation_value() == right.group_generation_value() &&
  left.rank() == right.rank() &&
  left.world_size() == right.world_size() &&
  left.group_digest() == right.group_digest() &&
  left.topology_digest() == right.topology_digest() &&
  left.worker_digest() == right.worker_digest() &&
  left.device_plan_digest() == right.device_plan_digest() &&
  left.execution_plan_digest() == right.execution_plan_digest() &&
  left.collective_digest() == right.collective_digest() &&
  left.kv_digest() == right.kv_digest()
}

///|
fn validate_configure_claims(
  binding : @rank_group_wire.RankGroupWireBinding,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  topology : @device_topology.LocalTopologyDeclaration,
  envelope : @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope,
) -> Unit raise TensorParallelRankConfigureError {
  if !source.has_valid_tensor_parallel_recipe_binding() {
    raise Invalid(BootstrapSource)
  }
  guard source.tensor_parallel_policy() is Some(policy) else {
    raise Invalid(BootstrapSource)
  }
  if policy.rank_count() != topology.rank_count() ||
    policy.rank_count() != binding.world_size() {
    raise Invalid(WorldSize)
  }
  for rank in 0..<policy.rank_count() {
    guard policy.rank(rank) is Some(claim) &&
      topology.rank(rank) is Some(topology_rank) else {
      raise Invalid(Topology)
    }
    if claim.rank() != rank || claim.device_ordinal() != topology_rank.ordinal() {
      raise Invalid(Topology)
    }
  }
  let collective_runtime_version_code = envelope.collective_runtime_version_code()
  if collective_runtime_version_code <
    policy.minimum_collective_runtime_version_code() ||
    collective_runtime_version_code >
    policy.maximum_collective_runtime_version_code() {
    raise Invalid(RankEnvelope)
  }
  if startup.identity() != binding.identity() ||
    envelope.identity() != binding.identity() {
    raise Invalid(ModelIdentity)
  }
  if startup.model_generation_value() != binding.model_generation_value() ||
    envelope.model_generation_value() != binding.model_generation_value() {
    raise Invalid(ModelGeneration)
  }
  if envelope.group_generation_value() != binding.group_generation_value() {
    raise Invalid(GroupGeneration)
  }
  if envelope.rank() != binding.rank() {
    raise Invalid(Rank)
  }
  if envelope.world_size() != binding.world_size() ||
    topology.rank_count() != binding.world_size() {
    raise Invalid(WorldSize)
  }
  guard topology.rank(binding.rank()) is Some(selected) else {
    raise Invalid(Rank)
  }
  if startup.device_ordinal() != envelope.device_ordinal() ||
    startup.device_ordinal() != selected.ordinal() {
    raise Invalid(DeviceOrdinal)
  }
  if startup.bootstrap_source_digest() != source.digest() {
    raise Invalid(BootstrapSource)
  }
  let worker_digest = @tensor_parallel_worker_bootstrap.worker_contract_digest(
    startup,
  ) catch {
    _ => raise Invalid(Startup)
  }
  if worker_digest != envelope.worker_digest() ||
    worker_digest.as_sha256() != binding.worker_digest() {
    raise Invalid(Startup)
  }
  if envelope.group_digest().as_sha256() != binding.group_digest() ||
    envelope.topology_digest().as_sha256() != binding.topology_digest() ||
    envelope.device_plan_digest().as_sha256() != binding.device_plan_digest() ||
    envelope.execution_plan_digest_sha256() != binding.execution_plan_digest() ||
    envelope.collective_digest().as_hex() != binding.collective_digest() ||
    envelope.kv_digest().as_sha256() != binding.kv_digest() {
    raise Invalid(Binding)
  }
}
