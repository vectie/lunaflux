///|
fn require_worker_contract(
  contract : @worker_wire.WorkerStartupContract,
  identity : @spec.ModelIdentity,
  model_generation_value : UInt64,
  rank : Int,
  device_ordinal : Int,
  reference : @worker_wire.WorkerStartupContract,
) -> Unit raise TensorParallelWorkerBootstrapError {
  if contract.identity() != identity {
    raise rejected(ModelIdentity, rank)
  }
  if contract.model_generation_value() != model_generation_value {
    raise rejected(ModelGeneration, rank)
  }
  if contract.device_ordinal() != device_ordinal {
    raise rejected(DeviceOrdinal, rank)
  }
  if contract.worker_limits() != reference.worker_limits() ||
    contract.inference_limits() != reference.inference_limits() ||
    contract.bootstrap_source_digest() != reference.bootstrap_source_digest() ||
    contract.has_predecessor() != reference.has_predecessor() ||
    contract.predecessor_value() != reference.predecessor_value() {
    raise rejected(WorkerContract, rank)
  }
}

///|
fn require_kv_binding(
  binding : @tensor_parallel_kv_plan.TensorParallelKvRankBinding,
  identity : @spec.ModelIdentity,
  model_generation_value : UInt64,
  rank : Int,
  world_size : Int,
  device_ordinal : Int,
  target : @catalog.DeviceTarget,
) -> Unit raise TensorParallelWorkerBootstrapError {
  if binding.identity() != identity {
    raise rejected(ModelIdentity, rank)
  }
  if binding.generation_value() != model_generation_value {
    raise rejected(ModelGeneration, rank)
  }
  if binding.rank() != rank || binding.world_size() != world_size {
    raise rejected(KvContract, rank)
  }
  if binding.process_ordinal() != device_ordinal || binding.target() != target {
    raise rejected(KvContract, rank)
  }
}

///|
fn require_execution_plan(
  execution_plan : @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  identity : @spec.ModelIdentity,
  rank : Int,
  world_size : Int,
  device_ordinal : Int,
  target : @catalog.DeviceTarget,
) -> Unit raise TensorParallelWorkerBootstrapError {
  if execution_plan.identity() != identity {
    raise rejected(ModelIdentity, rank)
  }
  if execution_plan.rank() != rank {
    raise rejected(Rank, rank)
  }
  if execution_plan.world_size() != world_size {
    raise rejected(WorldSize, rank)
  }
  if execution_plan.device_ordinal() != device_ordinal {
    raise rejected(DeviceOrdinal, rank)
  }
  if execution_plan.target() != target ||
    !is_lower_sha256(execution_plan.digest().as_sha256()) {
    raise rejected(ExecutionPlan, rank)
  }
}

///|
fn rank_claim(
  model_plan : @plan.ModelPlan,
  model_generation_value : UInt64,
  group_generation_value : UInt64,
  topology : @device_topology.AdmittedLocalTopology,
  rank : Int,
  startup : @worker_wire.WorkerStartupContract,
  reference_startup : @worker_wire.WorkerStartupContract,
  inspection : @device_materialize.DeviceShardedWeightFileInspection,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  execution_plan : @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  kv_plan : @tensor_parallel_kv_plan.TensorParallelKvPlan,
) -> RankBootstrapClaim raise TensorParallelWorkerBootstrapError {
  let identity = model_plan.identity()
  let world_size = topology.world_size()
  guard topology.rank(rank) is Some(topology_rank) else {
    raise rejected(IncompleteRankSet, rank)
  }
  let device_ordinal = topology_rank.ordinal()
  require_worker_contract(
    startup, identity, model_generation_value, rank, device_ordinal, reference_startup,
  )
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
  if expected_device_plan != device_plan {
    raise rejected(DevicePlan, rank)
  }
  require_execution_plan(
    execution_plan,
    identity,
    rank,
    world_size,
    device_ordinal,
    topology.target(),
  )
  let collective = @tensor_parallel_collective.authenticate_rank_plan(
    inspection.rank_plan(),
    expected_identity=identity,
    max_query_tokens=startup.worker_limits().max_plan_tokens(),
  ) catch {
    _ => raise rejected(CollectiveContract, rank)
  }
  if collective.rank() != rank || collective.world_size() != world_size {
    raise rejected(CollectiveContract, rank)
  }
  let kv_binding = kv_plan.rank_binding(rank) catch {
    _ => raise rejected(KvContract, rank)
  }
  require_kv_binding(
    kv_binding,
    identity,
    model_generation_value,
    rank,
    world_size,
    device_ordinal,
    topology.target(),
  )
  {
    identity,
    model_generation_value,
    group_generation_value,
    rank,
    world_size,
    device_ordinal,
    worker_digest: digest_worker_contract(startup),
    device_plan_digest: digest_device_plan(device_plan),
    execution_plan_digest_sha256: execution_plan.digest().as_sha256(),
    collective_digest: collective.digest(),
    kv_digest: kv_binding_digest_from_hex(kv_binding.digest().as_sha256()),
  }
}

///|
fn order_and_validate_claims(
  identity : @spec.ModelIdentity,
  model_generation_value : UInt64,
  group_generation_value : UInt64,
  world_size : Int,
  claims : ArrayView[RankBootstrapClaim],
) -> FixedArray[RankBootstrapClaim] raise TensorParallelWorkerBootstrapError {
  if world_size < 2 || world_size > 16 {
    raise rejected(WorldSize, -1)
  }
  if claims.length() != world_size {
    raise rejected(IncompleteRankSet, -1)
  }
  let ordered = FixedArray::make(world_size, claims[0])
  let seen = FixedArray::make(world_size, false)
  for claim in claims {
    if claim.rank < 0 || claim.rank >= world_size {
      raise rejected(Rank, claim.rank)
    }
    if seen[claim.rank] {
      raise rejected(DuplicateRank, claim.rank)
    }
    if claim.identity != identity {
      raise rejected(ModelIdentity, claim.rank)
    }
    if claim.model_generation_value != model_generation_value {
      raise rejected(ModelGeneration, claim.rank)
    }
    if claim.group_generation_value != group_generation_value {
      raise rejected(GroupGeneration, claim.rank)
    }
    if claim.world_size != world_size {
      raise rejected(WorldSize, claim.rank)
    }
    if !is_lower_sha256(claim.execution_plan_digest_sha256) {
      raise rejected(ExecutionPlan, claim.rank)
    }
    seen[claim.rank] = true
    ordered[claim.rank] = claim
  }
  for rank in 0..<world_size {
    if !seen[rank] {
      raise rejected(IncompleteRankSet, rank)
    }
  }
  ordered
}

///|
fn admit_claims(
  identity : @spec.ModelIdentity,
  model_generation_value : UInt64,
  group_generation_value : UInt64,
  topology_digest : TensorParallelTopologyDigest,
  collective_runtime_version_code : Int,
  rendezvous_id : @device.TensorCollectiveGroupId,
  world_size : Int,
  claims : ArrayView[RankBootstrapClaim],
  template_digest_sha256? : StringView = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
) -> TensorParallelGroupBootstrap raise TensorParallelWorkerBootstrapError {
  if model_generation_value == 0UL ||
    !is_lower_sha256(template_digest_sha256) ||
    template_digest_sha256 == "0".repeat(64) {
    raise rejected(ModelGeneration, -1)
  }
  if group_generation_value == 0UL || collective_runtime_version_code <= 0 {
    raise rejected(GroupGeneration, -1)
  }
  let ordered = order_and_validate_claims(
    identity, model_generation_value, group_generation_value, world_size, claims,
  )
  let rendezvous_bytes = rendezvous_id.to_bytes()
  if rendezvous_bytes.length() != 128 {
    raise rejected(RendezvousId, -1)
  }
  let group_digest = digest_group(
    template_digest_sha256, identity, model_generation_value, group_generation_value,
    topology_digest, collective_runtime_version_code, rendezvous_bytes, ordered,
  )
  let envelopes : Array[TensorParallelRankBootstrapEnvelope] = []
  for claim in ordered {
    envelopes.push({
      identity,
      model_generation_value,
      group_generation_value,
      rank: claim.rank,
      world_size: claim.world_size,
      device_ordinal: claim.device_ordinal,
      collective_runtime_version_code,
      group_digest,
      topology_digest,
      worker_digest: claim.worker_digest,
      device_plan_digest: claim.device_plan_digest,
      execution_plan_digest_sha256: claim.execution_plan_digest_sha256,
      collective_digest: claim.collective_digest,
      kv_digest: claim.kv_digest,
      rendezvous_id,
    })
  }
  {
    identity,
    model_generation_value,
    group_generation_value,
    collective_runtime_version_code,
    topology_digest,
    digest: group_digest,
    template_digest_sha256: template_digest_sha256.to_owned(),
    envelopes: ReadOnlyArray::from_array(envelopes),
  }
}

///|
/// Cross-authenticates one complete same-host tensor-parallel startup set.
/// Physical plan-buffer identities remain local to `RankGroupOwner`; this
/// wire identity binds their model/group generations and exact worker
/// bootstrap contracts without serializing an owner pointer or capability.
pub fn admit_with_template(
  template_digest_sha256 : StringView,
  model_plan : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  group_generation : @rank_group_protocol.RankGroupGeneration,
  topology : @device_topology.AdmittedLocalTopology,
  source : @worker_wire.EncodedBootstrapSource,
  startups : ArrayView[@worker_wire.WorkerStartupContract],
  inspections : ArrayView[@device_materialize.DeviceShardedWeightFileInspection],
  device_plans : ArrayView[
    @tensor_parallel_device_plan.TensorParallelDevicePlan,
  ],
  execution_plans : ArrayView[
    @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  ],
  kv_plan : @tensor_parallel_kv_plan.TensorParallelKvPlan,
  rendezvous_id : @device.TensorCollectiveGroupId,
  collective_runtime_version_code~ : Int,
) -> TensorParallelGroupBootstrap raise TensorParallelWorkerBootstrapError {
  let world_size = topology.world_size()
  if world_size < 2 || world_size > 16 {
    raise rejected(WorldSize, -1)
  }
  if startups.length() != world_size ||
    inspections.length() != world_size ||
    device_plans.length() != world_size ||
    execution_plans.length() != world_size ||
    kv_plan.world_size() != world_size {
    raise rejected(IncompleteRankSet, -1)
  }
  if startups[0].bootstrap_source_digest() != source.digest() {
    raise rejected(WorkerContract, 0)
  }
  guard source.tensor_parallel_policy() is Some(policy) else {
    raise rejected(WorkerContract, -1)
  }
  if collective_runtime_version_code <
    policy.minimum_collective_runtime_version_code() ||
    collective_runtime_version_code >
    policy.maximum_collective_runtime_version_code() {
    raise rejected(CollectiveContract, -1)
  }
  let runtime = admit_runtime_policy(source, topology)
  let execution = source.execution()
  let expected_kv_plan = @tensor_parallel_kv_plan.admit_from_topology(
    model_plan,
    model_generation,
    topology,
    runtime,
    tokens_per_page=execution.tokens_per_page(),
    total_page_count=execution.total_page_count(),
  ) catch {
    _ => raise rejected(KvContract, -1)
  }
  if expected_kv_plan != kv_plan {
    raise rejected(KvContract, -1)
  }
  let identity = model_plan.identity()
  let model_generation_value = model_generation.value()
  let group_generation_value = group_generation.value()
  if kv_plan.identity() != identity {
    raise rejected(ModelIdentity, -1)
  }
  if kv_plan.generation_value() != model_generation_value {
    raise rejected(ModelGeneration, -1)
  }
  let claims : Array[RankBootstrapClaim] = []
  for rank in 0..<world_size {
    claims.push(
      rank_claim(
        model_plan,
        model_generation_value,
        group_generation_value,
        topology,
        rank,
        startups[rank],
        startups[0],
        inspections[rank],
        device_plans[rank],
        execution_plans[rank],
        kv_plan,
      ),
    )
  }
  admit_claims(
    identity,
    model_generation_value,
    group_generation_value,
    digest_topology(topology),
    collective_runtime_version_code,
    rendezvous_id,
    world_size,
    claims,
    template_digest_sha256~,
  )
}

///|
/// Preserved v3 admission surface; production v4 uses `admit_with_template`.
pub fn admit(
  model_plan : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  group_generation : @rank_group_protocol.RankGroupGeneration,
  topology : @device_topology.AdmittedLocalTopology,
  source : @worker_wire.EncodedBootstrapSource,
  startups : ArrayView[@worker_wire.WorkerStartupContract],
  inspections : ArrayView[@device_materialize.DeviceShardedWeightFileInspection],
  device_plans : ArrayView[
    @tensor_parallel_device_plan.TensorParallelDevicePlan,
  ],
  execution_plans : ArrayView[
    @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  ],
  kv_plan : @tensor_parallel_kv_plan.TensorParallelKvPlan,
  rendezvous_id : @device.TensorCollectiveGroupId,
  collective_runtime_version_code~ : Int,
) -> TensorParallelGroupBootstrap raise TensorParallelWorkerBootstrapError {
  admit_with_template(
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    model_plan,
    model_generation,
    group_generation,
    topology,
    source,
    startups,
    inspections,
    device_plans,
    execution_plans,
    kv_plan,
    rendezvous_id,
    collective_runtime_version_code~,
  )
}
