///|
fn network_rank_claim(
  model : @plan.ModelPlan,
  compiled : @two_node_plan.TwoNodePlan,
  remote : @remote_startup.RemoteRankStartup,
  inspection : @device_materialize.DeviceShardedWeightFileInspection,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  execution : @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  kv : @tensor_parallel_kv_plan.TensorParallelKvPlan,
) -> RankBootstrapClaim raise TensorParallelWorkerBootstrapError {
  let rank = remote.rank()
  let startup = remote.worker()
  if compiled.model_identity() != model.identity() ||
    startup.identity() != model.identity() {
    raise rejected(ModelIdentity, rank)
  }
  if remote.plan_digest() != compiled.digest() {
    raise rejected(Topology, rank)
  }
  guard compiled.topology().rank(rank) is Some(node) &&
    compiled.rank_plan(rank) is Some(rank_plan) &&
    compiled.rank_budget(rank) is Some(budget) else {
    raise rejected(Rank, rank)
  }
  if node != remote.node() || startup.device_ordinal() != node.device_ordinal() {
    raise rejected(DeviceOrdinal, rank)
  }
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    startup.model_generation_value(),
  ) catch {
    _ => raise rejected(ModelGeneration, rank)
  }
  let expected_kv = @tensor_parallel_kv_plan.admit_from_two_node_plan(
    model, generation, compiled,
  ) catch {
    _ => raise rejected(KvContract, rank)
  }
  if kv != expected_kv {
    raise rejected(KvContract, rank)
  }
  let rank_device = @tensor_parallel_device_plan.TensorParallelRankDevice::from_network_topology(
    compiled.topology(),
    rank~,
  ) catch {
    _ => raise rejected(Topology, rank)
  }
  let expected_device = @tensor_parallel_device_plan.build(
    model,
    rank_plan,
    inspection,
    rank_device,
    node.target(),
  ) catch {
    _ => raise rejected(DevicePlan, rank)
  }
  if device_plan != expected_device {
    raise rejected(DevicePlan, rank)
  }
  require_execution_plan(
    execution,
    model.identity(),
    rank,
    2,
    node.device_ordinal(),
    node.target(),
  )
  // All separately allocated execution regions must fit this rank's budget.
  // The activation budget includes descriptors, constants and kernel workspace.
  let mut remaining = budget.activation_workspace_bytes()
  for
    bytes in [
      execution.activation_workspace_bytes(),
      execution.descriptor_bytes(),
      execution.rank_constants().byte_count(),
    ] {
    if bytes < 0L || bytes > remaining {
      raise rejected(ExecutionPlan, rank)
    }
    remaining -= bytes
  }
  if execution.weight_arena_bytes() != rank_plan.arena_bytes() ||
    execution.kv_arena_bytes() != compiled.kv_layout().arena_bytes() {
    raise rejected(ExecutionPlan, rank)
  }
  let collective = @tensor_parallel_collective.authenticate_rank_plan(
    rank_plan,
    expected_identity=model.identity(),
    max_query_tokens=startup.worker_limits().max_plan_tokens(),
  ) catch {
    _ => raise rejected(CollectiveContract, rank)
  }
  let binding = kv.rank_binding(rank) catch {
    _ => raise rejected(KvContract, rank)
  }
  {
    identity: model.identity(),
    model_generation_value: startup.model_generation_value(),
    group_generation_value: remote.group_generation_value(),
    rank,
    world_size: 2,
    device_ordinal: node.device_ordinal(),
    worker_digest: digest_worker_contract(startup),
    device_plan_digest: digest_device_plan(device_plan),
    execution_plan_digest_sha256: execution.digest().as_sha256(),
    collective_digest: collective.digest(),
    kv_digest: kv_binding_digest_from_hex(binding.digest().as_sha256()),
  }
}
