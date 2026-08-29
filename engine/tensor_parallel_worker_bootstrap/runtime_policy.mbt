///|
/// Authenticates one canonical legacy-v5 or approved-v8 source policy against
/// independently admitted local topology and returns the sole capacity proof
/// used by parent and child.
pub fn admit_runtime_policy(
  source : @worker_wire.EncodedBootstrapSource,
  topology : @device_topology.AdmittedLocalTopology,
) -> @tensor_parallel_admission.TensorParallelRuntimeAdmission raise TensorParallelWorkerBootstrapError {
  if !source.has_valid_tensor_parallel_recipe_binding() {
    raise rejected(WorkerContract, -1)
  }
  guard source.tensor_parallel_policy() is Some(policy) else {
    raise rejected(WorkerContract, -1)
  }
  let world_size = topology.world_size()
  if policy.rank_count() != world_size {
    raise rejected(WorldSize, -1)
  }
  let declarations : Array[
    @tensor_parallel_admission.TensorParallelRankDeclaration,
  ] = []
  for rank in 0..<world_size {
    guard policy.rank(rank) is Some(claim) &&
      topology.rank(rank) is Some(topology_rank) else {
      raise rejected(Topology, rank)
    }
    if claim.rank() != rank || claim.device_ordinal() != topology_rank.ordinal() {
      raise rejected(Topology, rank)
    }
    declarations.push(
      @tensor_parallel_admission.TensorParallelRankDeclaration::new(
        rank~,
        world_size~,
        process_ordinal=claim.device_ordinal(),
        device_name=topology_rank.name(),
        target=topology_rank.target(),
        peer_rank_count=world_size - 1,
        memory_ceiling_bytes=claim.memory_ceiling_bytes(),
        weight_reserved_bytes=claim.weight_reserved_bytes(),
        activation_workspace_reserved_bytes=claim.activation_workspace_reserved_bytes(),
        kv_reserved_bytes=claim.kv_reserved_bytes(),
        collective_reserved_bytes=claim.collective_reserved_bytes(),
      ) catch {
        _ => raise rejected(Topology, rank)
      },
    )
  }
  let declaration = @tensor_parallel_admission.TensorParallelRuntimeDeclaration::new(
    host_count=1,
    pipeline_stage_count=1,
    declarations,
  ) catch {
    _ => raise rejected(Topology, -1)
  }
  @tensor_parallel_admission.admit(topology, declaration) catch {
    _ => raise rejected(Topology, -1)
  }
}
