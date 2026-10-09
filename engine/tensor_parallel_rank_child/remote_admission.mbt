///|
/// Independently loads one rank from this node's approved roots. Root handles
/// never cross the network. Common model/manifest admission is shared with the
/// existing local rank child; no fictitious local topology is constructed.
pub fn load_remote_rank_plan(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  source : @worker_wire.EncodedBootstrapSource,
  compiled : @two_node_plan.TwoNodePlan,
  remote : @remote_startup.RemoteRankStartup,
  envelope : @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope,
  expected_group_digest : @tensor_parallel_worker_bootstrap.TensorParallelGroupBootstrapDigest,
) -> @tensor_parallel_device_worker.TensorParallelDeviceWorkerPlan raise {
  let startup = remote.worker()
  if source.digest() != startup.bootstrap_source_digest() ||
    remote.plan_digest() != compiled.digest() ||
    source.execution().tokens_per_page() != compiled.tokens_per_page() ||
    source.execution().total_page_count() != compiled.page_count() {
    raise Invalid(SourcePolicy)
  }
  let (metadata, model) = load_rank_child_model(model_root, startup, source)
  let rank = remote.rank()
  guard compiled.rank_capacity(rank) is Some(capacity) else {
    raise Invalid(SourcePolicy)
  }
  guard compiled.rank_budget(rank) is Some(budget) else {
    raise Invalid(SourcePolicy)
  }
  let weight_ceiling = capacity
    .weight_bytes()
    .min(source.ceilings().max_weight_arena_bytes())
  let inspection = @device_materialize.inspect_sharded_file(
    model_root,
    metadata,
    model,
    @approved_fs.ApprovedRelativeLocator::new(source.model().weights_locator()),
    rank~,
    world_size=2,
    alignment_bytes=RANK_CHILD_ALIGNMENT_BYTES,
    max_arena_bytes=weight_ceiling,
    limits=rank_child_weight_limits(
      source,
      model.operations().length(),
      weight_ceiling,
    ),
  )
  let device = @tensor_parallel_device_plan.build(
    model,
    inspection.rank_plan(),
    inspection,
    @tensor_parallel_device_plan.TensorParallelRankDevice::from_network_topology(
      compiled.topology(),
      rank~,
    ),
    compiled.topology().target(),
  )
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    startup.model_generation_value(),
  )
  let kv = @tensor_parallel_kv_plan.admit_from_two_node_plan(
    model, generation, compiled,
  )
  let limits = rank_child_execution_limits(
    startup,
    source,
    model.operations().length(),
    budget
    .activation_workspace_bytes()
    .min(source.ceilings().max_activation_arena_bytes()),
  )
  let execution = @execution_manifest_file.load_tensor_parallel(
    kernel_root,
    @approved_fs.ApprovedRelativeLocator::new(
      source.kernel().manifest_locator(),
    ),
    @execution_manifest_file.ExecutionManifestDigest::from_sha256(
      source.kernel().manifest_digest(),
    ),
    model,
    generation,
    inspection.rank_plan(),
    inspection,
    device,
    kv.rank_layout(rank),
    kv.rank_binding(rank),
    compiled.topology().target(),
    limits.0,
    limits.1,
    limits.2,
    limits.3,
    external_approval=source.kernel().luna_approval(),
  )
  @tensor_parallel_device_worker.admit_network_plan(
    model,
    compiled,
    remote,
    envelope,
    inspection,
    device,
    execution.contracts(),
    execution.artifacts(),
    execution.profile(),
    limits.3,
    kv,
    expected_group_digest,
    collective_runtime_version_code=envelope.collective_runtime_version_code(),
  )
}
