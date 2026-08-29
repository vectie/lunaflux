///|
fn checked_product(
  left : Int64,
  right : Int64,
) -> Int64 raise TensorParallelDeviceWorkerError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise Invalid(ActivationWorkspace)
  }
  left * right
}

///|
fn derive_descriptor_limits(
  startup : @worker_wire.WorkerStartupContract,
) -> @device_step.DeviceStepLimits raise TensorParallelDeviceWorkerError {
  let limits = startup.worker_limits()
  @device_step.DeviceStepLimits::new(
    max_rows=limits.max_plan_rows(),
    max_tokens=limits.max_plan_tokens(),
    max_pages=limits.max_plan_pages(),
    max_sequence_tokens=limits.max_sequence_tokens(),
    max_token_id=limits.max_token_id(),
  ) catch {
    error => raise Descriptor(error)
  }
}

///|
fn validate_profile_envelope(
  model : @plan.ModelPlan,
  startup : @worker_wire.WorkerStartupContract,
  profile : @launch_contract.PagedKernelProfile,
) -> Unit raise TensorParallelDeviceWorkerError {
  let worker = startup.worker_limits()
  let inference = startup.inference_limits()
  let shapes = model.shape_constraints()
  if profile.max_query_rows() != worker.max_plan_rows() ||
    profile.max_query_tokens() != worker.max_plan_tokens() ||
    profile.max_page_table_entries() != worker.max_plan_pages() ||
    worker.max_plan_rows() > shapes.max_batch_rows() ||
    worker.max_sequence_tokens() > shapes.max_sequence_tokens() ||
    worker.max_token_id() != shapes.vocabulary_size() - 1 ||
    inference.max_token_id() != worker.max_token_id() ||
    inference.max_context_tokens() > worker.max_sequence_tokens() {
    raise Invalid(ProfileEnvelope)
  }
}

///|
fn validate_rank_constants(
  execution : @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
) -> Unit raise TensorParallelDeviceWorkerError {
  let constants = launch_set.rank_constants()
  let region = execution.rank_constants()
  if constants.rank() != execution.rank() ||
    constants.world_size() != execution.world_size() ||
    constants.byte_count() != region.byte_count() ||
    constants.alignment() != region.alignment() ||
    region.offset_bytes() != 0L ||
    constants.compute_major() != execution.target().compute_major() ||
    constants.compute_minor() != execution.target().compute_minor() ||
    constants.kv_layout_version() !=
    launch_set.device_kv_layout().stable_layout_version() ||
    constants.tokens_per_page() !=
    launch_set.device_kv_layout().tokens_per_page() {
    raise Invalid(RankConstants)
  }
}

///|
fn validate_final_output(
  model : @plan.ModelPlan,
  execution : @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
) -> Unit raise TensorParallelDeviceWorkerError {
  guard execution.activation(model.final_output()) is Some(output) else {
    raise Invalid(FinalOutput)
  }
  let row_bytes = checked_product(
    model.shape_constraints().vocabulary_size().to_int64(),
    2L,
  )
  let expected_bytes = checked_product(
    execution.profile().max_query_tokens().to_int64(),
    row_bytes,
  )
  if output.region().byte_count() != expected_bytes ||
    output.region().end_offset_bytes() > execution.activation_workspace_bytes() {
    raise Invalid(FinalOutput)
  }
  let mut found = false
  for index in 0..<execution.collective_count() {
    let site = execution.collective_exact(index) catch {
      _ => raise Invalid(CollectiveOrder)
    }
    if site.operation_id() == output.produced_by() {
      if found || site.kind() != InPlaceLastAxisAllGather {
        raise Invalid(CollectiveOrder)
      }
      found = true
    }
  }
  if !found {
    raise Invalid(CollectiveOrder)
  }
}

///|
/// Rebuilds the rank-local physical execution plan from authenticated semantic,
/// launch, device, artifact, and root-derived sharded-file evidence. The
/// decoded bootstrap envelope is accepted only after authenticating that exact
/// rebuilt plan and every local startup binding.
pub fn admit_plan(
  model : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  group_generation : @rank_group_protocol.RankGroupGeneration,
  topology : @device_topology.AdmittedLocalTopology,
  startup : @worker_wire.WorkerStartupContract,
  envelope : @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope,
  inspection : @device_materialize.DeviceShardedWeightFileInspection,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  selected_profile : @launch_contract.PagedKernelProfile,
  execution_limits : @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits,
  kv_plan : @tensor_parallel_kv_plan.TensorParallelKvPlan,
  expected_group_digest : @tensor_parallel_worker_bootstrap.TensorParallelGroupBootstrapDigest,
  expected_rendezvous_id : @device.TensorCollectiveGroupId,
  collective_runtime_version_code~ : Int,
) -> TensorParallelDeviceWorkerPlan raise TensorParallelDeviceWorkerError {
  if model_generation.value() == 0UL ||
    startup.model_generation_value() != model_generation.value() {
    raise Invalid(ModelGeneration)
  }
  if group_generation.value() == 0UL {
    raise Invalid(GroupGeneration)
  }
  if startup.identity() != model.identity() {
    raise Invalid(StartupContract)
  }
  validate_profile_envelope(model, startup, selected_profile)
  let execution = @tensor_parallel_execution_plan.admit(
    model, launch_set, device_plan, artifacts, selected_profile, execution_limits,
  ) catch {
    error => raise ExecutionPlan(error)
  }
  let rank = execution.rank()
  if rank != envelope.rank() ||
    execution.world_size() != topology.world_size() ||
    execution.device_ordinal() != startup.device_ordinal() {
    raise Invalid(Rank)
  }
  let kv_binding = kv_plan.rank_binding(rank) catch {
    _ => raise Invalid(KvBinding)
  }
  let local_layout = kv_plan.rank_layout(rank) catch {
    _ => raise Invalid(KvBinding)
  }
  if kv_binding.arena_bytes() != execution.kv_arena_bytes() ||
    kv_binding.process_ordinal() != execution.device_ordinal() ||
    kv_binding.target() != execution.target() ||
    kv_binding.generation_value() != model_generation.value() ||
    launch_set.device_kv_layout() != local_layout {
    raise Invalid(KvBinding)
  }
  let collective_contract = envelope.authenticate_local(
    model,
    model_generation,
    group_generation,
    topology,
    startup,
    inspection,
    device_plan,
    execution,
    kv_binding,
    expected_group_digest~,
    expected_rendezvous_id~,
    expected_collective_runtime_version_code=collective_runtime_version_code,
  ) catch {
    error => raise Bootstrap(error)
  }
  if collective_contract.site_count() != execution.collective_count() ||
    collective_contract.rank() != rank ||
    collective_contract.world_size() != execution.world_size() {
    raise Invalid(CollectiveOrder)
  }
  validate_rank_constants(execution, launch_set)
  validate_final_output(model, execution)
  {
    startup,
    model,
    model_generation,
    group_generation,
    inspection,
    execution,
    artifacts,
    rank_constants: launch_set.rank_constants(),
    kv_plan,
    collective_contract,
    rendezvous_id: expected_rendezvous_id,
    collective_runtime_version_code,
    descriptor_limits: derive_descriptor_limits(startup),
  }
}
