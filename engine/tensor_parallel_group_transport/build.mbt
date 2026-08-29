///|
fn TensorParallelGroupTransport::mint_group_generation(
  self : TensorParallelGroupTransport,
) -> @rank_group_protocol.RankGroupGeneration raise TensorParallelGroupTransportError {
  let value = self.next_group_generation_value
  if value == 0UL {
    rejected(GroupGeneration)
  }
  let generation = @rank_group_protocol.RankGroupGeneration::from_wire(value) catch {
    _ => rejected(GroupGeneration)
  }
  self.next_group_generation_value = if value == 0xffffffffffffffffUL {
    0UL
  } else {
    value + 1UL
  }
  generation
}

///|
fn admit_collective_attempt(
  inputs : TensorParallelGroupPlanningInputs,
) -> (
  @tensor_parallel_admission.TensorParallelRuntimeAdmission,
  Int,
  @device.TensorCollectiveGroupId,
) raise TensorParallelGroupTransportError {
  let runtime_policy = @tensor_parallel_worker_bootstrap.admit_runtime_policy(
    inputs.source,
    inputs.topology,
  ) catch {
    _ => rejected(Topology)
  }
  guard inputs.source.tensor_parallel_policy() is Some(policy) else {
    rejected(Source)
  }
  let collective_runtime = @device.admit_tensor_collective_runtime(
    minimum_version_code=policy.minimum_collective_runtime_version_code(),
    maximum_version_code=policy.maximum_collective_runtime_version_code(),
  ) catch {
    _ => rejected(CollectiveRuntime)
  }
  let version = collective_runtime.collective_runtime_version_code()
  let rendezvous = collective_runtime.create_group_id() catch {
    _ => rejected(Rendezvous)
  }
  (runtime_policy, version, rendezvous)
}

///|
fn validate_rank_capacity(
  inputs : TensorParallelGroupPlanningInputs,
  runtime : @tensor_parallel_admission.TensorParallelRuntimeAdmission,
  rank : Int,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  execution_plan? : @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  kv_binding? : @tensor_parallel_kv_plan.TensorParallelKvRankBinding,
) -> Unit raise TensorParallelGroupTransportError {
  guard runtime.rank_capacity(rank) is Some(capacity) else {
    rejected(RankCapacity, rank~)
  }
  let ceilings = inputs.source.ceilings()
  if device_plan.arena_bytes() > capacity.weight_reserved_bytes() ||
    device_plan.arena_bytes() > ceilings.max_weight_arena_bytes() {
    rejected(RankCapacity, rank~)
  }
  match kv_binding {
    Some(binding) =>
      if binding.arena_bytes() > capacity.kv_reserved_bytes() ||
        binding.arena_bytes() > ceilings.max_kv_arena_bytes() {
        rejected(RankCapacity, rank~)
      }
    None => ()
  }
  match execution_plan {
    Some(plan) =>
      if plan.weight_arena_bytes() != device_plan.arena_bytes() ||
        plan.activation_workspace_bytes() >
        capacity.activation_workspace_reserved_bytes() ||
        plan.activation_workspace_bytes() >
        ceilings.max_activation_arena_bytes() {
        rejected(RankCapacity, rank~)
      }
    None => ()
  }
}

///|
fn fresh_rank_foundations(
  inputs : TensorParallelGroupPlanningInputs,
  model_root : @approved_fs.ApprovedRoot,
  runtime : @tensor_parallel_admission.TensorParallelRuntimeAdmission,
) -> (
  Array[@device_materialize.DeviceShardedWeightFileInspection],
  Array[@tensor_parallel_device_plan.TensorParallelDevicePlan],
) raise TensorParallelGroupTransportError {
  let inspections : Array[@device_materialize.DeviceShardedWeightFileInspection] = []
  let device_plans : Array[
    @tensor_parallel_device_plan.TensorParallelDevicePlan,
  ] = []
  for rank in 0..<inputs.topology.world_size() {
    let blueprint = inputs.inspection_blueprints[rank]
    let inspection = blueprint.reauthenticate(model_root) catch {
      _ => rejected(RankInspection, rank~)
    }
    if inspection.rank_plan() != blueprint.rank_plan() {
      rejected(RankInspection, rank~)
    }
    let rank_device = @tensor_parallel_device_plan.TensorParallelRankDevice::from_local_topology(
      inputs.topology,
      rank~,
    ) catch {
      _ => rejected(Topology, rank~)
    }
    let device_plan = @tensor_parallel_device_plan.build(
      inputs.model_plan,
      inspection.rank_plan(),
      inspection,
      rank_device,
      inputs.topology.target(),
    ) catch {
      _ => rejected(DevicePlan, rank~)
    }
    validate_rank_capacity(inputs, runtime, rank, device_plan)
    inspections.push(inspection)
    device_plans.push(device_plan)
  }
  (inspections, device_plans)
}

///|
fn fresh_execution_inputs(
  inputs : TensorParallelGroupPlanningInputs,
  previous_plan_sequence_value : UInt64,
  kernel_root : @approved_fs.ApprovedRoot,
  runtime : @tensor_parallel_admission.TensorParallelRuntimeAdmission,
  inspections : Array[@device_materialize.DeviceShardedWeightFileInspection],
  device_plans : Array[@tensor_parallel_device_plan.TensorParallelDevicePlan],
) -> (
  Array[@worker_wire.WorkerStartupContract],
  Array[@tensor_parallel_execution_plan.TensorParallelExecutionPlan],
  @tensor_parallel_kv_plan.TensorParallelKvPlan,
) raise TensorParallelGroupTransportError {
  let execution_source = inputs.source.execution()
  let kv_plan = @tensor_parallel_kv_plan.admit(
    inputs.model_plan,
    inputs.model_generation,
    runtime,
    device_plans,
    tokens_per_page=execution_source.tokens_per_page(),
    total_page_count=execution_source.total_page_count(),
  ) catch {
    _ => rejected(KvPlan)
  }
  let kernel_source = inputs.source.kernel()
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    kernel_source.manifest_locator(),
  ) catch {
    _ => rejected(Source)
  }
  let expected_digest = @execution_manifest_file.ExecutionManifestDigest::from_sha256(
    kernel_source.manifest_digest(),
  ) catch {
    _ => rejected(Source)
  }
  let startups : Array[@worker_wire.WorkerStartupContract] = []
  let execution_plans : Array[
    @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  ] = []
  for rank in 0..<inputs.topology.world_size() {
    let binding = kv_plan.rank_binding(rank) catch {
      _ => rejected(KvPlan, rank~)
    }
    let local_layout = kv_plan.rank_layout(rank) catch {
      _ => rejected(KvPlan, rank~)
    }
    let admission = @execution_manifest_file.load_tensor_parallel(
      kernel_root,
      locator,
      expected_digest,
      inputs.model_plan,
      inputs.model_generation,
      inspections[rank].rank_plan(),
      inspections[rank],
      device_plans[rank],
      local_layout,
      binding,
      inputs.topology.target(),
      inputs.device_step_limits,
      inputs.manifest_limits,
      inputs.launch_limits,
      inputs.execution_limits,
      external_approval=kernel_source.luna_approval(),
    ) catch {
      _ => rejected(Manifest, rank~)
    }
    let execution_plan = admission.execution_plan()
    validate_rank_capacity(
      inputs,
      runtime,
      rank,
      device_plans[rank],
      execution_plan~,
      kv_binding=binding,
    )
    let startup = @worker_wire.WorkerStartupContract::new(
      identity=inputs.model_plan.identity(),
      bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
        execution_plan.digest().as_sha256(),
      ),
      bootstrap_source_digest=inputs.source.digest(),
      device_ordinal=device_plans[rank].device_ordinal(),
      model_generation_value=inputs.model_generation.value(),
      worker_limits=inputs.worker_limits,
      inference_limits=inputs.inference_limits,
      has_predecessor=previous_plan_sequence_value != 0UL,
      predecessor_value=previous_plan_sequence_value,
    ) catch {
      _ => rejected(WorkerContract, rank~)
    }
    execution_plans.push(execution_plan)
    startups.push(startup)
    // `admission` is intentionally not retained: its artifact bundle owns
    // module bytes, while `execution_plan` is inert scalar/string evidence.
  }
  (startups, execution_plans, kv_plan)
}

///|
fn compose_rank_startups(
  inputs : TensorParallelGroupPlanningInputs,
  generation : @rank_group_protocol.RankGroupGeneration,
  collective_runtime_version_code : Int,
  rendezvous_id : @device.TensorCollectiveGroupId,
  startups : Array[@worker_wire.WorkerStartupContract],
  inspections : Array[@device_materialize.DeviceShardedWeightFileInspection],
  device_plans : Array[@tensor_parallel_device_plan.TensorParallelDevicePlan],
  execution_plans : Array[
    @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  ],
  kv_plan : @tensor_parallel_kv_plan.TensorParallelKvPlan,
) -> TensorParallelGroupAttempt raise TensorParallelGroupTransportError {
  let bootstrap = @tensor_parallel_worker_bootstrap.admit_with_template(
    inputs.template_digest.as_sha256(),
    inputs.model_plan,
    inputs.model_generation,
    generation,
    inputs.topology,
    inputs.source,
    startups,
    inspections,
    device_plans,
    execution_plans,
    kv_plan,
    rendezvous_id,
    collective_runtime_version_code~,
  ) catch {
    _ => rejected(Bootstrap)
  }
  if bootstrap.template_digest_sha256() != inputs.template_digest.as_sha256() {
    rejected(Bootstrap)
  }
  let rank_startups : Array[@rank_group_process.RankGroupProcessRankStartup] = []
  for rank in 0..<inputs.topology.world_size() {
    guard bootstrap.rank_envelope(rank) is Some(envelope) else {
      rejected(Bootstrap, rank~)
    }
    let binding = @rank_group_wire.RankGroupWireBinding::new(
      identity=inputs.model_plan.identity(),
      model_generation_value=inputs.model_generation.value(),
      group_generation_value=generation.value(),
      rank~,
      world_size=inputs.topology.world_size(),
      group_digest=bootstrap.digest().as_sha256(),
      topology_digest=bootstrap.topology_digest().as_sha256(),
      worker_digest=envelope.worker_digest().as_sha256(),
      device_plan_digest=envelope.device_plan_digest().as_sha256(),
      execution_plan_digest=envelope.execution_plan_digest_sha256(),
      collective_digest=envelope.collective_digest().as_hex(),
      kv_digest=envelope.kv_digest().as_sha256(),
    ) catch {
      _ => rejected(Bootstrap, rank~)
    }
    let configure = @tensor_parallel_rank_configure.compose_rank_configure(
      binding,
      startups[rank],
      inputs.source,
      inputs.topology_declaration,
      envelope,
    ) catch {
      _ => rejected(Configure, rank~)
    }
    rank_startups.push(
      @rank_group_process.RankGroupProcessRankStartup::new(configure),
    )
  }
  { startups: ReadOnlyArray::from_array(rank_startups) }
}

///|
fn build_attempt_from_roots(
  inputs : TensorParallelGroupPlanningInputs,
  previous_plan_sequence_value : UInt64,
  generation : @rank_group_protocol.RankGroupGeneration,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> TensorParallelGroupAttempt raise TensorParallelGroupTransportError {
  let (runtime, version, rendezvous) = admit_collective_attempt(inputs)
  let (inspections, device_plans) = fresh_rank_foundations(
    inputs, model_root, runtime,
  )
  let (startups, execution_plans, kv_plan) = fresh_execution_inputs(
    inputs, previous_plan_sequence_value, kernel_root, runtime, inspections, device_plans,
  )
  compose_rank_startups(
    inputs, generation, version, rendezvous, startups, inspections, device_plans,
    execution_plans, kv_plan,
  )
}

///|
fn TensorParallelGroupTransport::build_attempt(
  self : TensorParallelGroupTransport,
  generation : @rank_group_protocol.RankGroupGeneration,
) -> TensorParallelGroupAttempt raise TensorParallelGroupTransportError {
  let predecessor = self.current_previous_plan_sequence_value
  if predecessor == 0xffffffffffffffffUL {
    rejected(SchedulerPredecessor)
  }
  let model_root = @approved_fs.duplicate_worker_approved_root(
    self.roots,
    Model,
  ) catch {
    _ => rejected(Root)
  }
  let kernel_root = try
    @approved_fs.duplicate_worker_approved_root(self.roots, Kernel)
  catch {
    _ => {
      model_root.close() catch {
        _ => ()
      }
      rejected(Root)
    }
  } noraise {
    root => root
  }
  let outcome : Result[
    TensorParallelGroupAttempt,
    TensorParallelGroupTransportError,
  ] = Ok(
    build_attempt_from_roots(
      self.inputs,
      predecessor,
      generation,
      model_root,
      kernel_root,
    ),
  ) catch {
    error => Err(error)
  }
  let mut close_failed = false
  kernel_root.close() catch {
    _ => close_failed = true
  }
  model_root.close() catch {
    _ => close_failed = true
  }
  match outcome {
    Err(error) => raise error
    Ok(attempt) => {
      if close_failed {
        rejected(Root)
      }
      attempt
    }
  }
}
