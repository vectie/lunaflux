///|
fn[T] rejected(
  rule : TensorParallelGroupTransportRule,
  rank? : Int = -1,
) -> T raise TensorParallelGroupTransportError {
  raise Rejected(rule~, rank~)
}

///|
fn exact_topology_inputs(
  declaration : @device_topology.LocalTopologyDeclaration,
  topology : @device_topology.AdmittedLocalTopology,
) -> Bool {
  if declaration.rank_count() != topology.world_size() ||
    declaration.link_count() != topology.link_count() {
    return false
  }
  for rank in 0..<topology.world_size() {
    guard declaration.rank(rank) is Some(expected) &&
      topology.rank(rank) is Some(actual) else {
      return false
    }
    if expected != actual {
      return false
    }
  }
  true
}

///|
fn exact_source_target(
  source : @worker_wire.EncodedBootstrapSource,
  topology : @device_topology.AdmittedLocalTopology,
) -> Bool {
  let execution = source.execution()
  let target = @catalog.DeviceTarget::new(
    compute_major=execution.compute_major(),
    compute_minor=execution.compute_minor(),
    supports_bf16=execution.supports_bf16(),
    supports_cublas_lt=execution.supports_cublas_lt(),
  ) catch {
    _ => return false
  }
  target == topology.target()
}

///|
fn validate_limit_bindings(
  inputs : TensorParallelGroupPlanningInputs,
) -> Unit raise TensorParallelGroupTransportError {
  let shapes = inputs.model_plan.shape_constraints()
  let step = inputs.device_step_limits
  let worker = inputs.worker_limits
  if step.max_rows() != worker.max_plan_rows() ||
    step.max_tokens() != worker.max_plan_tokens() ||
    step.max_pages() != worker.max_plan_pages() ||
    step.max_sequence_tokens() != worker.max_sequence_tokens() ||
    step.max_token_id() != worker.max_token_id() ||
    worker.max_token_id() != shapes.vocabulary_size() - 1 ||
    inputs.inference_limits.max_token_id() != worker.max_token_id() ||
    inputs.inference_limits.max_context_tokens() > worker.max_sequence_tokens() {
    rejected(Limit)
  }
  let ceilings = inputs.source.ceilings()
  guard inputs.source.tensor_parallel_policy() is Some(policy) else {
    rejected(Source)
  }
  let artifacts = inputs.manifest_limits.artifact_file().artifact()
  if inputs.manifest_limits.max_manifest_bytes() !=
    ceilings.max_execution_manifest_bytes() ||
    artifacts.max_module_bytes() != ceilings.max_module_bytes() ||
    artifacts.max_total_module_bytes() != ceilings.max_total_module_bytes() ||
    inputs.process_limits.collective_exchange_timeout_millis() !=
    policy.collective_exchange_timeout_millis() ||
    inputs.execution_limits.max_activation_workspace_bytes() >
    ceilings.max_activation_arena_bytes() {
    rejected(Limit)
  }
}

///|
fn validate_inspection_blueprints(
  inputs : TensorParallelGroupPlanningInputs,
) -> Unit raise TensorParallelGroupTransportError {
  let world_size = inputs.topology.world_size()
  if inputs.inspection_blueprints.length() != world_size {
    rejected(RankInspection)
  }
  for rank in 0..<world_size {
    let rank_plan = inputs.inspection_blueprints[rank].rank_plan()
    let rank_topology = rank_plan.topology()
    if rank_plan.identity() != inputs.model_plan.identity() ||
      rank_topology.rank() != rank ||
      rank_topology.world_size() != world_size {
      rejected(RankInspection, rank~)
    }
  }
}

///|
fn validate_planning_inputs(
  inputs : TensorParallelGroupPlanningInputs,
) -> Unit raise TensorParallelGroupTransportError {
  if inputs.source.tensor_parallel_policy() is None {
    rejected(Source)
  }
  if !inputs.source.has_valid_tensor_parallel_recipe_binding() {
    rejected(Source)
  }
  if inputs.model_generation.value() == 0UL {
    rejected(ModelGeneration)
  }
  if !exact_topology_inputs(inputs.topology_declaration, inputs.topology) ||
    !exact_source_target(inputs.source, inputs.topology) {
    rejected(Topology)
  }
  ignore(
    // This helper is the canonical v4 tensor-parallel recipe discriminator;
    // ordinary and earlier source versions are rejected before any root use.
    @tensor_parallel_worker_bootstrap.admit_runtime_policy(
      inputs.source,
      inputs.topology,
    ) catch {
      _ => rejected(Topology)
    },
  )
  if inputs.admitted_rank_child_activation_path.length() == 0 {
    rejected(SchedulerPredecessor)
  }
  validate_limit_bindings(inputs)
  validate_inspection_blueprints(inputs)
}
