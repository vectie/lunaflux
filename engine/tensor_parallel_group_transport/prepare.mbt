///|
pub fn TensorParallelGroupPlanningInputs::new(
  source~ : @worker_wire.EncodedBootstrapSource,
  model_plan~ : @plan.ModelPlan,
  model_generation~ : @worker_protocol.ModelPlanGeneration,
  topology_declaration~ : @device_topology.LocalTopologyDeclaration,
  topology~ : @device_topology.AdmittedLocalTopology,
  inspection_blueprints~ : ArrayView[
    @device_materialize.DeviceShardedWeightFileInspection,
  ],
  device_step_limits~ : @device_step.DeviceStepLimits,
  manifest_limits~ : @execution_manifest_file.ExecutionManifestFileLimits,
  launch_limits~ : @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits,
  execution_limits~ : @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits,
  worker_limits~ : @worker_protocol.WorkerProtocolLimits,
  inference_limits~ : @inference.InferenceLimits,
  process_limits~ : @rank_group_process.RankGroupProcessLimits,
  admitted_rank_child_activation_path~ : Bytes,
  rank_child_executable_digest_sha256~ : StringView,
) -> TensorParallelGroupPlanningInputs raise TensorParallelGroupTransportError {
  let inspections : Array[@device_materialize.DeviceShardedWeightFileInspection] = []
  for inspection in inspection_blueprints {
    inspections.push(inspection)
  }
  let template_digest = derive_group_template_digest(
    source, model_plan, model_generation, topology_declaration, inspection_blueprints,
    device_step_limits, manifest_limits, launch_limits, execution_limits, worker_limits,
    inference_limits, process_limits, admitted_rank_child_activation_path, rank_child_executable_digest_sha256,
  )
  let inputs : TensorParallelGroupPlanningInputs = {
    source,
    model_plan,
    model_generation,
    topology_declaration,
    topology,
    inspection_blueprints: ReadOnlyArray::from_array(inspections),
    device_step_limits,
    manifest_limits,
    launch_limits,
    execution_limits,
    worker_limits,
    inference_limits,
    process_limits,
    admitted_rank_child_activation_path,
    rank_child_executable_digest_sha256: rank_child_executable_digest_sha256.to_owned(),
    template_digest,
  }
  validate_planning_inputs(inputs)
  inputs
}

///|
/// Pins exactly one independently duplicated model/kernel root pair after all
/// authority-free planning evidence has been validated. Caller roots remain
/// independently closeable and are never retained by this owner.
fn prepare_tensor_parallel_group_transport_internal(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  inputs : TensorParallelGroupPlanningInputs,
  approved_executable : @worker_executable_file.WorkerExecutableAdmission,
  plan_buffer_identities : ArrayView[@worker_protocol.PlanBufferIdentity],
  previous_plan_sequence_value~ : UInt64,
) -> TensorParallelGroupTransport raise TensorParallelGroupTransportError {
  validate_planning_inputs(inputs)
  if plan_buffer_identities.length() != 2 ||
    plan_buffer_identities[0].same_owner(plan_buffer_identities[1]) ||
    previous_plan_sequence_value == 0xffffffffffffffffUL {
    rejected(SchedulerPredecessor)
  }
  let identities : Array[@worker_protocol.PlanBufferIdentity] = []
  for identity in plan_buffer_identities {
    identities.push(identity)
  }
  let source_roots = inputs.source.roots()
  model_root.require_absolute_identity(source_roots.model_root()) catch {
    _ => rejected(Root)
  }
  kernel_root.require_absolute_identity(source_roots.kernel_root()) catch {
    _ => rejected(Root)
  }
  let roots = @approved_fs.acquire_worker_approved_roots(
    model_root~,
    kernel_root~,
  ) catch {
    _ => rejected(Root)
  }
  {
    inputs,
    approved_executable,
    roots,
    plan_buffer_identities: ReadOnlyArray::from_array(identities),
    next_group_generation_value: 1UL,
    current_previous_plan_sequence_value: previous_plan_sequence_value,
    active_sequence_value: 0UL,
    completion_accepted: false,
    replacement_required: false,
    retained_failure_report: None,
    failure_report_taken: false,
    invalid_failure_report: false,
    state: NoGroup,
  }
}

///|
/// Retains one exact executable capability for all current and replacement
/// ranks. The activation label in planning inputs remains digest evidence only.
pub fn prepare_tensor_parallel_group_transport_approved(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  inputs : TensorParallelGroupPlanningInputs,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  plan_buffer_identities : ArrayView[@worker_protocol.PlanBufferIdentity],
  previous_plan_sequence_value~ : UInt64,
) -> TensorParallelGroupTransport raise TensorParallelGroupTransportError {
  if inputs.admitted_rank_child_activation_path != executable.activation_path() ||
    inputs.rank_child_executable_digest_sha256 != executable.digest().as_hex() {
    rejected(Process)
  }
  prepare_tensor_parallel_group_transport_internal(
    model_root,
    kernel_root,
    inputs,
    executable,
    plan_buffer_identities,
    previous_plan_sequence_value~,
  )
}

///|
pub fn TensorParallelGroupTransport::lifecycle(
  self : TensorParallelGroupTransport,
) -> TensorParallelGroupTransportLifecycle {
  match self.state {
    NoGroup => Prepared
    Published(supervisor) =>
      if supervisor.lifecycle() == Faulted {
        Faulted
      } else {
        Ready
      }
    Failed(_) => Faulted
    Cleaning(_, replace_after_cleanup~) =>
      if replace_after_cleanup {
        Recovering
      } else {
        Closing
      }
    ClosingRoots => Closing
    OwnerClosed => Closed
  }
}

///|
pub fn TensorParallelGroupTransport::world_size(
  self : TensorParallelGroupTransport,
) -> Int {
  self.inputs.topology.world_size()
}

///|
pub fn TensorParallelGroupTransport::model_generation_value(
  self : TensorParallelGroupTransport,
) -> UInt64 {
  self.inputs.model_generation.value()
}

///|
pub fn TensorParallelGroupTransport::previous_plan_sequence_value(
  self : TensorParallelGroupTransport,
) -> UInt64 {
  self.current_previous_plan_sequence_value
}

///|
pub fn TensorParallelGroupTransport::group_generation_value(
  self : TensorParallelGroupTransport,
) -> UInt64 {
  match self.state {
    Published(supervisor) | Failed(supervisor) | Cleaning(supervisor, ..) =>
      supervisor.group_generation_value()
    _ => 0UL
  }
}

///|
/// Reauthenticates the backend-neutral service identity and exact scheduler
/// predecessor without exposing rank-local bootstrap or topology details.
pub fn TensorParallelGroupTransport::validate_service_identity(
  self : TensorParallelGroupTransport,
  model_identity~ : @spec.ModelIdentity,
  model_generation_value~ : UInt64,
  bootstrap_source_digest~ : @worker_wire.WorkerBootstrapSourceDigest,
  worker_limits~ : @worker_protocol.WorkerProtocolLimits,
  inference_limits~ : @inference.InferenceLimits,
  previous_plan_sequence_value~ : UInt64,
) -> Unit raise TensorParallelGroupTransportError {
  if model_identity != self.inputs.model_plan.identity() {
    rejected(ModelIdentity)
  }
  if model_generation_value != self.inputs.model_generation.value() {
    rejected(ModelGeneration)
  }
  if bootstrap_source_digest != self.inputs.source.digest() {
    rejected(Source)
  }
  if worker_limits != self.inputs.worker_limits ||
    inference_limits != self.inputs.inference_limits {
    rejected(WorkerContract)
  }
  if previous_plan_sequence_value != self.current_previous_plan_sequence_value {
    rejected(SchedulerPredecessor)
  }
}

///|
pub fn TensorParallelGroupTransport::has_recovery_obligation(
  self : TensorParallelGroupTransport,
) -> Bool {
  self.replacement_required &&
  (
    self.lifecycle() == Faulted ||
    self.lifecycle() == Recovering ||
    self.lifecycle() == Prepared
  )
}

///|
pub fn TensorParallelGroupPlanningInputs::model_identity(
  self : TensorParallelGroupPlanningInputs,
) -> @spec.ModelIdentity {
  self.model_plan.identity()
}

///|
pub fn TensorParallelGroupPlanningInputs::model_generation_value(
  self : TensorParallelGroupPlanningInputs,
) -> UInt64 {
  self.model_generation.value()
}

///|
pub fn TensorParallelGroupPlanningInputs::worker_limits(
  self : TensorParallelGroupPlanningInputs,
) -> @worker_protocol.WorkerProtocolLimits {
  self.worker_limits
}

///|
pub fn TensorParallelGroupPlanningInputs::inference_limits(
  self : TensorParallelGroupPlanningInputs,
) -> @inference.InferenceLimits {
  self.inference_limits
}

///|
pub fn TensorParallelGroupPlanningInputs::bootstrap_source_digest(
  self : TensorParallelGroupPlanningInputs,
) -> @worker_wire.WorkerBootstrapSourceDigest {
  self.source.digest()
}

///|
pub fn TensorParallelGroupPlanningInputs::template_digest(
  self : Self,
) -> TensorParallelGroupTemplateDigest {
  self.template_digest
}
