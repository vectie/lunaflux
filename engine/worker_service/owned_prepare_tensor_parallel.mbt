///|
fn validate_tensor_parallel_preflight(
  blueprint : @core.SchedulerBlueprint,
  inputs : @tensor_parallel_group_transport.TensorParallelGroupPlanningInputs,
) -> WorkerServiceIdentity raise OwnedWorkerServicePreparationError {
  if inputs.model_identity() != blueprint.model_identity() {
    raise owned_failure(OwnedServiceFailure(Invalid(Identity)))
  }
  if inputs.model_generation_value() !=
    blueprint.model_plan_generation().value() {
    raise owned_failure(OwnedServiceFailure(Invalid(Generation)))
  }
  if inputs.worker_limits() != blueprint.worker_protocol_limits() {
    raise owned_failure(OwnedServiceFailure(Invalid(WorkerEnvelope)))
  }
  worker_service_identity(
    inputs.model_identity(),
    inputs.model_generation_value(),
    inputs.bootstrap_source_digest(),
    inputs.worker_limits(),
    inputs.inference_limits(),
  ) catch {
    error => raise owned_failure(OwnedServiceFailure(error))
  }
}

///|
fn prepare_owned_tensor_parallel_internal(
  blueprint : @core.SchedulerBlueprint,
  inputs : @tensor_parallel_group_transport.TensorParallelGroupPlanningInputs,
  approved_executable : @worker_executable_file.WorkerExecutableAdmission,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  online_request : @core.TokenizedRequest?,
  online_clock : @monotonic_clock.MonotonicClock,
  restart_policy : WorkerRestartBackoffPolicy,
) -> OwnedWorkerServicePreparation raise OwnedWorkerServicePreparationError {
  let identity = validate_tensor_parallel_preflight(blueprint, inputs)
  let scheduler = blueprint.instantiate() catch {
    error => raise owned_failure(OwnedSchedulerFailure(error))
  }
  let (outcome, service, failed_owner) = prepare_owned_shell(
    blueprint,
    scheduler,
    identity,
    TensorParallelBinding,
    online_request,
    online_clock,
    restart_policy,
  )
  let plan_binding = scheduler.plan_buffer_binding()
  let group = @tensor_parallel_group_transport.prepare_tensor_parallel_group_transport_approved(
    model_root,
    kernel_root,
    inputs,
    approved_executable,
    [plan_binding.plan_a_identity(), plan_binding.plan_b_identity()],
    previous_plan_sequence_value=scheduler.last_retired_plan_sequence(),
  ) catch {
    error =>
      raise owned_failure(
        OwnedPhysicalTransportFailure(tensor_parallel_transport_failure(error)),
      )
  }
  group.validate_service_identity(
    model_identity=identity.model_identity,
    model_generation_value=identity.model_generation_value,
    bootstrap_source_digest=identity.bootstrap_source_digest,
    worker_limits=identity.worker_limits,
    inference_limits=identity.inference_limits,
    previous_plan_sequence_value=scheduler.last_retired_plan_sequence(),
  ) catch {
    error => {
      failed_owner.group = Some(group)
      failed_owner.cause_kind = 2
      failed_owner.service_cause = Some(tensor_parallel_transport_error(error))
      return publish_cleanup(outcome)
    }
  }
  let started = try group.start() catch {
    error => {
      failed_owner.group = Some(group)
      failed_owner.cause_kind = 2
      failed_owner.service_cause = Some(tensor_parallel_transport_error(error))
      return publish_cleanup(outcome)
    }
  } noraise {
    value => value
  }
  match started {
    GroupReady => {
      service.transport = tensor_parallel_group_transport(group)
      outcome.state = OwnedServiceReady
      outcome
    }
    GroupCleanupRequired => {
      failed_owner.group = Some(group)
      failed_owner.cause_kind = 2
      failed_owner.service_cause = Some(PhysicalTransport(PhysicalUnavailable))
      publish_cleanup(outcome)
    }
  }
}

///|
/// Prepares the same production-owned scheduler and online service around one
/// generation-scoped local tensor-parallel group. Rank and device details do
/// not escape through the returned service owner.
pub fn prepare_owned_tensor_parallel_approved(
  blueprint : @core.SchedulerBlueprint,
  inputs : @tensor_parallel_group_transport.TensorParallelGroupPlanningInputs,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  restart_policy : WorkerRestartBackoffPolicy,
) -> OwnedWorkerServicePreparation raise OwnedWorkerServicePreparationError {
  prepare_owned_tensor_parallel_internal(
    blueprint,
    inputs,
    executable,
    model_root,
    kernel_root,
    None,
    @monotonic_clock.MonotonicClock::system(),
    restart_policy,
  )
}

///|
/// Preallocates one exact prepared request before rooted group activation.
pub fn prepare_owned_tensor_parallel_online_approved(
  blueprint : @core.SchedulerBlueprint,
  inputs : @tensor_parallel_group_transport.TensorParallelGroupPlanningInputs,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  request : @core.TokenizedRequest,
  clock : @monotonic_clock.MonotonicClock,
  restart_policy : WorkerRestartBackoffPolicy,
) -> OwnedWorkerServicePreparation raise OwnedWorkerServicePreparationError {
  prepare_owned_tensor_parallel_internal(
    blueprint,
    inputs,
    executable,
    model_root,
    kernel_root,
    Some(request),
    clock,
    restart_policy,
  )
}
