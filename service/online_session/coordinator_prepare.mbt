///|
/// Constructs one transport-neutral coordinator and every inner owner. The
/// same monotonic clock governs first-byte receipt, online admission, and
/// framed-output stall enforcement.
fn prepare_owned_luna_online_framed_service_approved_internal(
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  framed_limits : @framed_wire.FramedWireLimits,
  preparation_lane_count : Int,
  preparation_step_budget : @request_admission.LunaRequestPreparationStepBudget,
  preparation_work_limit : @request_admission.LunaRequestPreparationWorkLimit,
  preparation_storage_budget : @request_admission.LunaRequestPreparationStorageBudget,
  event_step_budget : @framed_wire.LunaFramedEventStepBudget,
  blueprint : @core.SchedulerBlueprint,
  binding : @worker_service.WorkerServiceBinding,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  process_limits : @worker_process.WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  restart_policy : @worker_service.WorkerRestartBackoffPolicy,
) -> LunaOnlineFramedServicePreparation raise LunaOnlineInstancePreparationError {
  if framed_limits.inference_limits() != inference {
    raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  let clock = @monotonic_clock.MonotonicClock::system()
  let pool = @request_admission.LunaRequestPreparationPool::new(
    tokenizer,
    expected_tokenizer~,
    expected_model~,
    inference~,
    framed_limits~,
    clock~,
    lane_count=preparation_lane_count,
    step_budget=preparation_step_budget,
    work_limit=preparation_work_limit,
    storage_budget=preparation_storage_budget,
  ) catch {
    _ => raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  let outbound = @framed_wire.LunaFramedEventWorkspace::new(
    framed_limits, event_step_budget,
  ) catch {
    _ => raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  let prepared = prepare_owned_luna_online_instance_with_clock(
    expected_tokenizer, expected_model, inference, blueprint, binding, executable,
    startup, source, process_limits, model_root, kernel_root, clock, restart_policy,
  )
  match prepared.state() {
    LunaOnlineInstanceReady => {
      let instance = prepared.take_ready()
      {
        service: Some(
          new_luna_online_framed_coordinator(
            pool,
            instance,
            outbound,
            event_step_budget.work_units(),
            inference,
            framed_limits,
            expected_model,
            clock,
          ),
        ),
        cleanup: None,
        state: LunaOnlineInstanceReady,
        consumed: false,
      }
    }
    LunaOnlineInstanceCleanupRequired =>
      {
        service: None,
        cleanup: Some(prepared.take_cleanup()),
        state: LunaOnlineInstanceCleanupRequired,
        consumed: false,
      }
  }
}

///|
/// Production framed coordinator using only a pinned executable capability.
pub fn prepare_owned_luna_online_framed_service_approved(
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  framed_limits : @framed_wire.FramedWireLimits,
  preparation_lane_count : Int,
  preparation_step_budget : @request_admission.LunaRequestPreparationStepBudget,
  preparation_work_limit : @request_admission.LunaRequestPreparationWorkLimit,
  preparation_storage_budget : @request_admission.LunaRequestPreparationStorageBudget,
  event_step_budget : @framed_wire.LunaFramedEventStepBudget,
  blueprint : @core.SchedulerBlueprint,
  binding : @worker_service.WorkerServiceBinding,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  process_limits : @worker_process.WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  restart_policy : @worker_service.WorkerRestartBackoffPolicy,
) -> LunaOnlineFramedServicePreparation raise LunaOnlineInstancePreparationError {
  if framed_limits.inference_limits() != inference {
    raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  let clock = @monotonic_clock.MonotonicClock::system()
  let pool = @request_admission.LunaRequestPreparationPool::new(
    tokenizer,
    expected_tokenizer~,
    expected_model~,
    inference~,
    framed_limits~,
    clock~,
    lane_count=preparation_lane_count,
    step_budget=preparation_step_budget,
    work_limit=preparation_work_limit,
    storage_budget=preparation_storage_budget,
  ) catch {
    _ => raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  let outbound = @framed_wire.LunaFramedEventWorkspace::new(
    framed_limits, event_step_budget,
  ) catch {
    _ => raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  let prepared = prepare_owned_luna_online_instance_with_clock(
    expected_tokenizer, expected_model, inference, blueprint, binding, executable,
    startup, source, process_limits, model_root, kernel_root, clock, restart_policy,
  )
  match prepared.state() {
    LunaOnlineInstanceReady => {
      let instance = prepared.take_ready()
      {
        service: Some(
          new_luna_online_framed_coordinator(
            pool,
            instance,
            outbound,
            event_step_budget.work_units(),
            inference,
            framed_limits,
            expected_model,
            clock,
          ),
        ),
        cleanup: None,
        state: LunaOnlineInstanceReady,
        consumed: false,
      }
    }
    LunaOnlineInstanceCleanupRequired =>
      {
        service: None,
        cleanup: Some(prepared.take_cleanup()),
        state: LunaOnlineInstanceCleanupRequired,
        consumed: false,
      }
  }
}

///|
/// Prepares the standard framed coordinator around one fully admitted local
/// tensor-parallel group. Framed-wire and tokenizer owners are acquired here,
/// at the same coordinator boundary as the single-rank route. The returned
/// preparation carries either the sole ready owner or the sole cleanup
/// obligation; partial rank readiness never reaches the listener boundary.
pub fn prepare_owned_tensor_parallel_luna_online_framed_service(
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  framed_limits : @framed_wire.FramedWireLimits,
  preparation_lane_count : Int,
  preparation_step_budget : @request_admission.LunaRequestPreparationStepBudget,
  preparation_work_limit : @request_admission.LunaRequestPreparationWorkLimit,
  preparation_storage_budget : @request_admission.LunaRequestPreparationStorageBudget,
  event_step_budget : @framed_wire.LunaFramedEventStepBudget,
  blueprint : @core.SchedulerBlueprint,
  inputs : @tensor_parallel_group_transport.TensorParallelGroupPlanningInputs,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  restart_policy : @worker_service.WorkerRestartBackoffPolicy,
) -> LunaOnlineFramedServicePreparation raise LunaOnlineInstancePreparationError {
  if framed_limits.inference_limits() != inference {
    raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  let clock = @monotonic_clock.MonotonicClock::system()
  let pool = @request_admission.LunaRequestPreparationPool::new(
    tokenizer,
    expected_tokenizer~,
    expected_model~,
    inference~,
    framed_limits~,
    clock~,
    lane_count=preparation_lane_count,
    step_budget=preparation_step_budget,
    work_limit=preparation_work_limit,
    storage_budget=preparation_storage_budget,
  ) catch {
    _ => raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  let outbound = @framed_wire.LunaFramedEventWorkspace::new(
    framed_limits, event_step_budget,
  ) catch {
    _ => raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  let prepared = prepare_owned_tensor_parallel_luna_online_instance_with_clock(
    expected_tokenizer, expected_model, inference, blueprint, inputs, executable,
    model_root, kernel_root, clock, restart_policy,
  )
  match prepared.state() {
    LunaOnlineInstanceReady => {
      let instance = prepared.take_ready()
      {
        service: Some(
          new_luna_online_framed_coordinator(
            pool,
            instance,
            outbound,
            event_step_budget.work_units(),
            inference,
            framed_limits,
            expected_model,
            clock,
          ),
        ),
        cleanup: None,
        state: LunaOnlineInstanceReady,
        consumed: false,
      }
    }
    LunaOnlineInstanceCleanupRequired =>
      {
        service: None,
        cleanup: Some(prepared.take_cleanup()),
        state: LunaOnlineInstanceCleanupRequired,
        consumed: false,
      }
  }
}

///|
pub fn LunaOnlineFramedServicePreparation::state(
  self : LunaOnlineFramedServicePreparation,
) -> LunaOnlineInstancePreparationState {
  self.state
}

///|
pub fn LunaOnlineFramedServicePreparation::inference_limits(
  self : LunaOnlineFramedServicePreparation,
) -> @inference.InferenceLimits raise LunaOnlineInstancePreparationError {
  self.require_ready_service().inference
}

///|
pub fn LunaOnlineFramedServicePreparation::framed_limits(
  self : LunaOnlineFramedServicePreparation,
) -> @framed_wire.FramedWireLimits raise LunaOnlineInstancePreparationError {
  self.require_ready_service().framed_limits
}

///|
pub fn LunaOnlineFramedServicePreparation::model_identity(
  self : LunaOnlineFramedServicePreparation,
) -> @spec.ModelIdentity raise LunaOnlineInstancePreparationError {
  self.require_ready_service().expected_model
}

///|
fn LunaOnlineFramedServicePreparation::require_ready_service(
  self : LunaOnlineFramedServicePreparation,
) -> LunaOnlineFramedService raise LunaOnlineInstancePreparationError {
  if self.consumed || self.state != LunaOnlineInstanceReady {
    raise luna_coordinator_preparation_failure(LunaOnlineFramedOwnership)
  }
  match self.service {
    Some(value) => value
    None => abort("Luna framed service preparation invariant")
  }
}

///|
pub fn LunaOnlineFramedServicePreparation::maximum_transport_wait_millis(
  self : LunaOnlineFramedServicePreparation,
) -> Int raise LunaOnlineInstancePreparationError {
  admitted_maximum_transport_wait_millis(self.require_ready_service().inference)
}

///|
/// Resolve the one authenticated maximum transport wait used by framed
/// service preparation and transport policy admission.
pub fn admitted_maximum_transport_wait_millis(
  inference : @inference.InferenceLimits,
) -> Int raise LunaOnlineInstancePreparationError {
  let wait = inference.max_deadline_millis()
  if wait == 0UL || wait > 0x7fffffffUL {
    raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  wait.to_int()
}

///|
pub fn LunaOnlineFramedServicePreparation::take_ready(
  self : LunaOnlineFramedServicePreparation,
) -> LunaOnlineFramedService raise LunaOnlineInstancePreparationError {
  let service = self.require_ready_service()
  self.service = None
  self.consumed = true
  service
}

///|
pub fn LunaOnlineFramedServicePreparation::take_cleanup(
  self : LunaOnlineFramedServicePreparation,
) -> FailedLunaOnlineInstancePreparation raise LunaOnlineInstancePreparationError {
  if self.consumed || self.state != LunaOnlineInstanceCleanupRequired {
    raise luna_coordinator_preparation_failure(LunaOnlineFramedOwnership)
  }
  let cleanup = match self.cleanup {
    Some(value) => value
    None => abort("Luna framed service cleanup invariant")
  }
  self.cleanup = None
  self.consumed = true
  cleanup
}

///|
pub fn prepare_owned_luna_online_framed_coordinator_approved(
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  framed_limits : @framed_wire.FramedWireLimits,
  preparation_lane_count : Int,
  preparation_step_budget : @request_admission.LunaRequestPreparationStepBudget,
  preparation_work_limit : @request_admission.LunaRequestPreparationWorkLimit,
  preparation_storage_budget : @request_admission.LunaRequestPreparationStorageBudget,
  event_step_budget : @framed_wire.LunaFramedEventStepBudget,
  blueprint : @core.SchedulerBlueprint,
  binding : @worker_service.WorkerServiceBinding,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  process_limits : @worker_process.WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  restart_policy : @worker_service.WorkerRestartBackoffPolicy,
) -> LunaOnlineFramedCoordinatorPreparation raise LunaOnlineInstancePreparationError {
  let prepared = prepare_owned_luna_online_framed_service_approved_internal(
    tokenizer, expected_tokenizer, expected_model, inference, framed_limits, preparation_lane_count,
    preparation_step_budget, preparation_work_limit, preparation_storage_budget,
    event_step_budget, blueprint, binding, executable, startup, source, process_limits,
    model_root, kernel_root, restart_policy,
  )
  match prepared.state() {
    LunaOnlineInstanceCleanupRequired =>
      {
        cleanup: Some(prepared.take_cleanup()),
        coordinator: None,
        state: LunaOnlineInstanceCleanupRequired,
        consumed: false,
      }
    LunaOnlineInstanceReady => {
      let service = prepared.take_ready()
      let opened = service.open_stream()
      if opened.kind() != LunaOnlineFramedOpened {
        abort("Luna coordinator initial stream invariant")
      }
      let stream = service.take_open_stream(opened) catch {
        _ =>
          raise luna_coordinator_preparation_failure(LunaOnlineFramedOwnership)
      }
      {
        cleanup: None,
        coordinator: Some({ service, stream }),
        state: LunaOnlineInstanceReady,
        consumed: false,
      }
    }
  }
}

///|
pub fn LunaOnlineFramedCoordinatorPreparation::state(
  self : LunaOnlineFramedCoordinatorPreparation,
) -> LunaOnlineInstancePreparationState {
  self.state
}

///|
/// Returns the authenticated maximum interval for a transport to block before
/// polling the coordinator again. It is valid only while this ready
/// preparation still owns its coordinator.
pub fn LunaOnlineFramedCoordinatorPreparation::maximum_transport_wait_millis(
  self : LunaOnlineFramedCoordinatorPreparation,
) -> Int raise LunaOnlineInstancePreparationError {
  if self.consumed || self.state != LunaOnlineInstanceReady {
    raise luna_coordinator_preparation_failure(LunaOnlineFramedOwnership)
  }
  let coordinator = match self.coordinator {
    Some(value) => value
    None => abort("Luna coordinator preparation invariant")
  }
  let wait = coordinator.service.inference.max_deadline_millis()
  if wait == 0UL || wait > 0x7fffffffUL {
    raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  wait.to_int()
}

///|
pub fn LunaOnlineFramedCoordinatorPreparation::take_ready(
  self : LunaOnlineFramedCoordinatorPreparation,
) -> LunaOnlineFramedCoordinator raise LunaOnlineInstancePreparationError {
  if self.consumed || self.state != LunaOnlineInstanceReady {
    raise luna_coordinator_preparation_failure(LunaOnlineFramedOwnership)
  }
  let coordinator = match self.coordinator {
    Some(value) => value
    None => abort("Luna coordinator preparation invariant")
  }
  self.coordinator = None
  self.consumed = true
  coordinator
}

///|
pub fn LunaOnlineFramedCoordinatorPreparation::take_cleanup(
  self : LunaOnlineFramedCoordinatorPreparation,
) -> FailedLunaOnlineInstancePreparation raise LunaOnlineInstancePreparationError {
  if self.consumed || self.state != LunaOnlineInstanceCleanupRequired {
    raise luna_coordinator_preparation_failure(LunaOnlineFramedOwnership)
  }
  let cleanup = match self.cleanup {
    Some(value) => value
    None => abort("Luna coordinator cleanup invariant")
  }
  self.cleanup = None
  self.consumed = true
  cleanup
}
