///|
fn owned_failure(
  failure : OwnedWorkerServiceFailure,
) -> OwnedWorkerServicePreparationError {
  OwnedPreparationFailure(failure)
}

///|
fn initial_primary(
  failed : @worker_process.FailedRootBoundWorkerProcessStartup,
) -> OwnedWorkerServiceFailure {
  OwnedPhysicalTransportFailure(
    physical_transport_failure(RootBoundFailure(failed.cause())),
  )
}

///|
fn initial_cleanup_evidence(
  failed : @worker_process.FailedRootBoundWorkerProcessStartup,
) -> OwnedWorkerServiceFailure? {
  match (failed.child_cleanup_failure(), failed.roots_cleanup_failure()) {
    (Some(_), Some(_)) | (Some(_), None) | (None, Some(_)) =>
      Some(OwnedPhysicalTransportFailure(PhysicalCleanupFailed))
    (None, None) => None
  }
}

///|
fn validate_owned_preflight(
  blueprint : @core.SchedulerBlueprint,
  binding : WorkerServiceBinding,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> WorkerServiceIdentity raise OwnedWorkerServicePreparationError {
  if source.digest() != startup.bootstrap_source_digest() {
    raise owned_failure(OwnedServiceFailure(Invalid(BootstrapSourceIdentity)))
  }
  if startup.identity() != blueprint.model_identity() {
    raise owned_failure(OwnedServiceFailure(Invalid(Identity)))
  }
  if startup.model_generation_value() !=
    blueprint.model_plan_generation().value() {
    raise owned_failure(OwnedServiceFailure(Invalid(Generation)))
  }
  if startup.worker_limits() != blueprint.worker_protocol_limits() {
    raise owned_failure(OwnedServiceFailure(Invalid(WorkerEnvelope)))
  }
  if startup.bootstrap_digest() != binding.bootstrap_digest() {
    raise owned_failure(OwnedServiceFailure(Invalid(BootstrapIdentity)))
  }
  if startup.bootstrap_source_digest() != binding.bootstrap_source_digest() {
    raise owned_failure(OwnedServiceFailure(Invalid(BootstrapSourceIdentity)))
  }
  if startup.device_ordinal() != binding.device_ordinal() {
    raise owned_failure(OwnedServiceFailure(Invalid(DeviceAssignment)))
  }
  if startup.inference_limits() != binding.inference_limits() {
    raise owned_failure(OwnedServiceFailure(Invalid(InferenceEnvelope)))
  }
  if startup.has_predecessor() || startup.predecessor_value() != 0UL {
    raise owned_failure(OwnedServiceFailure(Invalid(Sequence)))
  }
  worker_service_identity(
    blueprint.model_identity(),
    blueprint.model_plan_generation().value(),
    source.digest(),
    blueprint.worker_protocol_limits(),
    binding.inference_limits(),
  ) catch {
    error => raise owned_failure(OwnedServiceFailure(error))
  }
}

///|
fn publish_cleanup(
  prepared : OwnedWorkerServicePreparation,
) -> OwnedWorkerServicePreparation {
  prepared.state = OwnedServiceCleanupRequired
  prepared
}

///|
fn install_owned_ready(
  prepared : OwnedWorkerServicePreparation,
  service : WorkerService,
  process : @worker_process.RootBoundWorkerProcessSupervisor,
) -> OwnedWorkerServicePreparation {
  service.transport = rooted_single_transport(process)
  prepared.state = OwnedServiceReady
  prepared
}

///|
/// Allocates the scheduler/service/online publication family once, before any
/// physical root is retained. Both physical preparations use this exact shell.
fn prepare_owned_shell(
  blueprint : @core.SchedulerBlueprint,
  scheduler : @core.Scheduler,
  identity : WorkerServiceIdentity,
  physical_binding : WorkerPhysicalBinding,
  online_request : @core.TokenizedRequest?,
  online_clock : @monotonic_clock.MonotonicClock,
  restart_policy : WorkerRestartBackoffPolicy,
) -> (
  OwnedWorkerServicePreparation,
  WorkerService,
  FailedOwnedWorkerServicePreparation,
) raise OwnedWorkerServicePreparationError {
  let prepared_admission = match online_request {
    Some(request) =>
      Some(
        scheduler.prepare_exclusive_admission(request) catch {
          error => raise owned_failure(OwnedSchedulerFailure(error))
        },
      )
    None => None
  }
  let online_lane_count = blueprint.request_slot_capacity()
  let service : WorkerService = {
    scheduler,
    identity,
    physical_binding,
    transport: Uninstalled,
    flight: FlightEmpty,
    pending_plan_sequence: 0UL,
    maintenance_mode: MAINTENANCE_NONE,
    maintenance_phase: 0,
    maintenance_failure: ExecutionFailed,
    lifecycle: ServiceReady,
    production_owned: true,
    ownership_mode: OnlineEligibleOwnership,
    online_epoch: 0UL,
    online_publication_cursor: 0UL,
    graph_telemetry: new_worker_service_graph_telemetry(),
  }
  let failed_owner : FailedOwnedWorkerServicePreparation = {
    initial: None,
    ready_process: None,
    group: None,
    cause_kind: 0,
    service_cause: None,
    process_cause: None,
    cleanup_process_failure: None,
    cleanup_group_failure: false,
    closed: false,
    reported_consumed_failure: false,
  }
  let online : OnlineWorkerLease = {
    service,
    epoch: 0UL,
    clock: online_clock,
    restart_policy,
    restart_attempts: 0,
    restart_generation: 0UL,
    restart_phase: 0,
    restart_delay_millis: 0UL,
    restart_deadline_millis: 0UL,
    restart_cached_remaining_millis: 0,
    restart_stability_generation: 0UL,
    restart_stability_deadline_millis: 0UL,
    restart_stability_predecessor: 0UL,
    restart_clock_initialized: false,
    restart_last_clock_millis: 0UL,
    restart_forbidden: false,
    prepared_admission,
    prepared_clock_read: online_clock.prepare_read(),
    lane_phases: FixedArray::make(online_lane_count, 0),
    lane_generations: FixedArray::make(online_lane_count, 0UL),
    lane_handles: FixedArray::makei(online_lane_count, _ => {
      Array::new(capacity=1)
    }),
    lane_semantic_retentions: FixedArray::makei(online_lane_count, _ => {
      @inference.LunaRequestStopTokenRetentionSlot::new()
    }),
    lane_request_generations: FixedArray::make(
      online_lane_count,
      @inference.RequestGeneration::initial(),
    ),
    lane_cut_source_generations: FixedArray::make(
      online_lane_count,
      @inference.RequestGeneration::initial(),
    ),
    lane_restart_dispositions: FixedArray::make(online_lane_count, 0),
    lane_expected_positions: FixedArray::make(online_lane_count, 0),
    free_lanes: FixedArray::makei(online_lane_count, index => index),
    free_head: 0,
    free_count: online_lane_count,
    active_lane_count: 0,
    compatibility_lane: -1,
    compatibility_generation: 0UL,
    publication_cursor: 0UL,
    cancel_lane: -1,
    cancel_generation: 0UL,
  }
  let outcome : OwnedWorkerServicePreparation = {
    service: Some(service),
    online: Some(online),
    cleanup: failed_owner,
    state: OwnedServiceCleanupRequired,
    consumed: false,
  }
  (outcome, service, failed_owner)
}

///|
/// Constructs both owners before root acquisition; ordinary roots are borrowed.
fn prepare_owned_physical(
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  process_limits : @worker_process.WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  predecessor : UInt64,
  parent_approval : @worker_wire.WorkerParentApprovalSubjects?,
) -> @worker_process.RootBoundWorkerProcessPreparation raise OwnedWorkerServicePreparationError {
  @worker_process.prepare_exchange_with_approved_executable(
    executable,
    startup,
    source,
    process_limits,
    model_root,
    kernel_root,
    predecessor,
    parent_approval~,
  ) catch {
    error =>
      raise owned_failure(
        OwnedPhysicalTransportFailure(physical_transport_failure(error)),
      )
  }
}

///|
fn prepare_owned_internal(
  blueprint : @core.SchedulerBlueprint,
  binding : WorkerServiceBinding,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  process_limits : @worker_process.WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  online_request : @core.TokenizedRequest?,
  online_clock : @monotonic_clock.MonotonicClock,
  restart_policy : WorkerRestartBackoffPolicy,
) -> OwnedWorkerServicePreparation raise OwnedWorkerServicePreparationError {
  let identity = validate_owned_preflight(blueprint, binding, startup, source)
  let scheduler = blueprint.instantiate() catch {
    error => raise owned_failure(OwnedSchedulerFailure(error))
  }
  let (outcome, service, failed_owner) = prepare_owned_shell(
    blueprint,
    scheduler,
    identity,
    RootedSingleBinding(binding),
    online_request,
    online_clock,
    restart_policy,
  )
  let rooted = prepare_owned_physical(
    executable,
    startup,
    source,
    process_limits,
    model_root,
    kernel_root,
    scheduler.last_retired_plan_sequence(),
    binding.parent_approval(),
  )
  match rooted {
    RootBoundCleanupRequired(failed) => {
      failed_owner.initial = Some(failed)
      failed_owner.cause_kind = 1
      failed_owner.closed = failed.is_closed()
      publish_cleanup(outcome)
    }
    ReadyRootBoundProcess(process) => {
      if !service.production_owned {
        abort("owned service provenance invariant")
      }
      install_owned_ready(outcome, service, process)
    }
  }
}

///|
/// Production preparation from one exact pinned worker executable.
pub fn prepare_owned_approved(
  blueprint : @core.SchedulerBlueprint,
  binding : WorkerServiceBinding,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  process_limits : @worker_process.WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  restart_policy : WorkerRestartBackoffPolicy,
) -> OwnedWorkerServicePreparation raise OwnedWorkerServicePreparationError {
  prepare_owned_internal(
    blueprint,
    binding,
    executable,
    startup,
    source,
    process_limits,
    model_root,
    kernel_root,
    None,
    @monotonic_clock.MonotonicClock::system(),
    restart_policy,
  )
}

///|
pub fn OwnedWorkerServicePreparation::state(
  self : OwnedWorkerServicePreparation,
) -> OwnedWorkerServicePreparationState {
  self.state
}

///|
fn publish_raw_transfer(
  prepared : OwnedWorkerServicePreparation,
  service : WorkerService,
) -> WorkerService {
  service.ownership_mode = FixtureRawOwnership
  prepared.consumed = true
  prepared.service = None
  prepared.online = None
  service
}

///|
fn publish_online_transfer(
  prepared : OwnedWorkerServicePreparation,
  service : WorkerService,
  online : OnlineWorkerLease,
  clock : @monotonic_clock.MonotonicClock,
  next_epoch : UInt64,
) -> OnlineWorkerLease {
  online.clock = clock
  online.epoch = next_epoch
  online.publication_cursor = service.online_publication_cursor
  service.online_epoch = next_epoch
  service.ownership_mode = OnlineActiveOwnership
  prepared.consumed = true
  prepared.service = None
  prepared.online = None
  online
}

///|
/// Transfers the ready service once. This lower-level migration/test seam is
/// boundary-restricted; the owned online aggregate uses `take_online`.
pub fn OwnedWorkerServicePreparation::take_raw_ready(
  self : OwnedWorkerServicePreparation,
) -> WorkerService raise OwnedWorkerServicePreparationError {
  if self.consumed || self.state != OwnedServiceReady {
    raise owned_failure(OwnedServiceFailure(Invalid(Lifecycle)))
  }
  match self.service {
    Some(service) => {
      let online = match self.online {
        Some(value) => value
        None => abort("owned raw lease shell invariant")
      }
      if online.prepared_admission is Some(_) || !service.can_claim_raw() {
        raise owned_failure(OwnedServiceFailure(Invalid(OwnershipMode)))
      }
      publish_raw_transfer(self, service)
    }
    None => abort("owned ready service invariant")
  }
}

///|
/// Atomically transfers a fresh production owner into its permanent online
/// family. The exact lease shell was allocated before root activation; after
/// the complete preflight, publication performs only nonraising assignments.
pub fn OwnedWorkerServicePreparation::take_online(
  self : OwnedWorkerServicePreparation,
  clock : @monotonic_clock.MonotonicClock,
) -> OnlineWorkerLease raise OwnedWorkerServicePreparationError {
  if self.consumed || self.state != OwnedServiceReady {
    raise owned_failure(OwnedServiceFailure(Invalid(Lifecycle)))
  }
  let service = match self.service {
    Some(value) => value
    None => abort("owned online service invariant")
  }
  if !service.production_owned ||
    service.ownership_mode != OnlineEligibleOwnership ||
    service.online_epoch == 0xffffffffffffffffUL ||
    service.lifecycle != ServiceReady ||
    service.raw_outstanding_count() != 0 ||
    !service.require_transport().has_exchange_capacity() {
    raise owned_failure(OwnedServiceFailure(Invalid(OwnershipMode)))
  }
  let next_epoch = service.online_epoch + 1UL
  let online = match self.online {
    Some(value) => value
    None => abort("owned online lease invariant")
  }
  if online.prepared_admission is Some(_) ||
    online.active_lane_count != 0 ||
    online.free_count != online.lane_phases.length() ||
    !service.scheduler.is_pristine() {
    raise owned_failure(OwnedServiceFailure(Invalid(OwnershipMode)))
  }
  publish_online_transfer(self, service, online, clock, next_epoch)
}

///|
/// Transfers only the owned-online preparation carrying the exclusive request
/// shell created before rooted worker activation.
pub fn OwnedWorkerServicePreparation::take_prepared_online(
  self : OwnedWorkerServicePreparation,
) -> OnlineWorkerLease raise OwnedWorkerServicePreparationError {
  if self.consumed || self.state != OwnedServiceReady {
    raise owned_failure(OwnedServiceFailure(Invalid(Lifecycle)))
  }
  let service = match self.service {
    Some(value) => value
    None => abort("owned prepared online service invariant")
  }
  let online = match self.online {
    Some(value) => value
    None => abort("owned prepared online lease invariant")
  }
  if !service.production_owned ||
    service.ownership_mode != OnlineEligibleOwnership ||
    service.online_epoch == 0xffffffffffffffffUL ||
    service.lifecycle != ServiceReady ||
    service.raw_outstanding_count() != 0 ||
    !service.require_transport().has_exchange_capacity() ||
    online.prepared_admission is None ||
    online.active_lane_count != 0 ||
    online.free_count != online.lane_phases.length() {
    raise owned_failure(OwnedServiceFailure(Invalid(OwnershipMode)))
  }
  let exclusive = match online.prepared_admission {
    Some(value) => value
    None => abort("owned prepared admission invariant")
  }
  if !service.scheduler.has_exclusive_admission(exclusive) {
    raise owned_failure(OwnedServiceFailure(Invalid(OwnershipMode)))
  }
  publish_online_transfer(
    self,
    service,
    online,
    online.clock,
    service.online_epoch + 1UL,
  )
}
