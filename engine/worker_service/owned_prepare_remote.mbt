///|
pub struct RemoteWorkerServicePreparation {
  priv outcome : OwnedWorkerServicePreparation
  priv control : RemoteWorkerServiceControl
  priv mut consumed : Bool
}

///|
pub fn RemoteWorkerServicePreparation::control(
  self : RemoteWorkerServicePreparation,
) -> RemoteWorkerServiceControl {
  self.control
}

///|
/// Preallocates the existing scheduler/service shell and its two plan buffers.
/// Ready is published only after both authenticated remote workers are Ready.
pub fn prepare_owned_remote(
  blueprint : @core.SchedulerBlueprint,
  remote : @remote_startup.RemoteGroupStartup,
  admitted : @remote_bootstrap.TensorParallelGroupBootstrap,
  limits : @remote_wire.RankGroupWireLimits,
  timeout_millis~ : UInt64,
  restart_policy : WorkerRestartBackoffPolicy,
) -> RemoteWorkerServicePreparation raise {
  let startup = remote.rank(0).unwrap().worker()
  if startup.identity() != blueprint.model_identity() ||
    startup.model_generation_value() !=
    blueprint.model_plan_generation().value() ||
    startup.worker_limits() != blueprint.worker_protocol_limits() ||
    startup.predecessor_value() != 0UL {
    raise Invalid(Identity)
  }
  let identity = worker_service_identity(
    startup.identity(),
    startup.model_generation_value(),
    startup.bootstrap_source_digest(),
    startup.worker_limits(),
    startup.inference_limits(),
  )
  let scheduler = blueprint.instantiate()
  let (outcome, service, _) = prepare_owned_shell(
    blueprint,
    scheduler,
    identity,
    RemoteBinding,
    None,
    @monotonic_clock.MonotonicClock::system(),
    restart_policy,
  )
  let binding = scheduler.plan_buffer_binding()
  let identities = FixedArray::makei(2, i => {
    if i == 0 {
      binding.plan_a_identity()
    } else {
      binding.plan_b_identity()
    }
  })
  let group = @remote_group.RemoteGroup::new(
    remote,
    admitted,
    identities,
    limits,
    timeout_millis~,
  )
  let control : RemoteWorkerServiceControl = {
    current: group,
    replacement: None,
    replacement_cleanup: false,
    generation_floor: group.generation(),
    identities,
    limits,
    timeout: timeout_millis,
    identity,
    previous: 0UL,
    recovering: false,
    failure_taken: false,
    closed: false,
  }
  service.transport = RemoteGroup(control)
  { outcome, control, consumed: false, }
}

///|
pub fn RemoteWorkerServicePreparation::poll_ready(
  self : RemoteWorkerServicePreparation,
) -> OwnedWorkerServicePreparation? raise {
  if self.consumed {
    raise Invalid(Lifecycle)
  }
  if !self.control.current.poll_startup() {
    return None
  }
  self.outcome.state = OwnedServiceReady
  self.consumed = true
  Some(self.outcome)
}
