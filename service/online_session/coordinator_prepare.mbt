///|
fn luna_coordinator_preparation_failure(
  rule : LunaOnlineFramedCoordinatorRule,
) -> LunaOnlineInstancePreparationError {
  let failure = match rule {
    LunaOnlineFramedOwnership => LunaOnlineInvalid(LunaOnlineOwnership)
    _ => LunaOnlineServicePreparationFailure
  }
  LunaOnlineInstancePreparationFailed(failure)
}

///|
fn new_luna_online_framed_coordinator(
  pool : @request_admission.LunaRequestPreparationPool,
  instance : LunaOnlineInstance,
  outbound : @framed_wire.LunaFramedEventWorkspace,
  event_copy_quantum : Int,
  inference : @inference.InferenceLimits,
  clock : @monotonic_clock.MonotonicClock,
) -> LunaOnlineFramedCoordinator {
  let lane_count = pool.lane_count()
  {
    pool,
    instance,
    outbound,
    event_copy_quantum,
    inference,
    clock,
    work_slots: FixedArray::makei(lane_count, _ => Array::new(capacity=1)),
    queue_sequences: FixedArray::make(lane_count, 0UL),
    prepared: Array::new(capacity=1),
    tickets: Array::new(capacity=1),
    event_credits: Array::new(capacity=1),
    outbound_works: Array::new(capacity=1),
    outbound_views: Array::new(capacity=1),
    queue_head: 0,
    queue_count: 0,
    receipt_slot: -1,
    next_sequence: 0UL,
    event_kind: 0,
    event_epoch: 0UL,
    offer_epoch: 0UL,
    event_length: 0,
    confirmed_bytes: 0,
    offered_bytes: 0,
    semantic_event_pending: false,
    event_ack_pending: false,
    maintenance_kind: 0,
    rejection_epoch: 0UL,
    rejection_sequence: 0UL,
    rejection_rule: 0,
    rejection_taken: false,
    stall_deadline_millis: 0UL,
    stall_active: false,
    turn_pool: true,
    drain_requested: false,
    disconnect_requested: false,
    closed: false,
  }
}

///|
/// Constructs one transport-neutral coordinator and every inner owner. The
/// same monotonic clock governs first-byte receipt, online admission, and
/// framed-output stall enforcement.
pub fn prepare_owned_luna_online_framed_coordinator(
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
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  process_limits : @worker_process.WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> LunaOnlineFramedCoordinatorPreparation raise LunaOnlineInstancePreparationError {
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
    startup, source, process_limits, model_root, kernel_root, clock,
  )
  match prepared.state() {
    LunaOnlineInstanceReady => {
      let instance = prepared.take_ready()
      {
        coordinator: Some(
          new_luna_online_framed_coordinator(
            pool,
            instance,
            outbound,
            event_step_budget.work_units(),
            inference,
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
        coordinator: None,
        cleanup: Some(prepared.take_cleanup()),
        state: LunaOnlineInstanceCleanupRequired,
        consumed: false,
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
