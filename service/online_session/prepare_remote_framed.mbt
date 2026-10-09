///|
pub struct RemoteLunaFramedPreparation {
  priv remote : RemoteLunaOnlinePreparation
  priv pool : @request_admission.LunaRequestPreparationPool
  priv outbound : @framed_wire.LunaFramedEventWorkspace
  priv framed_limits : @framed_wire.FramedWireLimits
  priv event_work_units : Int
  priv clock : @monotonic_clock.MonotonicClock
}

///|
/// Preallocates normal tokenizer/framed-request/event owners for a remote
/// preparation. Its ready result feeds existing native and OpenAI TCP servers.
pub fn RemoteLunaOnlinePreparation::framed(
  self : RemoteLunaOnlinePreparation,
  tokenizer : @tokenizer.TokenizerSpec,
  framed_limits : @framed_wire.FramedWireLimits,
  preparation_lane_count : Int,
  preparation_step_budget : @request_admission.LunaRequestPreparationStepBudget,
  preparation_work_limit : @request_admission.LunaRequestPreparationWorkLimit,
  preparation_storage_budget : @request_admission.LunaRequestPreparationStorageBudget,
  event_step_budget : @framed_wire.LunaFramedEventStepBudget,
) -> RemoteLunaFramedPreparation raise {
  if self.consumed ||
    framed_limits.inference_limits() != self.instance.inference {
    raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
  let clock = @monotonic_clock.MonotonicClock::system()
  let pool = @request_admission.LunaRequestPreparationPool::new(
    tokenizer,
    expected_tokenizer=self.instance.expected_tokenizer,
    expected_model=self.instance.expected_model,
    inference=self.instance.inference,
    framed_limits~,
    clock~,
    lane_count=preparation_lane_count,
    step_budget=preparation_step_budget,
    work_limit=preparation_work_limit,
    storage_budget=preparation_storage_budget,
  )
  let outbound = @framed_wire.LunaFramedEventWorkspace::new(
    framed_limits, event_step_budget,
  )
  {
    remote: self,
    pool,
    outbound,
    framed_limits,
    event_work_units: event_step_budget.work_units(),
    clock,
  }
}

///|
pub fn RemoteLunaFramedPreparation::control(
  self : RemoteLunaFramedPreparation,
) -> @worker_service.RemoteWorkerServiceControl {
  self.remote.control()
}

///|
pub fn RemoteLunaFramedPreparation::poll_ready(
  self : RemoteLunaFramedPreparation,
) -> LunaOnlineFramedServicePreparation? raise {
  guard self.remote.poll_ready() is Some(prepared) else { return None }
  let instance = prepared.take_ready()
  Some({
    service: Some(
      new_luna_online_framed_coordinator(
        self.pool,
        instance,
        self.outbound,
        self.event_work_units,
        instance.inference,
        self.framed_limits,
        instance.expected_model,
        self.clock,
      ),
    ),
    cleanup: None,
    state: LunaOnlineInstanceReady,
    consumed: false,
  })
}
