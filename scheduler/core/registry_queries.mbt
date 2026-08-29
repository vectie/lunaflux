///|
fn Scheduler::checked_handle_index(
  self : Scheduler,
  handle : RequestHandle,
) -> Int raise SchedulerError {
  let index = handle.slot().index()
  if index < 0 ||
    index >= self.slot_states.length() ||
    self.slot_generations[index] != handle.slot().generation() ||
    self.slot_states[index] == SLOT_FREE ||
    self.slot_states[index] == SLOT_RETIRED {
    raise invalid(Request, Stale)
  }
  guard self.requests[index] is Some(request) else {
    raise invalid(RequestState, InvalidTransition)
  }
  if request.request_id() != handle.request_id() ||
    self.request_generations[index] != handle.request_generation() {
    raise invalid(Request, Stale)
  }
  index
}

///|
fn Scheduler::scheduler_state(
  self : Scheduler,
  slot : Int,
) -> SchedulerRequestState {
  let state = self.slot_states[slot]
  match state {
    SLOT_WAITING =>
      if self.recompute_required[slot] {
        RecomputeWaiting
      } else {
        Waiting
      }
    SLOT_PREFILL =>
      if self.recompute_required[slot] {
        RecomputePrefill
      } else {
        Prefill
      }
    SLOT_DECODE => Decode
    SLOT_FINISHING => Finishing
    SLOT_CANCELLED => Cancelled
    SLOT_DEADLINE => DeadlineExpired
    _ => Failed
  }
}
