///|
pub fn RankGroupProcessSupervisor::received_completion(
  self : RankGroupProcessSupervisor,
) -> ReceivedRankGroupCompletion raise RankGroupProcessError {
  self.require_lifecycle(CompletionReady)
  ReceivedRankGroupCompletion(self, self.exchange_epoch)
}

///|
fn ReceivedRankGroupCompletion::owner(
  self : ReceivedRankGroupCompletion,
) -> RankGroupProcessSupervisor raise RankGroupProcessError {
  if self.1 == 0UL ||
    self.0.exchange_epoch != self.1 ||
    self.0.lifecycle_value != CompletionReady {
    raise Invalid(Owner)
  }
  self.0
}

///|
pub fn ReceivedRankGroupCompletion::frame(
  self : ReceivedRankGroupCompletion,
  owner : RankGroupProcessSupervisor,
) -> @worker_wire.ValidatedCompletionFrame raise RankGroupProcessError {
  let actual = self.owner()
  if !physical_equal(actual, owner) {
    raise Invalid(Owner)
  }
  actual.completion_frame.current() catch {
    error => actual.fail(WorkerWire(error))
  }
}

///|
/// Accepts the scheduler-owned semantic completion while its exact plan and
/// completion epochs are still live. This does not retire the group.
pub fn RankGroupProcessSupervisor::accept_completion(
  self : RankGroupProcessSupervisor,
  received : ReceivedRankGroupCompletion,
  plan : @worker_protocol.SubmittedSchedulePlan,
  completion : @worker_protocol.SubmittedCompletion,
) -> AcceptedRankGroupCompletion raise RankGroupProcessError {
  let owner = received.owner()
  if !physical_equal(owner, self) {
    raise Invalid(Owner)
  }
  let submitted = self.reauthenticate_plan(plan)
  let leader = submitted.lease(plan, 0) catch {
    error => self.fail(Protocol(error))
  }
  let completed = self.protocol.complete_leader(leader, plan, completion) catch {
    error => self.fail(Protocol(error))
  }
  self.accepted_epoch = self.exchange_epoch
  self.lifecycle_value = CompletionAccepted
  ignore(completed)
  AcceptedRankGroupCompletion(self, self.accepted_epoch)
}

///|
/// Rederives the scalar-authenticated retirement capability after scheduler
/// publication backpressure. This never re-reads a scheduler plan or
/// completion and lets callers retain only the physical flight sequence.
pub fn RankGroupProcessSupervisor::accepted_completion_for_sequence(
  self : RankGroupProcessSupervisor,
  sequence : UInt64,
) -> AcceptedRankGroupCompletion raise RankGroupProcessError {
  self.require_lifecycle(CompletionAccepted)
  if sequence == 0UL ||
    sequence != self.active_sequence_value ||
    self.accepted_epoch == 0UL ||
    self.accepted_epoch != self.exchange_epoch {
    raise Invalid(Owner)
  }
  ignore(
    self.protocol.reauthenticate_completed(self.accepted_epoch) catch {
      error => self.fail(Protocol(error))
    },
  )
  AcceptedRankGroupCompletion(self, self.accepted_epoch)
}

///|
/// Retires only the already validated group result. The caller invokes this
/// after scheduler commit; no now-stale scheduler plan/completion is re-read.
pub fn RankGroupProcessSupervisor::retire_after_scheduler_commit(
  self : RankGroupProcessSupervisor,
  accepted : AcceptedRankGroupCompletion,
) -> Unit raise RankGroupProcessError {
  self.require_lifecycle(CompletionAccepted)
  if !physical_equal(accepted.0, self) ||
    accepted.1 == 0UL ||
    accepted.1 != self.accepted_epoch {
    raise Invalid(Owner)
  }
  let completed = self.protocol.reauthenticate_completed(accepted.1) catch {
    error => self.fail(Protocol(error))
  }
  self.protocol.retire_completed(completed) catch {
    error => self.fail(Protocol(error))
  }
  self.active_slot_code = -1
  self.active_sequence_value = 0UL
  self.encoded_plan_length = 0
  self.active_frame_deadline_millis = 0L
  self.leader_received = false
  self.graph_telemetry.clear()
  for rank in 0..<self.rank_sidecar_pending.length() {
    self.rank_sidecar_pending[rank] = false
  }
  self.accepted_epoch = 0UL
  self.active_deadline_millis = 0UL
  self.lifecycle_value = Ready
}
