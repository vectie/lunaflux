///|
pub fn RankGroupProcessSupervisor::lifecycle(
  self : RankGroupProcessSupervisor,
) -> RankGroupProcessLifecycle {
  self.lifecycle_value
}

///|
pub fn RankGroupProcessSupervisor::world_size(
  self : RankGroupProcessSupervisor,
) -> Int {
  self.children.length()
}

///|
pub fn RankGroupProcessSupervisor::group_generation_value(
  self : RankGroupProcessSupervisor,
) -> UInt64 {
  self.protocol.generation().value()
}

///|
pub fn RankGroupProcessSupervisor::failure(
  self : RankGroupProcessSupervisor,
) -> RankGroupProcessFailure? {
  self.failure_value
}

///|
pub fn RankGroupProcessSupervisor::cleanup_failure(
  self : RankGroupProcessSupervisor,
) -> RankGroupProcessCleanupFailure? {
  self.cleanup_failure_value
}

///|
fn RankGroupProcessSupervisor::record_cleanup_failure(
  self : RankGroupProcessSupervisor,
  failure : RankGroupProcessCleanupFailure,
) -> Unit {
  if self.cleanup_failure_value is None {
    self.cleanup_failure_value = Some(failure)
  }
}

///|
fn[T] RankGroupProcessSupervisor::reject(
  self : RankGroupProcessSupervisor,
  rule : RankGroupProcessRule,
) -> T raise RankGroupProcessError {
  match self.failure_value {
    Some(failure) => raise Failed(failure)
    None => raise Invalid(rule)
  }
}

///|
fn RankGroupProcessSupervisor::record_failure(
  self : RankGroupProcessSupervisor,
  failure : RankGroupProcessFailure,
) -> Unit {
  if self.failure_value is None {
    self.failure_value = Some(failure)
  }
  if self.lifecycle_value != Faulted &&
    self.lifecycle_value != Cleaning &&
    self.lifecycle_value != Closed {
    let failed_rank = self.io_rank
    let lost_rank = match failure {
      Spawn(rank~)
      | StartupTransport(rank~)
      | TransportLost(rank~)
      | IdleRankExit(rank~, ..) => rank
      _ => failed_rank
    }
    if lost_rank >= 0 && lost_rank < self.cleanup_lost.length() {
      self.cleanup_lost[lost_rank] = true
    }
    self.healthy_terminal = false
    self.lifecycle_value = Faulted
    self.io_rank = -1
    self.io_phase = 0
    self.wire_transcript.abandon_on_transport_loss()
    self.seal_protocol_failure(lost_rank)
  }
}

///|
fn RankGroupProcessSupervisor::seal_protocol_failure(
  self : RankGroupProcessSupervisor,
  failed_rank : Int,
) -> Unit {
  if self.protocol.lifecycle() == Executing {
    self.protocol.report_active_host_failure() catch {
      _ => ()
    }
  } else if self.protocol.lifecycle() == Ready &&
    (failed_rank < 0 || failed_rank >= self.children.length()) {
    self.protocol.report_idle_host_failure() catch {
      _ => ()
    }
  }
  if self.protocol.lifecycle() == Faulted {
    try self.protocol.take_failure() catch {
      _ => ()
    } noraise {
      report => self.group_failure = Some(report)
    }
    if failed_rank >= 0 && failed_rank < self.children.length() {
      self.protocol.mark_drain_rank_lost(failed_rank) catch {
        _ => ()
      }
    }
  } else if self.protocol.lifecycle() == Ready {
    if failed_rank >= 0 && failed_rank < self.children.length() {
      self.protocol.report_idle_rank_loss(failed_rank) catch {
        _ => ()
      }
      try self.protocol.take_failure() catch {
        _ => ()
      } noraise {
        report => self.group_failure = Some(report)
      }
    }
  }
}

///|
/// Non-consuming view used by the outer group owner to authenticate the exact
/// report before committing the lower one-shot publication.
pub fn RankGroupProcessSupervisor::inspect_group_failure(
  self : RankGroupProcessSupervisor,
) -> @rank_group_protocol.RankGroupFailureReport raise RankGroupProcessError {
  if self.group_failure_taken {
    raise Invalid(Lifecycle)
  }
  match self.group_failure {
    Some(report) => report
    None => self.reject(Lifecycle)
  }
}

///|
pub fn RankGroupProcessSupervisor::take_group_failure(
  self : RankGroupProcessSupervisor,
) -> @rank_group_protocol.RankGroupFailureReport raise RankGroupProcessError {
  if self.group_failure_taken {
    raise Invalid(Lifecycle)
  }
  match self.group_failure {
    Some(report) => {
      self.group_failure_taken = true
      report
    }
    None => self.reject(Lifecycle)
  }
}

///|
fn[T] RankGroupProcessSupervisor::fail(
  self : RankGroupProcessSupervisor,
  failure : RankGroupProcessFailure,
) -> T raise RankGroupProcessError {
  self.record_failure(failure)
  raise Failed(self.failure_value.unwrap())
}

///|
fn RankGroupProcessSupervisor::require_lifecycle(
  self : RankGroupProcessSupervisor,
  expected : RankGroupProcessLifecycle,
) -> Unit raise RankGroupProcessError {
  if self.lifecycle_value != expected {
    self.reject(Lifecycle)
  }
}

///|
fn RankGroupProcessSupervisor::binding(
  self : RankGroupProcessSupervisor,
  rank : Int,
) -> @rank_group_wire.RankGroupWireBinding {
  match self.wire_transcript.binding(rank) {
    Some(binding) => binding
    None => abort("rank-group process binding invariant")
  }
}

///|
fn RankGroupProcessSupervisor::active_slot(
  self : RankGroupProcessSupervisor,
) -> @rank_group_wire.RankGroupWireSlot raise RankGroupProcessError {
  match self.active_slot_code {
    0 => A
    1 => B
    _ => self.reject(Submission)
  }
}

///|
fn RankGroupProcessSupervisor::slot_for_plan(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> @rank_group_wire.RankGroupWireSlot raise RankGroupProcessError {
  let identity = plan.buffer_identity() catch { _ => self.reject(Submission) }
  for index in 0..<self.plan_identities.length() {
    if self.plan_identities[index].same_owner(identity) {
      return if index == 0 { A } else { B }
    }
  }
  self.reject(Submission)
}

///|
fn RankGroupProcessSupervisor::reauthenticate_plan(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> @rank_group_protocol.SubmittedRankGroupPlan raise RankGroupProcessError {
  let sequence = plan.sequence_wire() catch { _ => self.reject(Submission) }
  let slot = self.slot_for_plan(plan)
  let slot_code = if slot == A { 0 } else { 1 }
  if sequence != self.active_sequence_value ||
    slot_code != self.active_slot_code {
    self.reject(Submission)
  }
  self.protocol.reauthenticate_submitted(plan) catch {
    error => self.fail(Protocol(error))
  }
}

///|
fn RankGroupProcessSupervisor::authenticate_active_scalars(
  self : RankGroupProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Unit raise RankGroupProcessError {
  let sequence = plan.sequence_wire() catch { _ => self.reject(Submission) }
  let slot = self.slot_for_plan(plan)
  let slot_code = if slot == A { 0 } else { 1 }
  if sequence != self.active_sequence_value ||
    slot_code != self.active_slot_code {
    self.reject(Submission)
  }
}
