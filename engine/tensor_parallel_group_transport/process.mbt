///|
fn TensorParallelGroupTransport::published_process(
  self : TensorParallelGroupTransport,
) -> @rank_group_process.RankGroupProcessSupervisor raise TensorParallelGroupTransportError {
  match self.state {
    Published(supervisor) => {
      if supervisor.lifecycle() == Faulted {
        self.state = Failed(supervisor)
        rejected(Process)
      }
      supervisor
    }
    _ => rejected(Lifecycle)
  }
}

///|
fn[T] TensorParallelGroupTransport::process_error(
  self : TensorParallelGroupTransport,
  supervisor : @rank_group_process.RankGroupProcessSupervisor,
  rule? : TensorParallelGroupTransportRule = Process,
) -> T raise TensorParallelGroupTransportError {
  if supervisor.lifecycle() == Faulted {
    self.state = Failed(supervisor)
  }
  rejected(rule)
}

///|
pub fn TensorParallelGroupTransport::start(
  self : TensorParallelGroupTransport,
) -> TensorParallelGroupStart raise TensorParallelGroupTransportError {
  match self.state {
    NoGroup => ()
    _ => rejected(Lifecycle)
  }
  if self.active_sequence_value != 0UL {
    rejected(Lifecycle)
  }
  if self.invalid_failure_report ||
    (self.retained_failure_report is Some(_) && !self.failure_report_taken) {
    rejected(Process)
  }
  self.retained_failure_report = None
  self.failure_report_taken = false
  self.invalid_failure_report = false
  let generation = self.mint_group_generation()
  let attempt = self.build_attempt(generation)
  let prepared = @rank_group_process.prepare_rank_group_process_approved(
    self.approved_executable,
    attempt.startups,
    self.plan_buffer_identities,
    previous_plan_sequence_value=self.current_previous_plan_sequence_value,
    self.inputs.process_limits,
  ) catch {
    _ => rejected(Process)
  }
  let spawn_authority = @approved_fs.worker_approved_root_spawn_authority(
    self.roots,
  )
  let preparation = prepared.spawn_with_approved_root_spawn_authority(
    spawn_authority,
  ) catch {
    _ => rejected(Process)
  }
  match preparation {
    ReadyRankGroupProcess(supervisor) => {
      self.state = Published(supervisor)
      self.replacement_required = false
      GroupReady
    }
    RankGroupProcessCleanupRequired(supervisor) => {
      if supervisor.lifecycle() == Faulted {
        self.state = Failed(supervisor)
        self.retain_failure_before_cleanup(supervisor)
        supervisor.begin_cleanup() catch {
          _ => {
            self.state = Failed(supervisor)
            rejected(Cleanup)
          }
        }
        self.state = Cleaning(supervisor, replace_after_cleanup=true)
      } else {
        self.state = Failed(supervisor)
      }
      GroupCleanupRequired
    }
  }
}

///|
pub fn TensorParallelGroupTransport::begin(
  self : TensorParallelGroupTransport,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Unit raise TensorParallelGroupTransportError {
  if self.active_sequence_value != 0UL {
    rejected(Lifecycle)
  }
  let supervisor = self.published_process()
  let sequence = plan.sequence_wire() catch { _ => rejected(Completion) }
  self.active_sequence_value = sequence
  self.completion_accepted = false
  supervisor.begin(plan) catch {
    _ => {
      if supervisor.lifecycle() != Faulted {
        self.active_sequence_value = 0UL
      }
      self.process_error(supervisor)
    }
  }
}

///|
pub fn TensorParallelGroupTransport::progress(
  self : TensorParallelGroupTransport,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> @rank_group_process.RankGroupProcessProgress raise TensorParallelGroupTransportError {
  let supervisor = self.published_process()
  supervisor.progress(plan) catch {
    _ => self.process_error(supervisor)
  }
}

///|
pub fn TensorParallelGroupTransport::received_completion(
  self : TensorParallelGroupTransport,
) -> TensorParallelGroupReceivedCompletion raise TensorParallelGroupTransportError {
  let supervisor = self.published_process()
  let received = supervisor.received_completion() catch {
    _ => self.process_error(supervisor, rule=Completion)
  }
  TensorParallelGroupReceivedCompletion(received)
}

///|
pub fn TensorParallelGroupTransport::completion_frame(
  self : TensorParallelGroupTransport,
  received : TensorParallelGroupReceivedCompletion,
) -> @worker_wire.ValidatedCompletionFrame raise TensorParallelGroupTransportError {
  let supervisor = self.published_process()
  received.0.frame(supervisor) catch {
    _ => self.process_error(supervisor, rule=Completion)
  }
}

///|
/// Returns the one logical report admitted only after every rank supplied an
/// exactly equal per-rank sidecar for this completed group sequence.
pub fn TensorParallelGroupTransport::graph_telemetry(
  self : TensorParallelGroupTransport,
  received : TensorParallelGroupReceivedCompletion,
) -> @worker_wire.WorkerGraphTelemetry raise TensorParallelGroupTransportError {
  let supervisor = self.published_process()
  received.0.graph_telemetry(supervisor) catch {
    _ => self.process_error(supervisor, rule=Completion)
  }
}

///|
pub fn TensorParallelGroupTransport::accept_completion(
  self : TensorParallelGroupTransport,
  received : TensorParallelGroupReceivedCompletion,
  plan : @worker_protocol.SubmittedSchedulePlan,
  completion : @worker_protocol.SubmittedCompletion,
) -> TensorParallelGroupAcceptedCompletion raise TensorParallelGroupTransportError {
  let supervisor = self.published_process()
  let sequence = plan.sequence_wire() catch {
    _ => self.process_error(supervisor, rule=Completion)
  }
  if sequence != self.active_sequence_value {
    self.process_error(supervisor, rule=Completion)
  }
  let accepted = supervisor.accept_completion(received.0, plan, completion) catch {
    _ => self.process_error(supervisor, rule=Completion)
  }
  self.completion_accepted = true
  TensorParallelGroupAcceptedCompletion(accepted)
}

///|
pub fn TensorParallelGroupTransport::accepted_completion_for_sequence(
  self : TensorParallelGroupTransport,
  sequence : UInt64,
) -> TensorParallelGroupAcceptedCompletion raise TensorParallelGroupTransportError {
  if !self.completion_accepted || self.active_sequence_value != sequence {
    rejected(Completion)
  }
  let supervisor = self.published_process()
  let accepted = supervisor.accepted_completion_for_sequence(sequence) catch {
    _ => self.process_error(supervisor, rule=Completion)
  }
  TensorParallelGroupAcceptedCompletion(accepted)
}

///|
fn TensorParallelGroupTransport::advance_scheduler_predecessor(
  self : TensorParallelGroupTransport,
  sequence : UInt64,
) -> Unit raise TensorParallelGroupTransportError {
  let previous = self.current_previous_plan_sequence_value
  if previous == 0xffffffffffffffffUL || sequence != previous + 1UL {
    rejected(SchedulerPredecessor)
  }
  self.current_previous_plan_sequence_value = sequence
}

///|
pub fn TensorParallelGroupTransport::retire_after_scheduler_commit(
  self : TensorParallelGroupTransport,
  accepted : TensorParallelGroupAcceptedCompletion,
) -> Unit raise TensorParallelGroupTransportError {
  let supervisor = self.published_process()
  let previous = self.current_previous_plan_sequence_value
  let sequence = self.active_sequence_value
  if previous == 0xffffffffffffffffUL || sequence != previous + 1UL {
    rejected(SchedulerPredecessor)
  }
  supervisor.retire_after_scheduler_commit(accepted.0) catch {
    _ => self.process_error(supervisor, rule=Completion)
  }
  self.advance_scheduler_predecessor(sequence)
  self.active_sequence_value = 0UL
  self.completion_accepted = false
}

///|
pub fn TensorParallelGroupTransport::has_exchange_capacity(
  self : TensorParallelGroupTransport,
) -> Bool {
  self.lifecycle() == Ready && self.active_sequence_value == 0UL
}

///|
pub fn TensorParallelGroupTransport::has_submission(
  self : TensorParallelGroupTransport,
  sequence : UInt64,
) -> Bool {
  sequence != 0UL && self.active_sequence_value == sequence
}

///|
pub fn TensorParallelGroupTransport::has_received(
  self : TensorParallelGroupTransport,
  sequence : UInt64,
) -> Bool {
  if sequence == 0UL || self.active_sequence_value != sequence {
    return false
  }
  match self.state {
    Published(supervisor) | Failed(supervisor) | Cleaning(supervisor, ..) =>
      supervisor.lifecycle() == CompletionReady ||
      supervisor.lifecycle() == CompletionAccepted
    _ => false
  }
}

///|
pub fn TensorParallelGroupTransport::abandon_cleaned_submission(
  self : TensorParallelGroupTransport,
  sequence : UInt64,
) -> Unit raise TensorParallelGroupTransportError {
  if self.lifecycle() != Prepared ||
    sequence == 0UL ||
    self.active_sequence_value != sequence {
    rejected(Lifecycle)
  }
  self.advance_scheduler_predecessor(sequence)
  self.active_sequence_value = 0UL
  self.completion_accepted = false
}

///|
/// Advances the replacement predecessor for a scheduler plan that was built
/// behind the physical flight but never submitted to any rank.
pub fn TensorParallelGroupTransport::retire_scheduler_only_failure(
  self : TensorParallelGroupTransport,
  sequence : UInt64,
) -> Unit raise TensorParallelGroupTransportError {
  if self.lifecycle() != Prepared || self.active_sequence_value != 0UL {
    rejected(Lifecycle)
  }
  self.advance_scheduler_predecessor(sequence)
}
