///|
fn WorkerProcessSupervisor::require_ready(
  self : WorkerProcessSupervisor,
) -> Unit raise WorkerProcessError {
  if self.lifecycle != Ready {
    raise Invalid(Lifecycle)
  }
}

///|
fn WorkerProcessSupervisor::require_child(
  self : WorkerProcessSupervisor,
) -> @process.ChildProcess {
  if !self.child_selected {
    abort("worker process child selection invariant")
  }
  self.child
}

///|
fn transport_error(error : @process.ProcessError) -> WorkerProcessError {
  match error {
    InvalidLimit(_) | InvalidRange(_) => Transport(InvalidConfiguration)
    InvalidState(_) => Transport(InvalidState)
    Timeout(_) => Transport(Timeout)
    ChannelClosed(_) => Transport(ChannelClosed)
    OperationFailed(_) => Transport(OperationFailed)
    SpawnFailed => Transport(SpawnFailed)
  }
}

///|
fn transport_status_error(status : Int) -> WorkerProcessError {
  Transport(
    match status {
      1 => InvalidState
      2 => Timeout
      3 => ChannelClosed
      _ => OperationFailed
    },
  )
}

///|
fn startup_raw_cause(failed : FailedWorkerProcessStartup) -> WorkerProcessError {
  match failed.cause_kind {
    1 => Transport(SpawnFailed)
    2 => transport_status_error(failed.cause_status)
    3 => Invalid(Submission)
    _ => Invalid(Lifecycle)
  }
}

///|
fn prepare_spawned(
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  prepared : PreparedWorkerProcessStartup,
  child : @process.ChildProcess,
) -> WorkerProcessPreparation {
  let status = perform_startup_handshake_status(
    child,
    source,
    limits.startup_io_timeout_millis,
    prepared,
  )
  if status == 0 {
    prepared.supervisor.child_selected = true
    prepared.failed.child_selected = false
    return ReadyProcess(prepared.supervisor)
  }
  if status >= 300 {
    prepared.failed.cause_kind = 3
    prepared.failed.cause_status = 0
  } else {
    prepared.failed.cause_kind = 2
    prepared.failed.cause_status = if status >= 200 {
      status - 201
    } else {
      status - 100
    }
  }
  let cleanup_status = child.startup_close_status()
  prepared.failed.cleanup_status = cleanup_status
  prepared.supervisor.child_selected = false
  prepared.failed.child_selected = !child.is_closed()
  prepared.failed.closed = child.is_closed()
  CleanupRequired(prepared.failed)
}

///|
fn prepare_spawn_failed(
  prepared : PreparedWorkerProcessStartup,
) -> WorkerProcessPreparation {
  prepared.failed.cause_kind = 1
  prepared.failed.cause_status = 0
  prepared.failed.cleanup_status = prepared.child.cleanup_status()
  prepared.failed.closed = prepared.child.child_owner().is_closed()
  prepared.supervisor.child_selected = false
  prepared.failed.child_selected = !prepared.failed.closed
  CleanupRequired(prepared.failed)
}

///|
fn closed_startup_error(
  failed : FailedWorkerProcessStartup,
) -> WorkerProcessError {
  startup_raw_cause(failed)
}

///|
fn prepare_preflighted(
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  prepared : PreparedWorkerProcessStartup,
  roots : @approved_fs.WorkerApprovedRoots,
) -> WorkerProcessPreparation {
  let spawn_status = prepared.child.spawn_with_approved_roots_status(roots)
  if spawn_status != 0 {
    return prepare_spawn_failed(prepared)
  }
  prepare_spawned(source, limits, prepared, prepared.child.child_owner())
}

///|
/// Compatibility-only spawn for deterministic echo fixtures. Production
/// workers use `prepare_with_approved_roots`, which retains exact root leases
/// across every replacement.
pub fn prepare(
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
) -> WorkerProcessPreparation raise WorkerProcessError {
  let prepared = preflight_prepare(executable, startup, source, limits)
  let child = prepared.child.spawn() catch {
    error => raise transport_error(error)
  }
  let result = prepare_spawned(source, limits, prepared, child)
  match result {
    CleanupRequired(failed) if failed.is_closed() && failed.cleanup_status == 0 =>
      raise closed_startup_error(failed)
    _ => result
  }
}

///|
fn WorkerProcessSupervisor::fail_process(
  self : WorkerProcessSupervisor,
  error : @process.ProcessError,
) -> Unit raise WorkerProcessError {
  self.lifecycle = Failed
  raise transport_error(error)
}

///|
pub fn WorkerProcessSupervisor::submit(
  self : WorkerProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> WorkerSubmission raise WorkerProcessError {
  self.require_ready()
  let sequence = plan.sequence_wire() catch { error => raise Protocol(error) }
  let generation = plan.model_plan_generation_wire() catch {
    error => raise Protocol(error)
  }
  if generation != self.startup.model_generation_value() {
    raise Invalid(Submission)
  }
  if self.last_submitted == 0xffffffffffffffffUL ||
    sequence != self.last_submitted + 1UL {
    raise Invalid(Sequence)
  }
  let side = if self.state_a == 0 {
    0
  } else if self.state_b == 0 {
    1
  } else {
    raise Backpressured
  }
  let frame = if side == 0 {
    self.plan_a.encode_plan(plan) catch {
      error => raise Wire(error)
    }
  } else {
    self.plan_b.encode_plan(plan) catch {
      error => raise Wire(error)
    }
  }
  let transport = if side == 0 { self.tx_a } else { self.tx_b }
  let length = frame.length() catch { error => raise Wire(error) }
  frame.copy_to(transport, destination_offset=0) catch {
    error => raise Wire(error)
  }
  self.require_child().write_frame(transport, offset=0, byte_count=length) catch {
    error => self.fail_process(error)
  }
  if side == 0 {
    self.state_a = 1
    self.sequence_a = sequence
  } else {
    self.state_b = 1
    self.sequence_b = sequence
  }
  self.last_submitted = sequence
  WorkerSubmission(self, side, sequence)
}

///|
fn WorkerSubmission::authenticate(
  self : WorkerSubmission,
  owner : WorkerProcessSupervisor,
) -> Unit raise WorkerProcessError {
  if !physical_equal(self.0, owner) {
    raise OwnerMismatch
  }
  let state = if self.1 == 0 { owner.state_a } else { owner.state_b }
  let sequence = if self.1 == 0 { owner.sequence_a } else { owner.sequence_b }
  if state != 1 || sequence != self.2 {
    raise Invalid(Submission)
  }
}

///|
/// Reconstructs the exact live submission capability for a retained sequence.
pub fn WorkerProcessSupervisor::submission_for_sequence(
  self : WorkerProcessSupervisor,
  sequence : UInt64,
) -> WorkerSubmission raise WorkerProcessError {
  if self.state_a == 1 && self.sequence_a == sequence {
    WorkerSubmission(self, 0, sequence)
  } else if self.state_b == 1 && self.sequence_b == sequence {
    WorkerSubmission(self, 1, sequence)
  } else {
    raise Invalid(Submission)
  }
}

///|
fn WorkerProcessSupervisor::oldest_submitted_side(
  self : WorkerProcessSupervisor,
) -> Int raise WorkerProcessError {
  if self.state_a == 2 || self.state_b == 2 {
    raise Backpressured
  }
  if self.state_a == 1 &&
    (self.state_b != 1 || self.sequence_a < self.sequence_b) {
    0
  } else if self.state_b == 1 {
    1
  } else {
    raise Invalid(Completion)
  }
}

///|
/// Receives and retains the oldest completion until explicit retirement.
pub fn WorkerProcessSupervisor::receive(
  self : WorkerProcessSupervisor,
  submission : WorkerSubmission,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> ReceivedWorkerCompletion raise WorkerProcessError {
  self.require_ready()
  submission.authenticate(self)
  let side = self.oldest_submitted_side()
  if side != submission.1 {
    raise Invalid(Sequence)
  }
  let plan_sequence = plan.sequence_wire() catch {
    error => raise Protocol(error)
  }
  if plan_sequence != submission.2 {
    raise Invalid(Submission)
  }
  let length = self.require_child().read_frame(self.rx) catch {
      error => {
        self.lifecycle = Failed
        raise transport_error(error)
      }
    }
  self.accept_completion_bytes(side, plan, length)
}

///|
fn WorkerProcessSupervisor::accept_completion_bytes(
  self : WorkerProcessSupervisor,
  side : Int,
  plan : @worker_protocol.SubmittedSchedulePlan,
  length : Int,
) -> ReceivedWorkerCompletion raise WorkerProcessError {
  self.accept_completion_bytes_in_place(side, plan, length)
  let sequence = if side == 0 { self.sequence_a } else { self.sequence_b }
  ReceivedWorkerCompletion(self, side, sequence)
}

///|
fn WorkerProcessSupervisor::accept_completion_bytes_in_place(
  self : WorkerProcessSupervisor,
  side : Int,
  plan : @worker_protocol.SubmittedSchedulePlan,
  length : Int,
) -> Unit raise WorkerProcessError {
  let frame = if side == 0 {
    self.completion_a.load(self.rx, length) catch {
      error => {
        self.lifecycle = Failed
        raise Wire(error)
      }
    }
  } else {
    self.completion_b.load(self.rx, length) catch {
      error => {
        self.lifecycle = Failed
        raise Wire(error)
      }
    }
  }
  frame.validate_for(plan) catch {
    error => {
      self.lifecycle = Failed
      raise Wire(error)
    }
  }
  if side == 0 {
    self.state_a = 2
  } else {
    self.state_b = 2
  }
}

///|
/// Returns the pinned completion frame without changing supervisor state.
pub fn ReceivedWorkerCompletion::frame(
  self : ReceivedWorkerCompletion,
  owner : WorkerProcessSupervisor,
) -> @worker_wire.ValidatedCompletionFrame raise WorkerProcessError {
  if !physical_equal(self.0, owner) {
    raise OwnerMismatch
  }
  let state = if self.1 == 0 { owner.state_a } else { owner.state_b }
  let sequence = if self.1 == 0 { owner.sequence_a } else { owner.sequence_b }
  if state != 2 || sequence != self.2 {
    raise Invalid(Completion)
  }
  if self.1 == 0 {
    owner.completion_a.current() catch {
      error => raise Wire(error)
    }
  } else {
    owner.completion_b.current() catch {
      error => raise Wire(error)
    }
  }
}

///|
/// Reconstructs the exact pinned completion capability for a retained sequence.
pub fn WorkerProcessSupervisor::received_for_sequence(
  self : WorkerProcessSupervisor,
  sequence : UInt64,
) -> ReceivedWorkerCompletion raise WorkerProcessError {
  if self.state_a == 2 && self.sequence_a == sequence {
    ReceivedWorkerCompletion(self, 0, sequence)
  } else if self.state_b == 2 && self.sequence_b == sequence {
    ReceivedWorkerCompletion(self, 1, sequence)
  } else {
    raise Invalid(Completion)
  }
}

///|
/// Nonmutating retirement proof for service commit ordering.
pub fn WorkerProcessSupervisor::validate_retirement(
  self : WorkerProcessSupervisor,
  received : ReceivedWorkerCompletion,
) -> Unit raise WorkerProcessError {
  ignore(received.frame(self))
  if self.last_retired == 0xffffffffffffffffUL ||
    received.2 != self.last_retired + 1UL {
    raise Invalid(Sequence)
  }
}

///|
/// Frees one retained side only after scheduler acceptance has succeeded.
pub fn WorkerProcessSupervisor::retire_received(
  self : WorkerProcessSupervisor,
  received : ReceivedWorkerCompletion,
) -> Unit raise WorkerProcessError {
  self.validate_retirement(received)
  if received.1 == 0 {
    self.state_a = 0
  } else {
    self.state_b = 0
  }
  self.last_retired = received.2
}

///|
/// Closes and reaps the child. Outstanding capabilities remain exact recovery
/// obligations until explicitly retired or abandoned in sequence order.
pub fn WorkerProcessSupervisor::close(
  self : WorkerProcessSupervisor,
) -> Unit raise WorkerProcessError {
  if self.lifecycle == Closed {
    return
  }
  self.lifecycle = ClosingFailed
  self.require_child().close() catch {
    error => raise transport_error(error)
  }
  self.lifecycle = Closed
}
