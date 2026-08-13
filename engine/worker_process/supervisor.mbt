///|
fn WorkerProcessSupervisor::require_ready(
  self : WorkerProcessSupervisor,
) -> Unit raise WorkerProcessError {
  if self.lifecycle != Ready {
    raise Invalid(Lifecycle)
  }
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
pub fn start(
  executable : Bytes,
  worker_limits : @worker_protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
  limits : WorkerProcessLimits,
  has_predecessor~ : Bool,
  predecessor_value~ : UInt64,
) -> WorkerProcessSupervisor raise WorkerProcessError {
  let plan_a = @worker_wire.PlanFrameBuffer::new(
    worker_limits, inference_limits,
  ) catch {
    error => raise Wire(error)
  }
  let plan_b = @worker_wire.PlanFrameBuffer::new(
    worker_limits, inference_limits,
  ) catch {
    error => raise Wire(error)
  }
  let completion_a = @worker_wire.CompletionFrameBuffer::new(
    worker_limits, inference_limits,
  ) catch {
    error => raise Wire(error)
  }
  let completion_b = @worker_wire.CompletionFrameBuffer::new(
    worker_limits, inference_limits,
  ) catch {
    error => raise Wire(error)
  }
  let required = plan_a.capacity().max(completion_a.capacity())
  if required > limits.max_frame_bytes ||
    (has_predecessor && predecessor_value == 0xffffffffffffffffUL) ||
    (!has_predecessor && predecessor_value != 0UL) {
    raise Invalid(Capacity)
  }
  let process_limits = @process.ProcessLimits::new(
    max_frame_bytes=limits.max_frame_bytes,
    io_timeout_millis=limits.io_timeout_millis,
    shutdown_timeout_millis=limits.shutdown_timeout_millis,
  ) catch {
    error => raise transport_error(error)
  }
  let child = @process.spawn(executable, process_limits) catch {
    error => raise transport_error(error)
  }
  {
    child,
    plan_a,
    plan_b,
    completion_a,
    completion_b,
    tx_a: FixedArray::make(plan_a.capacity(), b'\x00'),
    tx_b: FixedArray::make(plan_b.capacity(), b'\x00'),
    rx: FixedArray::make(completion_a.capacity(), b'\x00'),
    state_a: 0,
    state_b: 0,
    sequence_a: 0UL,
    sequence_b: 0UL,
    last_submitted: if has_predecessor {
      predecessor_value
    } else {
      0UL
    },
    last_retired: if has_predecessor {
      predecessor_value
    } else {
      0UL
    },
    lifecycle: Ready,
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
  self.child.write_frame(transport, offset=0, byte_count=length) catch {
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
  let length = self.child.read_frame(self.rx) catch {
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
  let sequence = if side == 0 { self.sequence_a } else { self.sequence_b }
  ReceivedWorkerCompletion(self, side, sequence)
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
/// Frees one retained side only after scheduler acceptance has succeeded.
pub fn WorkerProcessSupervisor::retire_received(
  self : WorkerProcessSupervisor,
  received : ReceivedWorkerCompletion,
) -> Unit raise WorkerProcessError {
  self.require_ready()
  ignore(received.frame(self))
  if received.2 != self.last_retired + 1UL {
    raise Invalid(Sequence)
  }
  if received.1 == 0 {
    self.state_a = 0
  } else {
    self.state_b = 0
  }
  self.last_retired = received.2
}

///|
/// Closes the child and invalidates every outstanding capability.
pub fn WorkerProcessSupervisor::close(
  self : WorkerProcessSupervisor,
) -> Unit raise WorkerProcessError {
  if self.lifecycle == Closed {
    return
  }
  self.lifecycle = ClosingFailed
  self.child.close() catch {
    error => raise transport_error(error)
  }
  self.lifecycle = Closed
}
