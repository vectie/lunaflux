///|
const ROOT_TRANSPORT_LEGACY : Int = 1

///|
const ROOT_TRANSPORT_EXCHANGE : Int = 2

///|
const EXCHANGE_IDLE : Int = -1

///|
const EXCHANGE_WRITING : Int = 0

///|
const EXCHANGE_READING_COMPLETION : Int = 1

///|
const EXCHANGE_READING_GRAPH_TELEMETRY : Int = 2

///|
fn RootBoundWorkerProcessSupervisor::clear_exchange(
  self : RootBoundWorkerProcessSupervisor,
) -> Unit {
  self.active_exchange_epoch = 0UL
  self.exchange_phase = EXCHANGE_IDLE
  self.exchange_side = -1
  self.exchange_sequence = 0UL
  self.exchange_plan.clear()
}

///|
fn[T] RootBoundWorkerProcessSupervisor::fail_exchange(
  self : RootBoundWorkerProcessSupervisor,
  error : WorkerProcessError,
) -> T raise RootBoundWorkerProcessError {
  self.process.lifecycle = Failed
  self.clear_exchange()
  self.cause = Some(Process(error))
  self.lifecycle = RootBoundRecoveryRequired
  raise RootBoundFailure(Process(error))
}

///|
fn WorkerProcessSupervisor::begin_exchange_write(
  self : WorkerProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
  timeout_millis : Int,
) -> Int raise WorkerProcessError {
  self.require_ready()
  if self.outstanding_count() != 0 {
    raise Backpressured
  }
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
  let side = if self.state_a == 0 { 0 } else { 1 }
  if side == 0 {
    self.state_a = 1
    self.sequence_a = sequence
  } else {
    self.state_b = 1
    self.sequence_b = sequence
  }
  self.last_submitted = sequence
  let frame = if side == 0 {
    self.plan_a.encode_plan(plan) catch {
      error => {
        self.lifecycle = Failed
        raise Wire(error)
      }
    }
  } else {
    self.plan_b.encode_plan(plan) catch {
      error => {
        self.lifecycle = Failed
        raise Wire(error)
      }
    }
  }
  let transport = if side == 0 { self.tx_a } else { self.tx_b }
  let length = frame.length() catch {
    error => {
      self.lifecycle = Failed
      raise Wire(error)
    }
  }
  let offset = frame.transport_offset() catch {
    error => {
      self.lifecycle = Failed
      raise Wire(error)
    }
  }
  self
  .require_child()
  .begin_write_frame(transport, offset~, byte_count=length, timeout_millis~) catch {
    error => {
      self.lifecycle = Failed
      raise transport_error(error)
    }
  }
  side
}

///|
fn WorkerProcessSupervisor::accept_exchange_completion_bytes(
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
}

///|
/// Commits this ready, idle owner to exchange mode for one exact predecessor.
/// The claim is intended for the thread-confined service constructor.
pub fn RootBoundWorkerProcessSupervisor::claim_exchange(
  self : RootBoundWorkerProcessSupervisor,
  predecessor : UInt64,
) -> Unit raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundReady ||
    self.transport_mode != 0 ||
    self.active_exchange_epoch != 0UL ||
    self.exchange_epoch == 0xffffffffffffffffUL ||
    self.process.outstanding_count() != 0 ||
    self.process.last_submitted != predecessor ||
    self.process.last_retired != predecessor {
    raise RootBoundFailure(Process(Invalid(Lifecycle)))
  }
  self.transport_mode = ROOT_TRANSPORT_EXCHANGE
}

///|
/// Encodes, reserves, and begins one exact plan exchange without performing
/// transport I/O. All mutable exchange state remains in this supervisor.
pub fn RootBoundWorkerProcessSupervisor::begin_exchange(
  self : RootBoundWorkerProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> UInt64 raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundReady {
    raise RootBoundInvalidLifecycle
  }
  if self.transport_mode == ROOT_TRANSPORT_LEGACY ||
    self.active_exchange_epoch != 0UL {
    raise RootBoundFailure(Process(Backpressured))
  }
  if self.exchange_epoch == 0xffffffffffffffffUL {
    raise RootBoundFailure(Process(Invalid(Sequence)))
  }
  let side = self.process.begin_exchange_write(
    plan,
    self.limits.io_timeout_millis(),
  ) catch {
    error => {
      if self.process.lifecycle() != Ready {
        self.fail_exchange(error)
      }
      raise RootBoundFailure(Process(error))
    }
  }
  let sequence = if side == 0 {
    self.process.sequence_a
  } else {
    self.process.sequence_b
  }
  let epoch = self.exchange_epoch + 1UL
  self.exchange_epoch = epoch
  self.active_exchange_epoch = epoch
  self.exchange_phase = EXCHANGE_WRITING
  self.exchange_side = side
  self.exchange_sequence = sequence
  self.exchange_plan.retain(plan) catch {
    error => self.fail_exchange(Protocol(error))
  }
  self.transport_mode = ROOT_TRANSPORT_EXCHANGE
  sequence
}

///|
pub fn RootBoundWorkerProcessSupervisor::has_exchange_capacity(
  self : RootBoundWorkerProcessSupervisor,
) -> Bool {
  self.lifecycle == RootBoundReady &&
  self.transport_mode != ROOT_TRANSPORT_LEGACY &&
  self.exchange_epoch != 0xffffffffffffffffUL &&
  self.active_exchange_epoch == 0UL &&
  self.process.outstanding_count() == 0 &&
  self.process.has_capacity()
}

///|
pub fn RootBoundWorkerProcessSupervisor::active_exchange_sequence(
  self : RootBoundWorkerProcessSupervisor,
) -> UInt64 raise RootBoundWorkerProcessError {
  if self.active_exchange_epoch == 0UL || self.exchange_sequence == 0UL {
    raise RootBoundInvalidLifecycle
  }
  self.exchange_sequence
}

///|
/// Advances at most one native pending-frame syscall. Completion is retained
/// in the existing received side and resolved with `received_for_sequence`.
pub fn RootBoundWorkerProcessSupervisor::progress_exchange(
  self : RootBoundWorkerProcessSupervisor,
) -> RootBoundWorkerExchangeProgress raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundReady ||
    self.transport_mode != ROOT_TRANSPORT_EXCHANGE ||
    self.active_exchange_epoch == 0UL {
    raise RootBoundInvalidLifecycle
  }
  if self.exchange_phase == EXCHANGE_WRITING {
    let complete = self.process.require_child().progress_write_frame() catch {
        error => self.fail_exchange(transport_error(error))
      }
    if !complete {
      return ExchangePending
    }
    self.process
    .require_child()
    .begin_read_frame(
      self.process.rx,
      staging=self.process.rx_staging,
      timeout_millis=self.limits.io_timeout_millis(),
    ) catch {
      error => self.fail_exchange(transport_error(error))
    }
    self.exchange_phase = EXCHANGE_READING_COMPLETION
    return ExchangePending
  }
  if self.exchange_phase == EXCHANGE_READING_COMPLETION {
    let plan = self.exchange_plan.current() catch {
      error => self.fail_exchange(Protocol(error))
    }
    let length = self.process.require_child().progress_read_frame() catch {
        error => self.fail_exchange(transport_error(error))
      }
    guard length is Some(byte_count) else { return ExchangePending }
    self.process.accept_exchange_completion_bytes(
      self.exchange_side,
      plan,
      byte_count,
    ) catch {
      error => self.fail_exchange(error)
    }
    self.process
    .require_child()
    .begin_read_frame(
      self.process.graph_rx,
      staging=self.process.graph_rx_staging,
      timeout_millis=self.limits.io_timeout_millis(),
    ) catch {
      error => self.fail_exchange(transport_error(error))
    }
    self.exchange_phase = EXCHANGE_READING_GRAPH_TELEMETRY
    return ExchangePending
  }
  if self.exchange_phase != EXCHANGE_READING_GRAPH_TELEMETRY {
    self.fail_exchange(Invalid(Completion))
  }
  let length = self.process.require_child().progress_read_frame() catch {
      error => self.fail_exchange(transport_error(error))
    }
  guard length is Some(byte_count) else { return ExchangePending }
  self.process.accept_graph_telemetry_bytes_in_place(
    self.exchange_side,
    byte_count,
  ) catch {
    error => self.fail_exchange(error)
  }
  self.clear_exchange()
  ExchangeCompleted
}
