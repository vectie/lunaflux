///|
fn TensorParallelRankChildReady::begin_next_receive(
  self : TensorParallelRankChildReady,
) -> Unit raise TensorParallelRankChildError {
  self.control_owner.begin_receive() catch {
    error => raise Control(error)
  }
  self.receive_active = true
}

///|
fn TensorParallelRankChildReady::latched_failure_class(
  self : TensorParallelRankChildReady,
) -> @tensor_parallel_device_worker.TensorParallelDeviceWorkerFailureClass {
  match self.worker.failure_class() {
    Some(failure) => failure
    None => RankExecutionFailure
  }
}

///|
fn TensorParallelRankChildReady::advance_execution(
  self : TensorParallelRankChildReady,
) -> Bool {
  match self.execution {
    ChildExecuting(frame, staged) => {
      let progress = try self.worker.progress_execution(staged, frame) catch {
        _ => {
          self.execution = ChildExecutionFailed(self.latched_failure_class())
          return true
        }
      } noraise {
        value => value
      }
      if progress.is_pending() {
        false
      } else {
        let executed = try progress.executed() catch {
          _ => {
            self.execution = ChildExecutionFailed(self.latched_failure_class())
            return true
          }
        } noraise {
          value => value
        }
        self.execution = ChildExecuted(frame, executed)
        true
      }
    }
    _ => false
  }
}

///|
fn TensorParallelRankChildReady::accept_submit(
  self : TensorParallelRankChildReady,
  inbound : @rank_child_control.RankChildInbound,
) -> Unit raise TensorParallelRankChildError {
  match self.execution {
    ChildIdle => ()
    _ => raise Invalid(InvalidLifecycle)
  }
  if self.worker.lifecycle() != WorkerReady {
    raise Invalid(InvalidLifecycle)
  }
  let length = inbound.payload_length() catch { error => raise Control(error) }
  if length <= 0 || length > self.plan_storage.length() {
    raise Invalid(PlanFrame)
  }
  inbound.copy_payload_to(self.plan_storage, destination_offset=0) catch {
    error => raise Control(error)
  }
  let frame = self.plan_buffer.load(self.plan_storage, length) catch {
    error => raise Wire(error)
  }
  let startup = self.startup
  let frame_sequence = frame.sequence_value() catch {
    error => raise Wire(error)
  }
  let inbound_sequence = inbound.sequence_value() catch {
    error => raise Control(error)
  }
  let frame_generation = frame.model_generation_value() catch {
    error => raise Wire(error)
  }
  if frame_sequence != inbound_sequence ||
    frame_generation != startup.model_generation_value() {
    raise Invalid(PlanFrame)
  }
  let staged = try self.worker.stage_frame(frame) catch {
    _ => {
      self.control_owner.accept_submit(inbound) catch {
        error => raise Control(error)
      }
      self.execution = ChildExecutionFailed(self.latched_failure_class())
      self.begin_next_receive()
      return
    }
  } noraise {
    value => value
  }
  self.control_owner.accept_submit(inbound) catch {
    error => raise Control(error)
  }
  self.execution = ChildExecuting(frame, staged)
  self.begin_next_receive()
}

///|
fn TensorParallelRankChildReady::begin_failure_response(
  self : TensorParallelRankChildReady,
  inbound : @rank_child_control.RankChildInbound,
  failure : @tensor_parallel_device_worker.TensorParallelDeviceWorkerFailureClass,
) -> Unit raise TensorParallelRankChildError {
  match failure {
    CollectiveFailure =>
      self.control_owner.begin_collective_failed(inbound) catch {
        error => raise Control(error)
      }
    RankExecutionFailure =>
      self.control_owner.begin_execution_failed(inbound) catch {
        error => raise Control(error)
      }
  }
  self.send_active = true
}

///|
fn TensorParallelRankChildReady::retain_graph_telemetry(
  self : TensorParallelRankChildReady,
  sequence : UInt64,
) -> Unit raise TensorParallelRankChildError {
  if self.graph_telemetry.is_available() {
    raise Invalid(InvalidLifecycle)
  }
  let report = self.worker.graph_runtime_telemetry(sequence) catch {
    error => raise Worker(error)
  }
  self.graph_telemetry.store(report)
}

///|
fn TensorParallelRankChildReady::finish_leader(
  self : TensorParallelRankChildReady,
  inbound : @rank_child_control.RankChildInbound,
  frame : @worker_wire.ValidatedPlanFrame,
  executed : @tensor_parallel_device_worker.ExecutedTensorParallelDeviceExecution,
) -> Unit raise TensorParallelRankChildError {
  let completion = try self.worker.finish_leader_frame(executed, frame) catch {
    _ => {
      let failure = self.latched_failure_class()
      self.execution = ChildExecutionFailed(failure)
      self.begin_failure_response(inbound, failure)
      return
    }
  } noraise {
    value => value
  }
  let completion_sequence = completion.sequence_value() catch {
    error => raise Wire(error)
  }
  let inbound_sequence = inbound.sequence_value() catch {
    error => raise Control(error)
  }
  let completion_generation = completion.model_generation_value() catch {
    error => raise Wire(error)
  }
  if completion_sequence != inbound_sequence ||
    completion_generation != self.startup.model_generation_value() {
    raise Invalid(Completion)
  }
  self.retain_graph_telemetry(completion_sequence)
  let length = completion.length() catch { error => raise Wire(error) }
  if length <= 0 || length > self.completion_storage.length() {
    raise Invalid(Completion)
  }
  completion.copy_to(self.completion_storage, destination_offset=0) catch {
    error => raise Wire(error)
  }
  self.control_owner.begin_leader_complete(
    inbound,
    self.completion_storage,
    payload_length=length,
  ) catch {
    error => raise Control(error)
  }
  self.execution = ChildIdle
  self.send_active = true
}

///|
fn TensorParallelRankChildReady::finish_follower(
  self : TensorParallelRankChildReady,
  inbound : @rank_child_control.RankChildInbound,
  frame : @worker_wire.ValidatedPlanFrame,
  executed : @tensor_parallel_device_worker.ExecutedTensorParallelDeviceExecution,
) -> Unit raise TensorParallelRankChildError {
  let ack = try self.worker.finish_follower(executed, frame) catch {
    _ => {
      let failure = self.latched_failure_class()
      self.execution = ChildExecutionFailed(failure)
      self.begin_failure_response(inbound, failure)
      return
    }
  } noraise {
    value => value
  }
  let binding = self.binding
  let inbound_sequence = inbound.sequence_value() catch {
    error => raise Control(error)
  }
  if ack.group_generation_value() != binding.group_generation_value() ||
    ack.model_generation_value() != binding.model_generation_value() ||
    ack.plan_sequence_value() != inbound_sequence ||
    ack.rank() != binding.rank() ||
    ack.world_size() != binding.world_size() {
    raise Invalid(Completion)
  }
  self.retain_graph_telemetry(ack.plan_sequence_value())
  self.control_owner.begin_follower_complete(inbound) catch {
    error => raise Control(error)
  }
  self.execution = ChildIdle
  self.send_active = true
}

///|
fn TensorParallelRankChildReady::accept_poll(
  self : TensorParallelRankChildReady,
  inbound : @rank_child_control.RankChildInbound,
) -> Unit raise TensorParallelRankChildError {
  match self.execution {
    ChildExecuting(_, _) => {
      self.control_owner.begin_pending(inbound) catch {
        error => raise Control(error)
      }
      self.send_active = true
    }
    ChildExecuted(frame, executed) =>
      if self.binding.rank() == 0 {
        self.finish_leader(inbound, frame, executed)
      } else {
        self.finish_follower(inbound, frame, executed)
      }
    ChildExecutionFailed(failure) =>
      self.begin_failure_response(inbound, failure)
    ChildIdle => raise Invalid(InvalidLifecycle)
  }
}

///|
fn TensorParallelRankChildReady::abort_active_worker(
  self : TensorParallelRankChildReady,
) -> Unit raise TensorParallelRankChildError {
  match self.execution {
    ChildIdle => raise Invalid(InvalidLifecycle)
    _ => ()
  }
  self.worker.abort_plan() catch {
    error => raise Worker(error)
  }
  self.execution = ChildIdle
}

///|
fn TensorParallelRankChildReady::drain_worker(
  self : TensorParallelRankChildReady,
) -> Unit raise TensorParallelRankChildError {
  match self.execution {
    ChildExecuting(_, _) | ChildExecuted(_, _) =>
      self.worker.abort_plan() catch {
        error => raise Worker(error)
      }
    ChildExecutionFailed(_) =>
      self.worker.begin_drain() catch {
        error => raise Worker(error)
      }
    ChildIdle =>
      match self.worker.lifecycle() {
        WorkerReady | WorkerFaulted =>
          self.worker.begin_drain() catch {
            error => raise Worker(error)
          }
        WorkerDraining => ()
        _ => raise Invalid(InvalidLifecycle)
      }
  }
  self.execution = ChildIdle
}

///|
fn TensorParallelRankChildReady::accept_inbound(
  self : TensorParallelRankChildReady,
  inbound : @rank_child_control.RankChildInbound,
) -> Unit raise TensorParallelRankChildError {
  let kind = inbound.kind() catch { error => raise Control(error) }
  match kind {
    Submit => self.accept_submit(inbound)
    PollComplete => self.accept_poll(inbound)
    Abort => {
      self.abort_active_worker()
      self.control_owner.begin_aborted(inbound) catch {
        error => raise Control(error)
      }
      self.send_active = true
    }
    Drain => {
      self.drain_worker()
      self.control_owner.begin_drained(inbound) catch {
        error => raise Control(error)
      }
      self.send_active = true
    }
    Close => {
      self.worker.close() catch {
        error => raise Worker(error)
      }
      self.control_owner.begin_closed(inbound) catch {
        error => raise Control(error)
      }
      self.send_active = true
      self.close_response = true
    }
    _ => raise Invalid(ControlBoundary)
  }
}

///|
/// Advances at most one response write, or one rank execution transition plus
/// one command read. A terminal result is returned only after the actual device
/// worker is closed and the canonical `Closed` response is fully published.
pub fn TensorParallelRankChildReady::progress_control(
  self : TensorParallelRankChildReady,
) -> TensorParallelRankChildControlProgress raise TensorParallelRankChildError {
  if self.send_active {
    let sent = self.control_owner.progress_send() catch {
      error => raise Control(error)
    }
    if !sent.is_complete() {
      return ControlPending
    }
    self.send_active = false
    if self.close_response {
      return ControlClosed
    }
    if self.graph_telemetry.is_available() {
      let report = self.graph_telemetry.current() catch {
        error => raise Wire(error)
      }
      self.control_owner.begin_graph_telemetry_sidecar(report) catch {
        error => raise Control(error)
      }
      self.graph_telemetry.clear()
      self.send_active = true
      return ControlAdvanced
    }
    self.begin_next_receive()
    return ControlAdvanced
  }
  let execution_advanced = self.advance_execution()
  if !self.receive_active {
    raise Invalid(InvalidLifecycle)
  }
  let received = self.control_owner.progress_receive() catch {
    error => raise Control(error)
  }
  if received.is_clean_eof() {
    raise Invalid(ControlBoundary)
  }
  if !received.has_inbound() {
    return if execution_advanced { ControlAdvanced } else { ControlPending }
  }
  self.receive_active = false
  let inbound = self.control_owner.inbound() catch {
    error => raise Control(error)
  }
  self.accept_inbound(inbound)
  ControlAdvanced
}

///|
/// Deterministically releases the prepared worker after a local control or
/// transport failure. The caller must terminate the inherited child channel.
pub fn TensorParallelRankChildReady::close_worker_after_failure(
  self : TensorParallelRankChildReady,
) -> Unit raise TensorParallelRankChildError {
  self.worker.close() catch {
    error => raise Worker(error)
  }
}

///|
/// Deterministically releases a worker whose collective startup or Ready
/// publication failed.
pub fn TensorParallelRankChildStarting::close_worker_after_failure(
  self : TensorParallelRankChildStarting,
) -> Unit raise TensorParallelRankChildError {
  self.worker.close() catch {
    error => raise Worker(error)
  }
}
