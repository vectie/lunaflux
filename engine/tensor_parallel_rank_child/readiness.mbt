///|
fn begin_worker_startup(
  configured : TensorParallelRankChildConfigured,
  worker : @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner,
) -> TensorParallelRankChildStarting raise TensorParallelRankChildError {
  if worker.lifecycle() != WorkerStarting {
    raise Invalid(WorkerMismatch)
  }
  let startup = configured.contract.startup()
  let plan_buffer = @worker_wire.PlanFrameBuffer::new(
    startup.worker_limits(),
    startup.inference_limits(),
  ) catch {
    error => raise Wire(error)
  }
  let plan_capacity = @worker_wire.plan_frame_byte_capacity(
    startup.worker_limits(),
  ) catch {
    error => raise Wire(error)
  }
  let completion_capacity = @worker_wire.completion_frame_byte_capacity(
    startup.worker_limits(),
  ) catch {
    error => raise Wire(error)
  }
  {
    configured,
    worker,
    plan_buffer,
    plan_storage: FixedArray::make(plan_capacity, b'\x00'),
    completion_storage: FixedArray::make(completion_capacity, b'\x00'),
    ready_started: false,
  }
}

///|
pub fn TensorParallelRankChildStarting::progress_ready(
  self : TensorParallelRankChildStarting,
) -> TensorParallelRankChildReadyProgress raise TensorParallelRankChildError {
  if !self.ready_started {
    let ready = self.worker.poll_startup() catch {
      error => raise Worker(error)
    }
    if !ready {
      return CollectivePending
    }
    let startup = self.worker.readiness_contract() catch {
      error => raise Worker(error)
    }
    if startup != self.configured.contract.startup() {
      raise Invalid(WorkerMismatch)
    }
    self.configured.receiver.control.begin_ready(
      self.configured.inbound,
      self.configured.contract.binding(),
      previous_sequence_value=startup.predecessor_value(),
    ) catch {
      error => raise Control(error)
    }
    self.ready_started = true
    return ReadySending
  }
  let sent = self.configured.receiver.control.progress_send() catch {
    error => raise Control(error)
  }
  if sent.is_complete() {
    ReadyPublished
  } else {
    ReadySending
  }
}

///|
pub fn TensorParallelRankChildStarting::ready(
  self : TensorParallelRankChildStarting,
) -> TensorParallelRankChildReady raise TensorParallelRankChildError {
  if !self.ready_started ||
    self.configured.receiver.control.lifecycle() != ReadyGroup {
    raise Invalid(InvalidLifecycle)
  }
  self.configured.receiver.control.begin_receive() catch {
    error => raise Control(error)
  }
  {
    control_owner: self.configured.receiver.control,
    startup: self.configured.contract.startup(),
    binding: self.configured.contract.binding(),
    worker: self.worker,
    plan_buffer: self.plan_buffer,
    plan_storage: self.plan_storage,
    completion_storage: self.completion_storage,
    graph_telemetry: @worker_wire.WorkerGraphTelemetrySlot::new(),
    execution: ChildIdle,
    receive_active: true,
    send_active: false,
    close_response: false,
  }
}
