///|
pub struct RemoteRankChildStarting {
  priv control : @rank_child_control.RankChildControl
  priv configure : @rank_child_control.RankChildInbound
  priv worker : @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner
  priv binding : @rank_group_wire.RankGroupWireBinding
  priv startup : @worker_wire.WorkerStartupContract
  priv mut ready_started : Bool
  priv mut consumed : Bool
}

///|
/// Joins an independently admitted network worker to the existing command
/// executor. Incoming startup bytes must equal the worker's retained contract.
pub fn start_remote_rank_child(
  control : @rank_child_control.RankChildControl,
  worker : @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner,
) -> RemoteRankChildStarting raise {
  let (remote, envelope) = worker.remote_contract()
  let configure = control.inbound()
  if configure.kind() != Configure || worker.lifecycle() != WorkerStarting {
    raise Invalid(InvalidLifecycle)
  }
  let length = configure.payload_length()
  let bytes = FixedArray::make(
    @remote_startup.remote_startup_frame_bytes(),
    b'\x00',
  )
  if length != bytes.length() {
    raise Invalid(ConfigureFrame)
  }
  configure.copy_payload_to(bytes, destination_offset=0)
  remote.verify_configure(bytes, length)
  let binding = @rank_group_wire.RankGroupWireBinding::new(
    identity=envelope.identity(),
    model_generation_value=envelope.model_generation_value(),
    group_generation_value=envelope.group_generation_value(),
    rank=envelope.rank(),
    world_size=envelope.world_size(),
    group_digest=envelope.group_digest().as_sha256(),
    topology_digest=envelope.topology_digest().as_sha256(),
    worker_digest=envelope.worker_digest().as_sha256(),
    device_plan_digest=envelope.device_plan_digest().as_sha256(),
    execution_plan_digest=envelope.execution_plan_digest_sha256(),
    collective_digest=envelope.collective_digest().as_hex(),
    kv_digest=envelope.kv_digest().as_sha256(),
  )
  // Full binding equality is checked by begin_ready against the transcript.
  {
    control,
    configure,
    worker,
    binding,
    startup: remote.worker(),
    ready_started: false,
    consumed: false,
  }
}

///|
pub fn RemoteRankChildStarting::progress(
  self : RemoteRankChildStarting,
) -> TensorParallelRankChildReady? raise {
  if self.consumed {
    raise Invalid(InvalidLifecycle)
  }
  self.worker.poll_remote_lease()
  if !self.ready_started {
    if !self.worker.poll_startup() {
      return None
    }
    if self.worker.readiness_contract() != self.startup {
      raise Invalid(WorkerMismatch)
    }
    self.control.begin_ready(
      self.configure,
      self.binding,
      previous_sequence_value=self.startup.predecessor_value(),
    )
    self.ready_started = true
    return None
  }
  if !self.control.progress_send().is_complete() {
    return None
  }
  let plan_buffer = @worker_wire.PlanFrameBuffer::new(
    self.startup.worker_limits(),
    self.startup.inference_limits(),
  )
  let plan_capacity = @worker_wire.plan_frame_byte_capacity(
    self.startup.worker_limits(),
  )
  let completion_capacity = @worker_wire.completion_frame_byte_capacity(
    self.startup.worker_limits(),
  )
  self.control.begin_receive()
  self.consumed = true
  Some({
    control_owner: self.control,
    startup: self.startup,
    binding: self.binding,
    worker: self.worker,
    plan_buffer,
    plan_storage: FixedArray::make(plan_capacity, b'\x00'),
    completion_storage: FixedArray::make(completion_capacity, b'\x00'),
    graph_telemetry: @worker_wire.WorkerGraphTelemetrySlot::new(),
    execution: ChildIdle,
    receive_active: true,
    send_active: false,
    close_response: false,
  })
}
