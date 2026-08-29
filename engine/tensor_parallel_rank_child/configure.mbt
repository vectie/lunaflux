///|
fn exact_child_binding(
  left : @rank_group_wire.RankGroupWireBinding,
  right : @rank_group_wire.RankGroupWireBinding,
) -> Bool {
  left.identity() == right.identity() &&
  left.model_generation_value() == right.model_generation_value() &&
  left.group_generation_value() == right.group_generation_value() &&
  left.rank() == right.rank() &&
  left.world_size() == right.world_size() &&
  left.group_digest() == right.group_digest() &&
  left.topology_digest() == right.topology_digest() &&
  left.worker_digest() == right.worker_digest() &&
  left.device_plan_digest() == right.device_plan_digest() &&
  left.execution_plan_digest() == right.execution_plan_digest() &&
  left.collective_digest() == right.collective_digest() &&
  left.kv_digest() == right.kv_digest()
}

///|
pub fn open_configure_receiver(
  wire_limits : @rank_group_wire.RankGroupWireLimits,
  io_timeout_millis~ : Int,
  shutdown_timeout_millis~ : Int,
) -> TensorParallelRankChildConfigureReceiver raise TensorParallelRankChildError {
  let control = @rank_child_control.open_rank_child_control(
    wire_limits,
    io_timeout_millis~,
    shutdown_timeout_millis~,
  ) catch {
    error => raise Control(error)
  }
  {
    control,
    storage: FixedArray::make(
      @tensor_parallel_rank_configure.tensor_parallel_rank_configure_capacity(),
      b'\x00',
    ),
    receive_started: false,
    configured: false,
  }
}

///|
pub fn TensorParallelRankChildConfigureReceiver::begin(
  self : TensorParallelRankChildConfigureReceiver,
) -> Unit raise TensorParallelRankChildError {
  if self.receive_started || self.configured {
    raise Invalid(InvalidLifecycle)
  }
  self.control.begin_receive() catch {
    error => raise Control(error)
  }
  self.receive_started = true
}

///|
pub fn TensorParallelRankChildConfigureReceiver::progress(
  self : TensorParallelRankChildConfigureReceiver,
) -> TensorParallelRankChildConfigureProgress raise TensorParallelRankChildError {
  if !self.receive_started || self.configured {
    raise Invalid(InvalidLifecycle)
  }
  let progress = self.control.progress_receive() catch {
    error => raise Control(error)
  }
  if progress.is_clean_eof() {
    raise Invalid(ControlBoundary)
  }
  if !progress.has_inbound() {
    return ConfigurePending
  }
  let inbound = self.control.inbound() catch { error => raise Control(error) }
  let kind = inbound.kind() catch { error => raise Control(error) }
  if kind != Configure {
    raise Invalid(ConfigureFrame)
  }
  let length = inbound.payload_length() catch { error => raise Control(error) }
  if length <= 0 || length > self.storage.length() {
    raise Invalid(ConfigureFrame)
  }
  inbound.copy_payload_to(self.storage, destination_offset=0) catch {
    error => raise Control(error)
  }
  let inbound_binding = inbound.binding() catch {
    error => raise Control(error)
  }
  let configure_buffer = @tensor_parallel_rank_configure.TensorParallelRankConfigureFrameBuffer::new(
    inbound_binding,
  )
  let frame = configure_buffer.load(self.storage, length) catch {
    error => raise Configure(error)
  }
  let contract = frame.contract() catch { error => raise Configure(error) }
  if !exact_child_binding(contract.binding(), inbound_binding) {
    raise Invalid(BindingMismatch)
  }
  self.configured = true
  ConfigureReady({ receiver: self, inbound, contract })
}
