///|
fn append_streaming_template(
  encoder : GroupTemplateEncoder,
  capability : @streaming_ir.Capability,
) -> Unit {
  for
    value in [
      capability.slots(),
      capability.page_count(),
      capability.reserve_pages(),
      capability.transfer_timeout_millis(),
      capability.shape_count(),
    ] {
    encoder.append_int(value)
  }
  for
    value in [
      capability.host_bytes(),
      capability.page_bytes(),
      capability.restore_ns(),
      capability.recompute_ns(),
      capability.workspace_budget(),
    ] {
    encoder.append_int64(value)
  }
  for index in 0..<capability.shape_count() {
    encoder.append_int(capability.shape_at(index).tokens())
    encoder.append_int64(capability.shape_at(index).workspace_bytes())
  }
}

///|
pub fn TensorParallelGroupPlanningInputs::streaming(
  self : TensorParallelGroupPlanningInputs,
) -> @streaming_ir.Capability? {
  self.streaming
}

///|
pub fn TensorParallelGroupTransport::streaming_epoch(
  self : TensorParallelGroupTransport,
) -> UInt64 raise TensorParallelGroupTransportError {
  self.published_process().group_generation_value()
}

///|
pub fn TensorParallelGroupTransport::streaming_now_millis(
  self : TensorParallelGroupTransport,
) -> UInt64 raise TensorParallelGroupTransportError {
  let process = self.published_process()
  process.streaming_now_millis() catch {
    _ => self.process_error(process)
  }
}

///|
pub fn TensorParallelGroupTransport::streaming_exchange_pending(
  self : TensorParallelGroupTransport,
) -> Bool {
  match self.state {
    Published(process) => process.streaming_exchange_pending()
    _ => false
  }
}

///|
pub fn TensorParallelGroupTransport::begin_streaming_exchange(
  self : TensorParallelGroupTransport,
  command : @streaming_ir.Command,
) -> Unit raise TensorParallelGroupTransportError {
  if self.active_sequence_value != 0UL {
    rejected(Lifecycle)
  }
  let process = self.published_process()
  process.begin_streaming_exchange(command) catch {
    _ => self.process_error(process)
  }
}

///|
pub fn TensorParallelGroupTransport::progress_streaming_exchange(
  self : TensorParallelGroupTransport,
) -> @streaming_ir.ReplyOption raise TensorParallelGroupTransportError {
  let process = self.published_process()
  process.progress_streaming_exchange() catch {
    _ => self.process_error(process)
  }
}

///|
pub fn TensorParallelGroupTransport::fail_streaming(
  self : TensorParallelGroupTransport,
) -> Unit {
  match self.state {
    Published(process) => {
      process.fail_streaming()
      if process.lifecycle() == Faulted {
        self.state = Failed(process)
      }
    }
    _ => ()
  }
}
