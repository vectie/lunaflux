///|
fn TensorParallelRankChildReady::accept_streaming(
  self : TensorParallelRankChildReady,
  inbound : @rank_child_control.RankChildInbound,
) -> Unit raise TensorParallelRankChildError {
  if !(self.execution is ChildIdle) || self.worker.lifecycle() != WorkerReady {
    raise Invalid(InvalidLifecycle)
  }
  let length = inbound.payload_length() catch { error => raise Control(error) }
  if length != 64 {
    raise Invalid(ControlBoundary)
  }
  inbound.copy_payload_to(self.plan_storage, destination_offset=0) catch {
    error => raise Control(error)
  }
  let request = @streaming_ir.decode(self.plan_storage, length) catch {
    _ => raise Invalid(ControlBoundary)
  }
  if !(request.status is Pending) ||
    !request.host.same(request.command.host) ||
    request.command.sequence !=
    (inbound.sequence_value() catch { error => raise Control(error) }) {
    raise Invalid(ControlBoundary)
  }
  let reply = self.worker.streaming_command(request.command) catch {
    error => raise Worker(error)
  }
  @streaming_ir.encode(reply, self.plan_storage) catch {
    _ => raise Invalid(ControlBoundary)
  }
  self.control_owner.begin_streaming_reply(inbound, self.plan_storage) catch {
    error => raise Control(error)
  }
  self.send_active = true
}
