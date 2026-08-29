///|
fn RankChildControl::begin_response(
  self : RankChildControl,
  inbound : RankChildInbound,
  expected : @rank_group_wire.RankGroupWireKind,
  response : @rank_group_wire.RankGroupWireKind,
  payload : FixedArray[Byte]?,
  payload_length : Int,
) -> Unit raise RankChildControlError {
  self.require_usable()
  self.require_inbound_kind(inbound, expected)
  if self.send_active {
    return self.reject(SendBusy)
  }
  let binding = match self.transcript.binding() {
    Some(value) => value
    None => return self.poison(Binding)
  }
  let length = match payload {
    Some(bytes) =>
      @rank_group_wire.encode_rank_group_wire_payload_into(
        self.tx_storage,
        self.wire_limits,
        binding,
        response,
        ChildToParent,
        self.inbound_slot,
        self.inbound_sequence,
        bytes,
        payload_length,
      ) catch {
        error => return self.map_wire(error)
      }
    None =>
      @rank_group_wire.encode_rank_group_wire_control_into(
        self.tx_storage,
        self.wire_limits,
        binding,
        response,
        ChildToParent,
        self.inbound_slot,
        self.inbound_sequence,
      ) catch {
        error => return self.map_wire(error)
      }
  }
  ignore(
    self.transcript.accept_child(self.tx_storage, length, self.wire_limits) catch {
      error => return self.map_wire(error)
    },
  )
  self.channel.begin_write_frame(self.tx_storage, offset=0, byte_count=length) catch {
    error => return self.map_transport(error)
  }
  self.consume_inbound()
  self.send_active = true
}

///|
pub fn RankChildControl::begin_ready(
  self : RankChildControl,
  configure : RankChildInbound,
  admitted_binding : @rank_group_wire.RankGroupWireBinding,
  previous_sequence_value~ : UInt64,
) -> Unit raise RankChildControlError {
  self.require_inbound_kind(configure, Configure)
  if self.send_active {
    return self.reject(SendBusy)
  }
  self.transcript.admit_ready(admitted_binding, previous_sequence_value~) catch {
    error => return self.map_wire(error)
  }
  self.begin_response(configure, Configure, Ready, None, 0)
}

///|
pub fn RankChildControl::begin_pending(
  self : RankChildControl,
  poll : RankChildInbound,
) -> Unit raise RankChildControlError {
  self.begin_response(poll, PollComplete, Pending, None, 0)
}

///|
pub fn RankChildControl::begin_leader_complete(
  self : RankChildControl,
  poll : RankChildInbound,
  payload : FixedArray[Byte],
  payload_length~ : Int,
) -> Unit raise RankChildControlError {
  self.require_inbound_kind(poll, PollComplete)
  if poll.binding().rank() != 0 ||
    payload_length <= 0 ||
    payload_length > payload.length() {
    return self.reject(Payload)
  }
  self.begin_response(
    poll,
    PollComplete,
    Complete,
    Some(payload),
    payload_length,
  )
}

///|
pub fn RankChildControl::begin_follower_complete(
  self : RankChildControl,
  poll : RankChildInbound,
) -> Unit raise RankChildControlError {
  self.require_inbound_kind(poll, PollComplete)
  if poll.binding().rank() == 0 {
    return self.reject(Payload)
  }
  self.begin_response(poll, PollComplete, Complete, None, 0)
}

///|
/// Starts the distinct fixed graph-telemetry frame only after the canonical
/// rank-group `Complete` response has finished. This raw private sidecar does
/// not alter the rank-group or worker completion wire schemas.
pub fn RankChildControl::begin_graph_telemetry_sidecar(
  self : RankChildControl,
  report : @worker_wire.WorkerGraphTelemetry,
) -> Unit raise RankChildControlError {
  self.require_usable()
  if self.send_active ||
    self.transcript.lifecycle() != ReadyGroup ||
    report.sequence_value() == 0UL ||
    report.sequence_value() != self.transcript.previous_sequence_value() ||
    report.sequence_value() <= self.last_graph_sidecar_sequence {
    return self.reject(Binding)
  }
  let length = @worker_wire.encode_worker_graph_telemetry(
    report,
    self.tx_storage,
  ) catch {
    _ => return self.poison(Wire)
  }
  self.channel.begin_write_frame(self.tx_storage, offset=0, byte_count=length) catch {
    error => return self.map_transport(error)
  }
  self.last_graph_sidecar_sequence = report.sequence_value()
  self.send_active = true
}

///|
/// Publishes a payload-free collective failure for the exact inbound poll.
pub fn RankChildControl::begin_collective_failed(
  self : RankChildControl,
  poll : RankChildInbound,
) -> Unit raise RankChildControlError {
  self.begin_response(
    poll,
    PollComplete,
    PlanFailed(CollectiveFailure),
    None,
    0,
  )
}

///|
/// Publishes a payload-free rank-execution failure for the exact inbound poll.
pub fn RankChildControl::begin_execution_failed(
  self : RankChildControl,
  poll : RankChildInbound,
) -> Unit raise RankChildControlError {
  self.begin_response(
    poll,
    PollComplete,
    PlanFailed(RankExecutionFailure),
    None,
    0,
  )
}

///|
pub fn RankChildControl::begin_aborted(
  self : RankChildControl,
  abort_frame : RankChildInbound,
) -> Unit raise RankChildControlError {
  self.begin_response(abort_frame, Abort, Aborted, None, 0)
}

///|
pub fn RankChildControl::begin_drained(
  self : RankChildControl,
  drain : RankChildInbound,
) -> Unit raise RankChildControlError {
  self.begin_response(drain, Drain, Drained, None, 0)
}

///|
pub fn RankChildControl::begin_closed(
  self : RankChildControl,
  close : RankChildInbound,
) -> Unit raise RankChildControlError {
  self.begin_response(close, Close, Closed, None, 0)
}

///|
/// Advances at most one inherited-channel write. TX storage is retained until
/// the full prefix and payload have been published.
pub fn RankChildControl::progress_send(
  self : RankChildControl,
) -> RankChildSendProgress raise RankChildControlError {
  self.require_usable()
  if !self.send_active {
    return self.reject(SendBusy)
  }
  let complete = self.channel.progress_write_frame() catch {
    error => return self.map_transport(error)
  }
  if complete {
    self.send_active = false
    { kind: 1 }
  } else {
    { kind: 0 }
  }
}
