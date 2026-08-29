///|
pub fn RankChildControl::begin_receive(
  self : RankChildControl,
) -> Unit raise RankChildControlError {
  self.require_usable()
  if self.receive_active || self.receive_held {
    return self.reject(ReceiveBusy)
  }
  if self.inbound_active {
    return self.reject(Lifecycle)
  }
  self.channel.begin_read_frame_or_eof(
    self.rx_destination,
    staging=self.rx_staging,
  ) catch {
    error => return self.map_transport(error)
  }
  self.receive_active = true
}

///|
fn RankChildControl::publish_received(
  self : RankChildControl,
  length : Int,
) -> RankChildReceiveProgress raise RankChildControlError {
  let header = if self.transcript.lifecycle() == NewGroup {
    let configured = self.transcript.accept_configure(
      self.rx_destination,
      length,
      self.wire_limits,
    ) catch {
      error => return self.map_wire(error)
    }
    configured.header()
  } else {
    self.transcript.accept_parent(self.rx_destination, length, self.wire_limits) catch {
      error => return self.map_wire(error)
    }
  }
  if self.inbound_epoch == 0xffffffffffffffffUL {
    return self.poison(Epoch)
  }
  self.inbound_epoch += 1UL
  self.inbound_kind = header.kind()
  self.inbound_slot = header.slot()
  self.inbound_sequence = header.sequence_value()
  self.inbound_payload_length = header.payload_length()
  self.inbound_active = true
  { kind: 2 }
}

///|
fn RankChildControl::finish_held_receive(
  self : RankChildControl,
) -> RankChildReceiveProgress raise RankChildControlError {
  if self.send_active {
    return { kind: 1 }
  }
  let held_eof = self.receive_held_eof
  let held_length = self.receive_held_length
  self.receive_held = false
  self.receive_held_eof = false
  self.receive_held_length = 0
  if held_eof {
    if self.transcript.lifecycle() == ClosedGroup {
      return { kind: 3 }
    }
    return self.poison(Transport)
  }
  self.publish_received(held_length)
}

///|
/// Advances at most one inherited-channel read. A completed frame is held
/// without validation while an earlier response still owns TX storage.
pub fn RankChildControl::progress_receive(
  self : RankChildControl,
) -> RankChildReceiveProgress raise RankChildControlError {
  self.require_usable()
  if self.receive_held {
    return self.finish_held_receive()
  }
  if !self.receive_active {
    return self.reject(ReceiveBusy)
  }
  let result = self.channel.progress_read_frame_or_eof() catch {
    error => return self.map_transport(error)
  }
  match result {
    None => { kind: 0 }
    Some(length) => {
      self.receive_active = false
      if self.send_active {
        self.receive_held = true
        self.receive_held_eof = length == 0
        self.receive_held_length = length
        return { kind: 1 }
      }
      if length == 0 {
        if self.transcript.lifecycle() == ClosedGroup {
          { kind: 3 }
        } else {
          self.poison(Transport)
        }
      } else {
        self.publish_received(length)
      }
    }
  }
}

///|
pub fn RankChildControl::inbound(
  self : RankChildControl,
) -> RankChildInbound raise RankChildControlError {
  self.require_usable()
  if !self.inbound_active || self.inbound_epoch == 0UL {
    return self.reject(Lifecycle)
  }
  RankChildInbound(self, self.inbound_epoch)
}
