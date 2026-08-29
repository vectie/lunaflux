///|
fn RankChildInbound::owner(
  self : RankChildInbound,
) -> RankChildControl raise RankChildControlError {
  if self.1 == 0UL || !self.0.inbound_active || self.0.inbound_epoch != self.1 {
    raise Invalid(Epoch)
  }
  self.0.require_usable()
  self.0
}

///|
pub fn RankChildInbound::kind(
  self : RankChildInbound,
) -> @rank_group_wire.RankGroupWireKind raise RankChildControlError {
  self.owner().inbound_kind
}

///|
pub fn RankChildInbound::slot(
  self : RankChildInbound,
) -> @rank_group_wire.RankGroupWireSlot? raise RankChildControlError {
  self.owner().inbound_slot
}

///|
pub fn RankChildInbound::sequence_value(
  self : RankChildInbound,
) -> UInt64 raise RankChildControlError {
  self.owner().inbound_sequence
}

///|
pub fn RankChildInbound::payload_length(
  self : RankChildInbound,
) -> Int raise RankChildControlError {
  self.owner().inbound_payload_length
}

///|
pub fn RankChildInbound::binding(
  self : RankChildInbound,
) -> @rank_group_wire.RankGroupWireBinding raise RankChildControlError {
  let owner = self.owner()
  match owner.transcript.binding() {
    Some(binding) => binding
    None => owner.reject(Binding)
  }
}

///|
pub fn RankChildInbound::copy_payload_to(
  self : RankChildInbound,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise RankChildControlError {
  let owner = self.owner()
  let payload_length = owner.inbound_payload_length
  if destination_offset < 0 ||
    destination_offset > destination.length() - payload_length {
    return owner.reject(Payload)
  }
  let header_bytes = @rank_group_wire.rank_group_wire_header_bytes()
  for index in 0..<payload_length {
    destination[destination_offset + index] = owner.rx_destination[header_bytes +
      index]
  }
}

///|
fn RankChildControl::require_inbound_kind(
  self : RankChildControl,
  inbound : RankChildInbound,
  expected : @rank_group_wire.RankGroupWireKind,
) -> Unit raise RankChildControlError {
  let actual = inbound.owner()
  if !physical_equal(actual, self) || actual.inbound_kind != expected {
    return self.reject(Lifecycle)
  }
}

///|
fn RankChildControl::consume_inbound(self : RankChildControl) -> Unit {
  self.inbound_active = false
  self.inbound_payload_length = 0
}

///|
/// Consumes one locally admitted Submit without fabricating a response.
pub fn RankChildControl::accept_submit(
  self : RankChildControl,
  inbound : RankChildInbound,
) -> Unit raise RankChildControlError {
  self.require_inbound_kind(inbound, Submit)
  self.consume_inbound()
}
