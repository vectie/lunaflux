///|
fn ValidatedRankGroupWireFrame::owner(
  self : ValidatedRankGroupWireFrame,
) -> RankGroupWireFrameBuffer raise RankGroupWireError {
  if self.1 == 0UL || self.0.epoch != self.1 || self.0.length == 0 {
    raise Invalid(FrameEpoch)
  }
  self.0
}

///|
pub fn ValidatedRankGroupWireFrame::kind(
  self : ValidatedRankGroupWireFrame,
) -> RankGroupWireKind raise RankGroupWireError {
  wire_kind_from_code(wire_read_i32(self.owner().storage, 16))
}

///|
pub fn ValidatedRankGroupWireFrame::direction(
  self : ValidatedRankGroupWireFrame,
) -> RankGroupWireDirection raise RankGroupWireError {
  direction_from_code(wire_read_i32(self.owner().storage, 60))
}

///|
pub fn ValidatedRankGroupWireFrame::slot(
  self : ValidatedRankGroupWireFrame,
) -> RankGroupWireSlot? raise RankGroupWireError {
  wire_slot_from_code(wire_read_i32(self.owner().storage, 20))
}

///|
pub fn ValidatedRankGroupWireFrame::sequence_value(
  self : ValidatedRankGroupWireFrame,
) -> UInt64 raise RankGroupWireError {
  wire_read_u64(self.owner().storage, 48)
}

///|
pub fn ValidatedRankGroupWireFrame::payload_length(
  self : ValidatedRankGroupWireFrame,
) -> Int raise RankGroupWireError {
  wire_read_i32(self.owner().storage, 56)
}

///|
pub fn ValidatedRankGroupWireFrame::length(
  self : ValidatedRankGroupWireFrame,
) -> Int raise RankGroupWireError {
  self.owner().length
}

///|
pub fn ValidatedRankGroupWireFrame::binding(
  self : ValidatedRankGroupWireFrame,
) -> RankGroupWireBinding raise RankGroupWireError {
  decode_rank_group_wire_binding(self.owner().storage)
}

///|
fn ValidatedRankGroupWireFrame::matches_binding(
  self : ValidatedRankGroupWireFrame,
  expected : RankGroupWireBinding,
) -> Bool raise RankGroupWireError {
  rank_group_wire_source_matches_binding(self.owner().storage, expected)
}

///|
pub fn ValidatedRankGroupWireFrame::copy_payload_to(
  self : ValidatedRankGroupWireFrame,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise RankGroupWireError {
  let owner = self.owner()
  let payload_length = wire_read_i32(owner.storage, 56)
  if destination_offset < 0 ||
    destination_offset > destination.length() - payload_length {
    raise Invalid(Destination)
  }
  for index in 0..<payload_length {
    destination[destination_offset + index] = owner.storage[RANK_GROUP_WIRE_HEADER_BYTES +
      index]
  }
}

///|
pub fn ValidatedRankGroupWireFrame::copy_to(
  self : ValidatedRankGroupWireFrame,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise RankGroupWireError {
  let owner = self.owner()
  if destination_offset < 0 ||
    destination_offset > destination.length() - owner.length {
    raise Invalid(Destination)
  }
  for index in 0..<owner.length {
    destination[destination_offset + index] = owner.storage[index]
  }
}
