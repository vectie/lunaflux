///|
pub fn RankGroupWireLimits::new(
  max_payload_bytes~ : Int,
) -> RankGroupWireLimits raise RankGroupWireError {
  if max_payload_bytes <= 0 ||
    max_payload_bytes > RANK_GROUP_WIRE_MAX_PAYLOAD_BYTES {
    raise Invalid(Limit)
  }
  { max_payload_bytes, }
}

///|
/// The maximum payload is chosen so the canonical header plus payload fits
/// the process boundary's admitted 64 MiB whole-frame envelope.

///|
pub fn RankGroupWireLimits::max_payload_bytes(
  self : RankGroupWireLimits,
) -> Int {
  self.max_payload_bytes
}

///|
pub fn RankGroupWireLimits::max_frame_bytes(self : RankGroupWireLimits) -> Int {
  RANK_GROUP_WIRE_HEADER_BYTES + self.max_payload_bytes
}

///|
pub fn rank_group_wire_header_bytes() -> Int {
  RANK_GROUP_WIRE_HEADER_BYTES
}
