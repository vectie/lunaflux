///|
/// Source-independent scalar facts copied from one authenticated wire frame.
///
/// This value retains no reference to the caller-owned source storage.
#valtype
pub struct RankGroupWireDecodedHeader {
  priv kind : RankGroupWireKind
  priv direction : RankGroupWireDirection
  priv slot : RankGroupWireSlot?
  priv sequence_value : UInt64
  priv payload_length : Int
  priv length : Int
} derive(Debug, Eq)

///|
/// The authenticated first `Configure` header and its copied exact binding.
///
/// This value retains no reference to the caller-owned source storage.
pub struct RankGroupWireDecodedConfigure {
  priv header : RankGroupWireDecodedHeader
  priv binding : RankGroupWireBinding
}

///|
pub fn RankGroupWireDecodedHeader::kind(
  self : RankGroupWireDecodedHeader,
) -> RankGroupWireKind {
  self.kind
}

///|
pub fn RankGroupWireDecodedHeader::direction(
  self : RankGroupWireDecodedHeader,
) -> RankGroupWireDirection {
  self.direction
}

///|
pub fn RankGroupWireDecodedHeader::slot(
  self : RankGroupWireDecodedHeader,
) -> RankGroupWireSlot? {
  self.slot
}

///|
pub fn RankGroupWireDecodedHeader::sequence_value(
  self : RankGroupWireDecodedHeader,
) -> UInt64 {
  self.sequence_value
}

///|
pub fn RankGroupWireDecodedHeader::payload_length(
  self : RankGroupWireDecodedHeader,
) -> Int {
  self.payload_length
}

///|
pub fn RankGroupWireDecodedHeader::length(
  self : RankGroupWireDecodedHeader,
) -> Int {
  self.length
}

///|
pub fn RankGroupWireDecodedConfigure::header(
  self : RankGroupWireDecodedConfigure,
) -> RankGroupWireDecodedHeader {
  self.header
}

///|
pub fn RankGroupWireDecodedConfigure::binding(
  self : RankGroupWireDecodedConfigure,
) -> RankGroupWireBinding {
  self.binding
}
