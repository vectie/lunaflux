///|
const RANK_GROUP_WIRE_MAGIC : UInt = 0x5747524c

///|
const RANK_GROUP_WIRE_VERSION : UInt = 2

///|
const RANK_GROUP_WIRE_CHECKSUM_OFFSET : Int = 12

///|
fn wire_kind_code(kind : RankGroupWireKind) -> Int {
  match kind {
    Configure => 1
    Ready => 2
    Submit => 3
    PollComplete => 4
    Pending => 5
    Complete => 6
    Abort => 7
    Aborted => 8
    Drain => 9
    Drained => 10
    Close => 11
    Closed => 12
    PlanFailed(CollectiveFailure) => 13
    PlanFailed(RankExecutionFailure) => 14
  }
}

///|
fn wire_kind_from_code(
  code : Int,
) -> RankGroupWireKind raise RankGroupWireError {
  match code {
    1 => Configure
    2 => Ready
    3 => Submit
    4 => PollComplete
    5 => Pending
    6 => Complete
    7 => Abort
    8 => Aborted
    9 => Drain
    10 => Drained
    11 => Close
    12 => Closed
    13 => PlanFailed(CollectiveFailure)
    14 => PlanFailed(RankExecutionFailure)
    _ => raise Invalid(Kind)
  }
}

///|
fn wire_slot_code(slot : RankGroupWireSlot?) -> Int {
  match slot {
    None => -1
    Some(A) => 0
    Some(B) => 1
  }
}

///|
fn wire_slot_from_code(
  code : Int,
) -> RankGroupWireSlot? raise RankGroupWireError {
  match code {
    -1 => None
    0 => Some(A)
    1 => Some(B)
    _ => raise Invalid(Slot)
  }
}

///|
fn kind_uses_slot(kind : RankGroupWireKind) -> Bool {
  match kind {
    Submit | PollComplete | Pending | Complete | PlanFailed(_) => true
    Configure | Ready | Abort | Aborted | Drain | Drained | Close | Closed =>
      false
  }
}

///|
fn kind_uses_sequence(kind : RankGroupWireKind) -> Bool {
  kind_uses_slot(kind)
}

///|
fn direction_code(direction : RankGroupWireDirection) -> Int {
  match direction {
    ParentToChild => 0
    ChildToParent => 1
  }
}

///|
fn direction_from_code(
  code : Int,
) -> RankGroupWireDirection raise RankGroupWireError {
  match code {
    0 => ParentToChild
    1 => ChildToParent
    _ => raise Invalid(Direction)
  }
}

///|
fn expected_direction(kind : RankGroupWireKind) -> RankGroupWireDirection {
  match kind {
    Configure | Submit | PollComplete | Abort | Drain | Close => ParentToChild
    Ready | Pending | Complete | PlanFailed(_) | Aborted | Drained | Closed =>
      ChildToParent
  }
}

///|
fn kind_uses_payload(kind : RankGroupWireKind) -> Bool {
  match kind {
    Configure | Submit => true
    _ => false
  }
}

///|
fn validate_message_shape(
  kind : RankGroupWireKind,
  slot : RankGroupWireSlot?,
  sequence_value : UInt64,
  payload_length : Int,
) -> Unit raise RankGroupWireError {
  match kind {
    Abort | Aborted =>
      if (sequence_value == 0UL) != (slot is None) {
        raise Invalid(Slot)
      }
    _ => if kind_uses_slot(kind) != (slot is Some(_)) { raise Invalid(Slot) }
  }
  if kind_uses_sequence(kind) ||
    ((kind == Abort || kind == Aborted) && slot is Some(_)) {
    if sequence_value == 0UL {
      raise Invalid(Sequence)
    }
  } else if kind != Abort && kind != Aborted && sequence_value != 0UL {
    raise Invalid(Sequence)
  }
  if kind_uses_payload(kind) {
    if payload_length <= 0 {
      raise Invalid(Payload)
    }
  } else if kind != Complete && payload_length != 0 {
    raise Invalid(Payload)
  }
}

///|
fn wire_checksum(bytes : FixedArray[Byte], length : Int) -> UInt {
  let mut hash : UInt = 2166136261
  for index in 0..<length {
    if index < RANK_GROUP_WIRE_CHECKSUM_OFFSET ||
      index >= RANK_GROUP_WIRE_CHECKSUM_OFFSET + 4 {
      hash = (hash ^ bytes[index].to_uint()) * 16777619
    }
  }
  hash
}

///|
fn write_binding(
  destination : FixedArray[Byte],
  binding : RankGroupWireBinding,
) -> Unit {
  wire_write_i32(destination, 24, binding.rank)
  wire_write_i32(destination, 28, binding.world_size)
  wire_write_u64(destination, 32, binding.model_generation_value)
  wire_write_u64(destination, 40, binding.group_generation_value)
  for index in 0..<binding.canonical_digests.length() {
    destination[64 + index] = binding.canonical_digests[index]
  }
}

///|
fn validate_digest_bytes(
  source : FixedArray[Byte],
  offset : Int,
) -> Unit raise RankGroupWireError {
  for index in 0..<64 {
    let value = source[offset + index]
    if !((value >= b'0' && value <= b'9') || (value >= b'a' && value <= b'f')) {
      raise Invalid(Digest)
    }
  }
}

///|
fn validate_frame_source(
  source : FixedArray[Byte],
  length : Int,
  limits : RankGroupWireLimits,
) -> Unit raise RankGroupWireError {
  if length < RANK_GROUP_WIRE_HEADER_BYTES ||
    length > limits.max_frame_bytes() ||
    source.length() < length {
    raise Invalid(Length)
  }
  if wire_read_u32(source, 0) != RANK_GROUP_WIRE_MAGIC {
    raise Invalid(Magic)
  }
  if wire_read_u32(source, 4) != RANK_GROUP_WIRE_VERSION {
    raise Invalid(Version)
  }
  if wire_read_i32(source, 8) != length {
    raise Invalid(Length)
  }
  if wire_read_u32(source, 640) != 0U {
    raise Invalid(Reserved)
  }
  if wire_read_u32(source, RANK_GROUP_WIRE_CHECKSUM_OFFSET) !=
    wire_checksum(source, length) {
    raise Invalid(Checksum)
  }
  let kind = wire_kind_from_code(wire_read_i32(source, 16))
  let slot = wire_slot_from_code(wire_read_i32(source, 20))
  let direction = direction_from_code(wire_read_i32(source, 60))
  if direction != expected_direction(kind) {
    raise Invalid(Direction)
  }
  let payload_length = wire_read_i32(source, 56)
  if payload_length < 0 ||
    payload_length > limits.max_payload_bytes() ||
    RANK_GROUP_WIRE_HEADER_BYTES + payload_length != length {
    raise Invalid(Length)
  }
  validate_message_shape(kind, slot, wire_read_u64(source, 48), payload_length)
  let rank = wire_read_i32(source, 24)
  let world_size = wire_read_i32(source, 28)
  if world_size < 2 || world_size > RANK_GROUP_WIRE_MAX_WORLD_SIZE {
    raise Invalid(WorldSize)
  }
  if rank < 0 || rank >= world_size {
    raise Invalid(Rank)
  }
  if wire_read_u64(source, 32) == 0UL {
    raise Invalid(ModelGeneration)
  }
  if wire_read_u64(source, 40) == 0UL {
    raise Invalid(GroupGeneration)
  }
  let mut offset = 64
  while offset <= 576 {
    validate_digest_bytes(source, offset)
    offset += 64
  }
}

///|
fn RankGroupWireFrameBuffer::encode_internal(
  self : RankGroupWireFrameBuffer,
  binding : RankGroupWireBinding,
  kind : RankGroupWireKind,
  direction : RankGroupWireDirection,
  slot : RankGroupWireSlot?,
  sequence_value : UInt64,
  payload : FixedArray[Byte]?,
  payload_length : Int,
) -> ValidatedRankGroupWireFrame raise RankGroupWireError {
  if self.epoch == 0xffffffffffffffffUL {
    raise Invalid(FrameEpoch)
  }
  let length = encode_rank_group_wire_into(
    self.storage,
    self.limits,
    binding,
    kind,
    direction,
    slot,
    sequence_value,
    payload,
    payload_length,
  )
  self.epoch += 1UL
  self.length = length
  ValidatedRankGroupWireFrame(self, self.epoch)
}

///|
pub fn RankGroupWireFrameBuffer::new(
  limits : RankGroupWireLimits,
) -> RankGroupWireFrameBuffer {
  {
    limits,
    storage: FixedArray::make(limits.max_frame_bytes(), b'\x00'),
    epoch: 0UL,
    length: 0,
  }
}

///|
pub fn RankGroupWireFrameBuffer::capacity(
  self : RankGroupWireFrameBuffer,
) -> Int {
  self.storage.length()
}

///|
pub fn RankGroupWireFrameBuffer::encode_control(
  self : RankGroupWireFrameBuffer,
  binding : RankGroupWireBinding,
  kind : RankGroupWireKind,
  direction : RankGroupWireDirection,
  slot : RankGroupWireSlot?,
  sequence_value : UInt64,
) -> ValidatedRankGroupWireFrame raise RankGroupWireError {
  self.encode_internal(binding, kind, direction, slot, sequence_value, None, 0)
}

///|
pub fn RankGroupWireFrameBuffer::encode_payload(
  self : RankGroupWireFrameBuffer,
  binding : RankGroupWireBinding,
  kind : RankGroupWireKind,
  direction : RankGroupWireDirection,
  slot : RankGroupWireSlot,
  sequence_value : UInt64,
  payload : FixedArray[Byte],
  payload_length : Int,
) -> ValidatedRankGroupWireFrame raise RankGroupWireError {
  self.encode_internal(
    binding,
    kind,
    direction,
    Some(slot),
    sequence_value,
    Some(payload),
    payload_length,
  )
}

///|
/// Encodes the first `Configure` frame with bounded opaque startup bytes.
pub fn RankGroupWireFrameBuffer::encode_configure_payload(
  self : RankGroupWireFrameBuffer,
  binding : RankGroupWireBinding,
  payload : FixedArray[Byte],
  payload_length : Int,
) -> ValidatedRankGroupWireFrame raise RankGroupWireError {
  self.encode_internal(
    binding,
    Configure,
    ParentToChild,
    None,
    0UL,
    Some(payload),
    payload_length,
  )
}

///|
pub fn RankGroupWireFrameBuffer::load(
  self : RankGroupWireFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
) -> ValidatedRankGroupWireFrame raise RankGroupWireError {
  if self.epoch == 0xffffffffffffffffUL {
    raise Invalid(FrameEpoch)
  }
  ignore(decode_rank_group_wire_header(source, length, self.limits))
  for index in 0..<length {
    self.storage[index] = source[index]
  }
  self.epoch += 1UL
  self.length = length
  ValidatedRankGroupWireFrame(self, self.epoch)
}
