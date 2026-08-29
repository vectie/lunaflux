///|
fn decode_wire_digest_string(
  source : FixedArray[Byte],
  offset : Int,
) -> String raise RankGroupWireError {
  let encoded = Bytes::makei(64, index => source[offset + index])
  let value = @utf8.decode(encoded) catch { _ => raise Invalid(Digest) }
  if !is_lower_sha256(value) {
    raise Invalid(Digest)
  }
  value
}

///|
fn decode_rank_group_wire_binding(
  source : FixedArray[Byte],
) -> RankGroupWireBinding raise RankGroupWireError {
  // Decoded identity is an untrusted claim until exact admission against a
  // rebuilt ModelPlan identity.
  let content = @spec.ContentDigest::from_sha256(
    decode_wire_digest_string(source, 64),
  ) catch {
    _ => raise Invalid(ModelIdentity)
  }
  let plan = @spec.PlanDigest::from_sha256(
    decode_wire_digest_string(source, 128),
  ) catch {
    _ => raise Invalid(ModelIdentity)
  }
  let identity = @spec.ModelIdentity::new(content, plan) catch {
    _ => raise Invalid(ModelIdentity)
  }
  RankGroupWireBinding::new(
    identity~,
    model_generation_value=wire_read_u64(source, 32),
    group_generation_value=wire_read_u64(source, 40),
    rank=wire_read_i32(source, 24),
    world_size=wire_read_i32(source, 28),
    group_digest=decode_wire_digest_string(source, 192),
    topology_digest=decode_wire_digest_string(source, 256),
    worker_digest=decode_wire_digest_string(source, 320),
    device_plan_digest=decode_wire_digest_string(source, 384),
    execution_plan_digest=decode_wire_digest_string(source, 448),
    collective_digest=decode_wire_digest_string(source, 512),
    kv_digest=decode_wire_digest_string(source, 576),
  )
}

///|
fn rank_group_wire_source_matches_binding(
  source : FixedArray[Byte],
  expected : RankGroupWireBinding,
) -> Bool {
  if wire_read_i32(source, 24) != expected.rank ||
    wire_read_i32(source, 28) != expected.world_size ||
    wire_read_u64(source, 32) != expected.model_generation_value ||
    wire_read_u64(source, 40) != expected.group_generation_value {
    return false
  }
  for index in 0..<expected.canonical_digests.length() {
    if source[64 + index] != expected.canonical_digests[index] {
      return false
    }
  }
  true
}

///|
/// Authenticates one caller-owned frame and copies only scalar header facts.
///
/// The result does not retain `source`.
pub fn decode_rank_group_wire_header(
  source : FixedArray[Byte],
  length : Int,
  limits : RankGroupWireLimits,
) -> RankGroupWireDecodedHeader raise RankGroupWireError {
  validate_frame_source(source, length, limits)
  {
    kind: wire_kind_from_code(wire_read_i32(source, 16)),
    direction: direction_from_code(wire_read_i32(source, 60)),
    slot: wire_slot_from_code(wire_read_i32(source, 20)),
    sequence_value: wire_read_u64(source, 48),
    payload_length: wire_read_i32(source, 56),
    length,
  }
}

///|
/// Authenticates the first nonempty `Configure` frame and copies its binding.
///
/// The result does not retain `source`. Constructing the binding is intended
/// only for startup; subsequent frames should use
/// `validate_rank_group_wire_binding`.
pub fn decode_rank_group_wire_configure(
  source : FixedArray[Byte],
  length : Int,
  limits : RankGroupWireLimits,
) -> RankGroupWireDecodedConfigure raise RankGroupWireError {
  let header = decode_rank_group_wire_header(source, length, limits)
  if header.kind() != Configure {
    raise Invalid(Kind)
  }
  { header, binding: decode_rank_group_wire_binding(source) }
}

///|
/// Authenticates a frame against an already copied exact binding.
///
/// This steady-state operation allocates no binding and retains no source
/// reference in its scalar result.
pub fn validate_rank_group_wire_binding(
  source : FixedArray[Byte],
  length : Int,
  limits : RankGroupWireLimits,
  binding : RankGroupWireBinding,
) -> RankGroupWireDecodedHeader raise RankGroupWireError {
  let header = decode_rank_group_wire_header(source, length, limits)
  if !rank_group_wire_source_matches_binding(source, binding) {
    raise Invalid(Binding)
  }
  header
}

///|
fn encode_rank_group_wire_into(
  destination : FixedArray[Byte],
  limits : RankGroupWireLimits,
  binding : RankGroupWireBinding,
  kind : RankGroupWireKind,
  direction : RankGroupWireDirection,
  slot : RankGroupWireSlot?,
  sequence_value : UInt64,
  payload : FixedArray[Byte]?,
  payload_length : Int,
) -> Int raise RankGroupWireError {
  if payload_length < 0 || payload_length > limits.max_payload_bytes() {
    raise Invalid(Payload)
  }
  match payload {
    Some(source) =>
      if source.length() < payload_length {
        raise Invalid(Payload)
      }
    None => if payload_length != 0 { raise Invalid(Payload) }
  }
  validate_message_shape(kind, slot, sequence_value, payload_length)
  if direction != expected_direction(kind) {
    raise Invalid(Direction)
  }
  let length = RANK_GROUP_WIRE_HEADER_BYTES + payload_length
  if destination.length() < length {
    raise Invalid(Destination)
  }
  // Copy high-to-low before touching the header so the operation also remains
  // deterministic when a caller passes the destination itself as payload.
  match payload {
    Some(source) => {
      let mut remaining = payload_length
      while remaining > 0 {
        remaining -= 1
        destination[RANK_GROUP_WIRE_HEADER_BYTES + remaining] = source[remaining]
      }
    }
    None => ()
  }
  for index in 0..<RANK_GROUP_WIRE_HEADER_BYTES {
    destination[index] = b'\x00'
  }
  wire_write_u32(destination, 0, RANK_GROUP_WIRE_MAGIC)
  wire_write_u32(destination, 4, RANK_GROUP_WIRE_VERSION)
  wire_write_i32(destination, 8, length)
  wire_write_i32(destination, 16, wire_kind_code(kind))
  wire_write_i32(destination, 20, wire_slot_code(slot))
  write_binding(destination, binding)
  wire_write_u64(destination, 48, sequence_value)
  wire_write_i32(destination, 56, payload_length)
  wire_write_i32(destination, 60, direction_code(direction))
  wire_write_u32(destination, 640, 0U)
  wire_write_u32(
    destination,
    RANK_GROUP_WIRE_CHECKSUM_OFFSET,
    wire_checksum(destination, length),
  )
  length
}

///|
/// Encodes one zero-payload control into caller-owned storage.
pub fn encode_rank_group_wire_control_into(
  destination : FixedArray[Byte],
  limits : RankGroupWireLimits,
  binding : RankGroupWireBinding,
  kind : RankGroupWireKind,
  direction : RankGroupWireDirection,
  slot : RankGroupWireSlot?,
  sequence_value : UInt64,
) -> Int raise RankGroupWireError {
  encode_rank_group_wire_into(
    destination,
    limits,
    binding,
    kind,
    direction,
    slot,
    sequence_value,
    None,
    0,
  )
}

///|
/// Encodes one payload-form frame into caller-owned storage.
///
/// `slot` is optional because startup `Configure` carries opaque admission
/// bytes without an execution slot, while `Submit` and leader `Complete`
/// carry an exact live slot.
pub fn encode_rank_group_wire_payload_into(
  destination : FixedArray[Byte],
  limits : RankGroupWireLimits,
  binding : RankGroupWireBinding,
  kind : RankGroupWireKind,
  direction : RankGroupWireDirection,
  slot : RankGroupWireSlot?,
  sequence_value : UInt64,
  payload : FixedArray[Byte],
  payload_length : Int,
) -> Int raise RankGroupWireError {
  encode_rank_group_wire_into(
    destination,
    limits,
    binding,
    kind,
    direction,
    slot,
    sequence_value,
    Some(payload),
    payload_length,
  )
}

///|
/// Seals a payload already encoded at the canonical header offset. This is the
/// copy-free production form: the caller exclusively owns `destination`, and
/// must not mutate it from this call until the framed write completes.
pub fn seal_rank_group_wire_embedded_payload(
  destination : FixedArray[Byte],
  limits : RankGroupWireLimits,
  binding : RankGroupWireBinding,
  kind : RankGroupWireKind,
  direction : RankGroupWireDirection,
  slot : RankGroupWireSlot?,
  sequence_value : UInt64,
  payload_length : Int,
) -> Int raise RankGroupWireError {
  if payload_length < 0 || payload_length > limits.max_payload_bytes() {
    raise Invalid(Payload)
  }
  validate_message_shape(kind, slot, sequence_value, payload_length)
  if direction != expected_direction(kind) {
    raise Invalid(Direction)
  }
  let length = RANK_GROUP_WIRE_HEADER_BYTES + payload_length
  if destination.length() < length {
    raise Invalid(Destination)
  }
  for index in 0..<RANK_GROUP_WIRE_HEADER_BYTES {
    destination[index] = b'\x00'
  }
  wire_write_u32(destination, 0, RANK_GROUP_WIRE_MAGIC)
  wire_write_u32(destination, 4, RANK_GROUP_WIRE_VERSION)
  wire_write_i32(destination, 8, length)
  wire_write_i32(destination, 16, wire_kind_code(kind))
  wire_write_i32(destination, 20, wire_slot_code(slot))
  write_binding(destination, binding)
  wire_write_u64(destination, 48, sequence_value)
  wire_write_i32(destination, 56, payload_length)
  wire_write_i32(destination, 60, direction_code(direction))
  wire_write_u32(destination, 640, 0U)
  wire_write_u32(
    destination,
    RANK_GROUP_WIRE_CHECKSUM_OFFSET,
    wire_checksum(destination, length),
  )
  length
}
