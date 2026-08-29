///|
const RANK_ENVELOPE_MAGIC : UInt = 0x4254504c

///|
const RANK_ENVELOPE_VERSION : UInt = 3

///|
const RANK_ENVELOPE_LENGTH : Int = 752

///|
const RANK_ENVELOPE_CHECKSUM_OFFSET : Int = 12

///|
const RANK_ENVELOPE_RENDEZVOUS_OFFSET : Int = 624

///|
fn rank_envelope_checksum(bytes : FixedArray[Byte]) -> UInt {
  let mut hash : UInt = 2166136261
  for index in 0..<RANK_ENVELOPE_LENGTH {
    if index < RANK_ENVELOPE_CHECKSUM_OFFSET ||
      index >= RANK_ENVELOPE_CHECKSUM_OFFSET + 4 {
      hash = (hash ^ bytes[index].to_uint()) * 16777619
    }
  }
  hash
}

///|
fn frame_write_digest(
  bytes : FixedArray[Byte],
  offset : Int,
  value : StringView,
) -> Unit raise TensorParallelWorkerBootstrapError {
  if !is_lower_sha256(value) {
    raise rejected(Digest, -1)
  }
  let encoded = @utf8.encode(value)
  for index in 0..<64 {
    bytes[offset + index] = encoded[index]
  }
}

///|
fn frame_read_digest(
  bytes : FixedArray[Byte],
  offset : Int,
) -> String raise TensorParallelWorkerBootstrapError {
  let encoded = Bytes::makei(64, index => bytes[offset + index])
  let value = @utf8.decode(encoded) catch { _ => raise rejected(Digest, -1) }
  if !is_lower_sha256(value) {
    raise rejected(Digest, -1)
  }
  value
}

///|
fn write_rank_envelope(
  bytes : FixedArray[Byte],
  envelope : TensorParallelRankBootstrapEnvelope,
) -> Unit raise TensorParallelWorkerBootstrapError {
  for index in 0..<RANK_ENVELOPE_LENGTH {
    bytes[index] = b'\x00'
  }
  let rendezvous = envelope.rendezvous_id.to_bytes()
  if rendezvous.length() != 128 {
    raise rejected(RendezvousId, envelope.rank)
  }
  frame_write_u32(bytes, 0, RANK_ENVELOPE_MAGIC)
  frame_write_u32(bytes, 4, RANK_ENVELOPE_VERSION)
  frame_write_i32(bytes, 8, RANK_ENVELOPE_LENGTH)
  frame_write_i32(bytes, 16, envelope.rank)
  frame_write_i32(bytes, 20, envelope.world_size)
  frame_write_i32(bytes, 24, envelope.device_ordinal)
  frame_write_i32(bytes, 28, envelope.collective_runtime_version_code)
  frame_write_u64(bytes, 32, envelope.model_generation_value)
  frame_write_u64(bytes, 40, envelope.group_generation_value)
  frame_write_digest(bytes, 48, envelope.identity.content().as_hex())
  frame_write_digest(bytes, 112, envelope.identity.plan().as_hex())
  frame_write_digest(bytes, 176, envelope.group_digest.hex)
  frame_write_digest(bytes, 240, envelope.topology_digest.hex)
  frame_write_digest(bytes, 304, envelope.worker_digest.hex)
  frame_write_digest(bytes, 368, envelope.device_plan_digest.hex)
  frame_write_digest(bytes, 432, envelope.execution_plan_digest_sha256)
  frame_write_digest(bytes, 496, envelope.collective_digest.as_hex())
  frame_write_digest(bytes, 560, envelope.kv_digest.hex)
  for index in 0..<128 {
    bytes[RANK_ENVELOPE_RENDEZVOUS_OFFSET + index] = rendezvous[index]
  }
  frame_write_u32(
    bytes,
    RANK_ENVELOPE_CHECKSUM_OFFSET,
    rank_envelope_checksum(bytes),
  )
}

///|
fn validate_rank_envelope_source(
  bytes : FixedArray[Byte],
  length : Int,
) -> Unit raise TensorParallelWorkerBootstrapError {
  if length != RANK_ENVELOPE_LENGTH || bytes.length() < RANK_ENVELOPE_LENGTH {
    raise rejected(FrameLength, -1)
  }
  if frame_read_u32(bytes, 0) != RANK_ENVELOPE_MAGIC {
    raise rejected(FrameMagic, -1)
  }
  if frame_read_u32(bytes, 4) != RANK_ENVELOPE_VERSION {
    raise rejected(FrameVersion, -1)
  }
  if frame_read_i32(bytes, 8) != RANK_ENVELOPE_LENGTH {
    raise rejected(FrameLength, -1)
  }
  if frame_read_i32(bytes, 28) <= 0 {
    raise rejected(FrameReserved, -1)
  }
  if frame_read_u32(bytes, RANK_ENVELOPE_CHECKSUM_OFFSET) !=
    rank_envelope_checksum(bytes) {
    raise rejected(FrameChecksum, -1)
  }
  let rank = frame_read_i32(bytes, 16)
  let world_size = frame_read_i32(bytes, 20)
  let ordinal = frame_read_i32(bytes, 24)
  if world_size < 2 || world_size > 16 {
    raise rejected(WorldSize, rank)
  }
  if rank < 0 || rank >= world_size {
    raise rejected(Rank, rank)
  }
  if ordinal < 0 || ordinal > 1023 {
    raise rejected(DeviceOrdinal, rank)
  }
  if frame_read_u64(bytes, 32) == 0UL {
    raise rejected(ModelGeneration, rank)
  }
  if frame_read_u64(bytes, 40) == 0UL {
    raise rejected(GroupGeneration, rank)
  }
  for offset in [48, 112, 176, 240, 304, 368, 432, 496, 560] {
    ignore(frame_read_digest(bytes, offset))
  }
}

///|
fn decode_rank_envelope(
  bytes : FixedArray[Byte],
) -> TensorParallelRankBootstrapEnvelope raise TensorParallelWorkerBootstrapError {
  // Decoded identity is an untrusted claim until exact admission against a
  // rebuilt ModelPlan identity.
  let content = @spec.ContentDigest::from_sha256(frame_read_digest(bytes, 48)) catch {
    _ => raise rejected(ModelIdentity, -1)
  }
  let plan = @spec.PlanDigest::from_sha256(frame_read_digest(bytes, 112)) catch {
    _ => raise rejected(ModelIdentity, -1)
  }
  let identity = @spec.ModelIdentity::new(content, plan) catch {
    _ => raise rejected(ModelIdentity, -1)
  }
  let collective_digest = @device.TensorCollectiveContractDigest::from_sha256(
    frame_read_digest(bytes, 496),
  ) catch {
    _ => raise rejected(CollectiveContract, -1)
  }
  let rendezvous_bytes = Bytes::makei(128, index => {
    bytes[RANK_ENVELOPE_RENDEZVOUS_OFFSET + index]
  })
  let rendezvous_id = @device.tensor_collective_group_id_from_bytes(
    rendezvous_bytes,
  ) catch {
    _ => raise rejected(RendezvousId, -1)
  }
  {
    identity,
    model_generation_value: frame_read_u64(bytes, 32),
    group_generation_value: frame_read_u64(bytes, 40),
    rank: frame_read_i32(bytes, 16),
    world_size: frame_read_i32(bytes, 20),
    device_ordinal: frame_read_i32(bytes, 24),
    collective_runtime_version_code: frame_read_i32(bytes, 28),
    group_digest: group_digest_from_hex(frame_read_digest(bytes, 176)),
    topology_digest: topology_digest_from_hex(frame_read_digest(bytes, 240)),
    worker_digest: worker_digest_from_hex(frame_read_digest(bytes, 304)),
    device_plan_digest: device_plan_digest_from_hex(
      frame_read_digest(bytes, 368),
    ),
    execution_plan_digest_sha256: frame_read_digest(bytes, 432),
    collective_digest,
    kv_digest: kv_binding_digest_from_hex(frame_read_digest(bytes, 560)),
    rendezvous_id,
  }
}

///|
pub fn tensor_parallel_rank_envelope_frame_bytes() -> Int {
  RANK_ENVELOPE_LENGTH
}

///|
pub fn TensorParallelRankEnvelopeFrameBuffer::new() -> TensorParallelRankEnvelopeFrameBuffer {
  {
    storage: FixedArray::make(RANK_ENVELOPE_LENGTH, b'\x00'),
    epoch: 0UL,
    length: 0,
  }
}

///|
pub fn TensorParallelRankEnvelopeFrameBuffer::capacity(
  self : TensorParallelRankEnvelopeFrameBuffer,
) -> Int {
  self.storage.length()
}

///|
pub fn TensorParallelRankEnvelopeFrameBuffer::encode(
  self : TensorParallelRankEnvelopeFrameBuffer,
  envelope : TensorParallelRankBootstrapEnvelope,
) -> ValidatedTensorParallelRankEnvelopeFrame raise TensorParallelWorkerBootstrapError {
  if self.epoch == 0xffffffffffffffffUL {
    raise FrameEpochOverflow
  }
  write_rank_envelope(self.storage, envelope)
  self.epoch += 1UL
  self.length = RANK_ENVELOPE_LENGTH
  ValidatedTensorParallelRankEnvelopeFrame(self, self.epoch)
}

///|
pub fn TensorParallelRankEnvelopeFrameBuffer::load(
  self : TensorParallelRankEnvelopeFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
) -> ValidatedTensorParallelRankEnvelopeFrame raise TensorParallelWorkerBootstrapError {
  if self.epoch == 0xffffffffffffffffUL {
    raise FrameEpochOverflow
  }
  validate_rank_envelope_source(source, length)
  for index in 0..<RANK_ENVELOPE_LENGTH {
    self.storage[index] = source[index]
  }
  self.epoch += 1UL
  self.length = RANK_ENVELOPE_LENGTH
  ValidatedTensorParallelRankEnvelopeFrame(self, self.epoch)
}

///|
fn ValidatedTensorParallelRankEnvelopeFrame::owner(
  self : ValidatedTensorParallelRankEnvelopeFrame,
) -> TensorParallelRankEnvelopeFrameBuffer raise TensorParallelWorkerBootstrapError {
  if self.1 == 0UL || self.0.epoch != self.1 || self.0.length == 0 {
    raise StaleFrame
  }
  self.0
}

///|
pub fn ValidatedTensorParallelRankEnvelopeFrame::envelope(
  self : ValidatedTensorParallelRankEnvelopeFrame,
) -> TensorParallelRankBootstrapEnvelope raise TensorParallelWorkerBootstrapError {
  decode_rank_envelope(self.owner().storage)
}

///|
pub fn ValidatedTensorParallelRankEnvelopeFrame::length(
  self : ValidatedTensorParallelRankEnvelopeFrame,
) -> Int raise TensorParallelWorkerBootstrapError {
  self.owner().length
}

///|
pub fn ValidatedTensorParallelRankEnvelopeFrame::copy_to(
  self : ValidatedTensorParallelRankEnvelopeFrame,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise TensorParallelWorkerBootstrapError {
  let owner = self.owner()
  if destination_offset < 0 ||
    destination_offset > destination.length() - owner.length {
    raise DestinationTooSmall
  }
  for index in 0..<owner.length {
    destination[destination_offset + index] = owner.storage[index]
  }
}
