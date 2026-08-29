///|
const RANK_CONFIGURE_MAGIC : UInt = 0x4352464c

///|
const RANK_CONFIGURE_VERSION : UInt = 1

///|
const RANK_CONFIGURE_HEADER_BYTES : Int = 128

///|
const RANK_CONFIGURE_CHECKSUM_OFFSET : Int = 12

///|
const RANK_CONFIGURE_DIGEST_OFFSET : Int = 96

///|
const RANK_CONFIGURE_CAPACITY : Int = 38912

///|
pub fn tensor_parallel_rank_configure_capacity() -> Int {
  RANK_CONFIGURE_CAPACITY
}

///|
fn rank_configure_digest(
  bytes : FixedArray[Byte],
  length : Int,
) -> TensorParallelRankConfigureDigest {
  TensorParallelRankConfigureDigest(
    @crypto.bytes_to_hex_string(
      @crypto.sha256(
        Bytes::makei(length, index => {
          if (
              index >= RANK_CONFIGURE_CHECKSUM_OFFSET &&
              index < RANK_CONFIGURE_CHECKSUM_OFFSET + 4
            ) ||
            (
              index >= RANK_CONFIGURE_DIGEST_OFFSET &&
              index < RANK_CONFIGURE_DIGEST_OFFSET + 32
            ) {
            b'\x00'
          } else {
            bytes[index]
          }
        }),
      ),
    ),
  )
}

///|
fn write_rank_configure_header(
  bytes : FixedArray[Byte],
  length : Int,
  startup_length : Int,
  source_length : Int,
  topology_length : Int,
  envelope_length : Int,
  startup : @worker_wire.WorkerStartupContract,
  envelope : @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope,
) -> Unit raise TensorParallelRankConfigureError {
  configure_write_u32(bytes, 0, RANK_CONFIGURE_MAGIC)
  configure_write_u32(bytes, 4, RANK_CONFIGURE_VERSION)
  configure_write_i32(bytes, 8, length)
  configure_write_i32(bytes, 16, startup_length)
  configure_write_i32(bytes, 20, source_length)
  configure_write_i32(bytes, 24, topology_length)
  configure_write_i32(bytes, 28, envelope_length)
  configure_write_u64(bytes, 32, startup.model_generation_value())
  configure_write_u64(bytes, 40, envelope.group_generation_value())
  configure_write_u64(bytes, 48, startup.predecessor_value())
  configure_write_i32(bytes, 56, envelope.rank())
  configure_write_i32(bytes, 60, envelope.world_size())
  configure_write_i32(bytes, 64, envelope.device_ordinal())
  configure_write_i32(bytes, 68, if startup.has_predecessor() { 1 } else { 0 })
  let digest = rank_configure_digest(bytes, length)
  configure_write_raw_digest(
    bytes,
    RANK_CONFIGURE_DIGEST_OFFSET,
    digest.as_sha256(),
  )
  configure_write_u32(
    bytes,
    RANK_CONFIGURE_CHECKSUM_OFFSET,
    configure_checksum(bytes, 0, length, RANK_CONFIGURE_CHECKSUM_OFFSET),
  )
}

///|
pub fn compose_rank_configure(
  binding : @rank_group_wire.RankGroupWireBinding,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  topology : @device_topology.LocalTopologyDeclaration,
  envelope : @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope,
) -> TensorParallelRankConfigureContract raise TensorParallelRankConfigureError {
  validate_configure_claims(binding, startup, source, topology, envelope)
  let storage = FixedArray::make(RANK_CONFIGURE_CAPACITY, b'\x00')
  let startup_buffer = @worker_wire.StartupFrameBuffer::new()
  let startup_frame = startup_buffer.encode(Configure, startup) catch {
    _ => raise Invalid(Startup)
  }
  let startup_length = startup_frame.length() catch {
    _ => raise Invalid(Startup)
  }
  let startup_offset = RANK_CONFIGURE_HEADER_BYTES
  startup_frame.copy_to(storage, destination_offset=startup_offset) catch {
    _ => raise Invalid(Startup)
  }
  let source_offset = startup_offset + startup_length
  let source_length = source.length()
  source.copy_to(storage, destination_offset=source_offset) catch {
    _ => raise Invalid(BootstrapSource)
  }
  let topology_offset = source_offset + source_length
  let topology_length = write_selected_topology(
    storage, topology_offset, topology,
  )
  let envelope_offset = topology_offset + topology_length
  let envelope_buffer = @tensor_parallel_worker_bootstrap.TensorParallelRankEnvelopeFrameBuffer::new()
  let envelope_frame = envelope_buffer.encode(envelope) catch {
    _ => raise Invalid(RankEnvelope)
  }
  let envelope_length = envelope_frame.length() catch {
    _ => raise Invalid(RankEnvelope)
  }
  let length = envelope_offset + envelope_length
  if length > RANK_CONFIGURE_CAPACITY {
    raise Invalid(Limit)
  }
  envelope_frame.copy_to(storage, destination_offset=envelope_offset) catch {
    _ => raise Invalid(RankEnvelope)
  }
  write_rank_configure_header(
    storage, length, startup_length, source_length, topology_length, envelope_length,
    startup, envelope,
  )
  {
    storage,
    length,
    digest: configure_read_raw_digest(storage, RANK_CONFIGURE_DIGEST_OFFSET),
    binding,
    startup,
    source,
    topology,
    envelope,
  }
}

///|
fn validate_rank_configure_header(
  source : FixedArray[Byte],
  length : Int,
) -> Unit raise TensorParallelRankConfigureError {
  if length < RANK_CONFIGURE_HEADER_BYTES ||
    length > RANK_CONFIGURE_CAPACITY ||
    source.length() < length ||
    configure_read_i32(source, 8) != length {
    raise Invalid(Length)
  }
  if configure_read_u32(source, 0) != RANK_CONFIGURE_MAGIC {
    raise Invalid(Magic)
  }
  if configure_read_u32(source, 4) != RANK_CONFIGURE_VERSION {
    raise Invalid(Version)
  }
  for index in 72..<RANK_CONFIGURE_DIGEST_OFFSET {
    if source[index] != b'\x00' {
      raise Invalid(Reserved)
    }
  }
  if configure_read_u32(source, RANK_CONFIGURE_CHECKSUM_OFFSET) !=
    configure_checksum(source, 0, length, RANK_CONFIGURE_CHECKSUM_OFFSET) {
    raise Invalid(Checksum)
  }
  if configure_read_raw_digest(source, RANK_CONFIGURE_DIGEST_OFFSET) !=
    rank_configure_digest(source, length) {
    raise Invalid(Digest)
  }
  if configure_read_i32(source, 68) != 0 && configure_read_i32(source, 68) != 1 {
    raise Invalid(Reserved)
  }
}

///|
pub fn decode_rank_configure(
  source : FixedArray[Byte],
  length : Int,
  expected_binding : @rank_group_wire.RankGroupWireBinding,
) -> TensorParallelRankConfigureContract raise TensorParallelRankConfigureError {
  validate_rank_configure_header(source, length)
  let startup_length = configure_read_i32(source, 16)
  let source_length = configure_read_i32(source, 20)
  let topology_length = configure_read_i32(source, 24)
  let envelope_length = configure_read_i32(source, 28)
  let startup_capacity = @worker_wire.StartupFrameBuffer::new().capacity()
  let envelope_capacity = @tensor_parallel_worker_bootstrap.tensor_parallel_rank_envelope_frame_bytes()
  if startup_length != startup_capacity ||
    source_length <= 0 ||
    source_length > @worker_wire.bootstrap_source_capacity() ||
    topology_length < SELECTED_TOPOLOGY_HEADER_BYTES ||
    topology_length > SELECTED_TOPOLOGY_MAX_BYTES ||
    envelope_length != envelope_capacity ||
    RANK_CONFIGURE_HEADER_BYTES +
    startup_length +
    source_length +
    topology_length +
    envelope_length !=
    length {
    raise Invalid(Length)
  }
  let startup_offset = RANK_CONFIGURE_HEADER_BYTES
  let startup_wire = FixedArray::make(startup_length, b'\x00')
  for index in 0..<startup_length {
    startup_wire[index] = source[startup_offset + index]
  }
  let startup_frame = @worker_wire.StartupFrameBuffer::new().load(
    startup_wire, startup_length,
  ) catch {
    _ => raise Invalid(Startup)
  }
  let startup_kind = startup_frame.kind() catch { _ => raise Invalid(Startup) }
  if startup_kind != Configure {
    raise Invalid(Startup)
  }
  let startup = startup_frame.contract() catch { _ => raise Invalid(Startup) }
  let source_offset = startup_offset + startup_length
  let source_wire = FixedArray::make(source_length, b'\x00')
  for index in 0..<source_length {
    source_wire[index] = source[source_offset + index]
  }
  let bootstrap_source = @worker_wire.decode_bootstrap_source(
    source_wire, source_length,
  ) catch {
    _ => raise Invalid(BootstrapSource)
  }
  let topology_offset = source_offset + source_length
  let topology = decode_selected_topology(
    source, topology_offset, topology_length,
  )
  let envelope_offset = topology_offset + topology_length
  let envelope_wire = FixedArray::make(envelope_length, b'\x00')
  for index in 0..<envelope_length {
    envelope_wire[index] = source[envelope_offset + index]
  }
  let envelope_buffer = @tensor_parallel_worker_bootstrap.TensorParallelRankEnvelopeFrameBuffer::new()
  let envelope_frame = envelope_buffer.load(envelope_wire, envelope_length) catch {
    _ => raise Invalid(RankEnvelope)
  }
  let envelope = envelope_frame.envelope() catch {
    _ => raise Invalid(RankEnvelope)
  }
  validate_configure_claims(
    expected_binding, startup, bootstrap_source, topology, envelope,
  )
  if configure_read_u64(source, 32) != startup.model_generation_value() ||
    configure_read_u64(source, 40) != envelope.group_generation_value() ||
    configure_read_u64(source, 48) != startup.predecessor_value() ||
    configure_read_i32(source, 56) != envelope.rank() ||
    configure_read_i32(source, 60) != envelope.world_size() ||
    configure_read_i32(source, 64) != envelope.device_ordinal() ||
    (configure_read_i32(source, 68) == 1) != startup.has_predecessor() {
    raise Invalid(Binding)
  }
  let canonical = compose_rank_configure(
    expected_binding, startup, bootstrap_source, topology, envelope,
  )
  if canonical.length != length {
    raise Invalid(Length)
  }
  for index in 0..<length {
    if canonical.storage[index] != source[index] {
      raise Invalid(Binding)
    }
  }
  canonical
}

///|
pub fn TensorParallelRankConfigureContract::copy_to(
  self : TensorParallelRankConfigureContract,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise TensorParallelRankConfigureError {
  if destination_offset < 0 ||
    destination_offset > destination.length() - self.length {
    raise Invalid(Destination)
  }
  for index in 0..<self.length {
    destination[destination_offset + index] = self.storage[index]
  }
}
