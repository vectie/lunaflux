///|
const FRAME_BYTES : Int = 1112

///|
const GROUP_DIGEST_OFFSET : Int = 192

///|
const WORKER_OFFSET : Int = 640

///|
const WORKER_BYTES : Int = 408

///|
const CHECKSUM_OFFSET : Int = 1048

///|
fn put_u32(bytes : FixedArray[Byte], offset : Int, value : UInt) -> Unit {
  for index in 0..<4 {
    bytes[offset + index] = (value >> (index * 8)).to_byte()
  }
}

///|
fn put_u64(bytes : FixedArray[Byte], offset : Int, value : UInt64) -> Unit {
  for index in 0..<8 {
    bytes[offset + index] = (value >> (index * 8)).to_byte()
  }
}

///|
fn put_text(bytes : FixedArray[Byte], offset : Int, value : String) -> Unit {
  for index, byte in @utf8.encode(value) {
    bytes[offset + index] = byte
  }
}

///|
fn seal(bytes : FixedArray[Byte]) -> Unit {
  let hash = @canonical_sha256.CanonicalSha256::new()
  ignore(hash.update_bytes(b"lunaflux.remote-rank-configure.v1"))
  for index in 0..<CHECKSUM_OFFSET {
    ignore(hash.update_byte(bytes[index]))
  }
  put_text(bytes, CHECKSUM_OFFSET, hash.finalize_hex())
}

///|
fn encode_rank(
  plan : @two_node_plan.TwoNodePlan,
  worker : @worker_wire.WorkerStartupContract,
  executable : @spec.ContentDigest,
  deployment : @spec.ContentDigest,
  generation : UInt64,
  rendezvous : @device.TensorCollectiveGroupId,
  lease_millis : UInt64,
  rank : Int,
) -> FixedArray[Byte] raise RemoteStartupError {
  let bytes = FixedArray::make(FRAME_BYTES, b'\x00')
  guard plan.topology().rank(rank) is Some(node) else { raise Rejected(Count) }
  put_u32(bytes, 0, 0x5253464cU)
  put_u32(bytes, 4, 1U)
  put_u32(bytes, 8, FRAME_BYTES.reinterpret_as_uint())
  put_u32(bytes, 12, rank.reinterpret_as_uint())
  put_u64(bytes, 16, generation)
  put_u32(bytes, 24, node.node().label().length().reinterpret_as_uint())
  put_u32(bytes, 28, node.rdma_interface().length().reinterpret_as_uint())
  put_u32(bytes, 32, 2U)
  put_u32(bytes, 36, WORKER_BYTES.reinterpret_as_uint())
  put_u32(bytes, 40, 1U) // Explicit RoCE profile; no automatic socket fallback.
  put_u64(bytes, 48, lease_millis)
  put_text(bytes, 64, plan.digest().as_sha256())
  put_text(bytes, 128, deployment.as_hex())
  put_text(bytes, 256, executable.as_hex())
  put_text(bytes, 320, node.node().label())
  put_text(bytes, 448, node.rdma_interface())
  for index, byte in rendezvous.to_bytes() {
    bytes[512 + index] = byte
  }
  let owner = @worker_wire.StartupFrameBuffer::new()
  let frame = owner.encode(Configure, worker) catch {
    _ => raise Rejected(WorkerSchema)
  }
  let length = frame.length() catch { _ => raise Rejected(WorkerSchema) }
  if length != WORKER_BYTES {
    raise Rejected(WorkerSchema)
  }
  frame.copy_to(bytes, destination_offset=WORKER_OFFSET) catch {
    _ => raise Rejected(WorkerSchema)
  }
  bytes
}

///|
pub fn remote_startup_frame_bytes() -> Int {
  FRAME_BYTES
}

///|
/// Copies the complete immutable startup envelope into caller-owned storage.
/// Capacity is checked before writing; an oversized destination tail is intact.
pub fn RemoteRankStartup::copy_to(
  self : RemoteRankStartup,
  destination : FixedArray[Byte],
) -> Int raise RemoteStartupError {
  if destination.length() < FRAME_BYTES {
    raise Rejected(Destination)
  }
  for index in 0..<FRAME_BYTES {
    destination[index] = self.frame[index]
  }
  FRAME_BYTES
}

///|
/// Compare a complete message against an independently composed expectation.
/// Equality includes the version, reserved bytes, node/NIC, both-rank group
/// binding, rendezvous, worker limits, predecessor and source/executable pins.
/// This is not peer authentication and must run only during startup; callers
/// still need an authenticated control channel and local device admission.
pub fn RemoteRankStartup::verify_configure(
  self : RemoteRankStartup,
  source : FixedArray[Byte],
  length : Int,
) -> Unit raise RemoteStartupError {
  if length != FRAME_BYTES || source.length() < FRAME_BYTES {
    raise Rejected(FrameLength)
  }
  for index in 0..<FRAME_BYTES {
    if source[index] != self.frame[index] {
      raise Rejected(FrameMismatch)
    }
  }
}
