///|
let startup_magic : UInt = 0x5358464c

///|
let startup_version : UInt = 4

///|
let startup_length : Int = 408

///|
let startup_source_digest_offset : Int = 344

///|
let startup_checksum_offset : Int = 336

///|
/// V4 preserves every v3 field offset and appends the source digest. The
/// checksum therefore covers the appended tail while excluding only its own
/// four bytes at the retained v3 offset.
fn startup_checksum(bytes : FixedArray[Byte]) -> UInt {
  let mut hash : UInt = 2166136261
  for index in 0..<read_i32(bytes, 16) {
    if index < startup_checksum_offset || index >= startup_checksum_offset + 4 {
      hash = (hash ^ bytes[index].to_uint()) * 16777619
    }
  }
  hash
}

///|
fn write_digest(
  bytes : FixedArray[Byte],
  offset : Int,
  value : String,
) -> Unit raise WorkerWireError {
  let encoded = @utf8.encode(value)
  if encoded.length() != 64 {
    raise InvalidFrame(StartupIdentity)
  }
  for index in 0..<64 {
    bytes[offset + index] = encoded[index]
  }
}

///|
fn read_digest(
  bytes : FixedArray[Byte],
  offset : Int,
) -> String raise WorkerWireError {
  let encoded = Bytes::makei(64, index => bytes[offset + index])
  @utf8.decode(encoded) catch {
    _ => raise InvalidFrame(StartupIdentity)
  }
}

///|
fn write_worker_limits(
  bytes : FixedArray[Byte],
  limits : @worker_protocol.WorkerProtocolLimits,
) -> Unit {
  for
    index, value in [
      limits.max_prefill_rows(),
      limits.max_decode_rows(),
      limits.max_plan_rows(),
      limits.max_plan_tokens(),
      limits.max_plan_pages(),
      limits.max_capabilities(),
      limits.max_completion_slots(),
      limits.max_sequence_tokens(),
      limits.max_token_id(),
    ] {
    write_i32(bytes, 168 + index * 4, value)
  }
}

///|
fn read_worker_limits(
  bytes : FixedArray[Byte],
) -> @worker_protocol.WorkerProtocolLimits raise WorkerWireError {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=read_i32(bytes, 168),
    max_decode_rows=read_i32(bytes, 172),
    max_plan_rows=read_i32(bytes, 176),
    max_plan_tokens=read_i32(bytes, 180),
    max_plan_pages=read_i32(bytes, 184),
    max_capabilities=read_i32(bytes, 188),
    max_completion_slots=read_i32(bytes, 192),
    max_sequence_tokens=read_i32(bytes, 196),
    max_token_id=read_i32(bytes, 200),
  ) catch {
    error => raise Protocol(error)
  }
}

///|
fn write_inference_limits(
  bytes : FixedArray[Byte],
  limits : @inference.InferenceLimits,
) -> Unit {
  for
    index, value in [
      limits.max_text_bytes(),
      limits.max_input_tokens(),
      limits.max_new_tokens(),
      limits.max_context_tokens(),
      limits.max_token_id(),
      limits.max_stop_token_ids(),
      limits.max_stop_strings(),
      limits.max_stop_string_bytes(),
      limits.max_trace_bytes(),
      limits.max_cache_scope_bytes(),
      limits.max_decoded_delta_bytes(),
    ] {
    write_i32(bytes, 204 + index * 4, value)
  }
  write_u64(bytes, 248, limits.max_deadline_millis())
  write_i32(bytes, 256, limits.max_top_k())
  write_u32(bytes, 260, 0)
  write_u64(bytes, 264, limits.max_temperature().reinterpret_as_uint64())
}

///|
fn read_inference_limits(
  bytes : FixedArray[Byte],
) -> @inference.InferenceLimits raise WorkerWireError {
  @inference.InferenceLimits::new(
    max_text_bytes=read_i32(bytes, 204),
    max_input_tokens=read_i32(bytes, 208),
    max_new_tokens=read_i32(bytes, 212),
    max_context_tokens=read_i32(bytes, 216),
    max_token_id=read_i32(bytes, 220),
    max_stop_token_ids=read_i32(bytes, 224),
    max_stop_strings=read_i32(bytes, 228),
    max_stop_string_bytes=read_i32(bytes, 232),
    max_trace_bytes=read_i32(bytes, 236),
    max_cache_scope_bytes=read_i32(bytes, 240),
    max_decoded_delta_bytes=read_i32(bytes, 244),
    max_deadline_millis=read_u64(bytes, 248),
    max_top_k=read_i32(bytes, 256),
    max_temperature=read_u64(bytes, 264).reinterpret_as_double(),
  ) catch {
    _ => raise InvalidFrame(Counts)
  }
}

///|
fn write_startup_contract(
  bytes : FixedArray[Byte],
  kind : StartupFrameKind,
  contract : WorkerStartupContract,
) -> Unit raise WorkerWireError {
  let length = if contract.streaming is Some(_) { 1024 } else { startup_length }
  if bytes.length() < length {
    raise DestinationTooSmall
  }
  for index in 0..<length {
    bytes[index] = b'\x00'
  }
  write_u32(bytes, 0, startup_magic)
  write_u32(
    bytes,
    4,
    if contract.streaming is Some(_) {
      5
    } else {
      startup_version
    },
  )
  write_i32(bytes, 8, @inference.ProtocolVersion::v1().wire_code())
  write_i32(bytes, 12, if kind == Configure { 0 } else { 1 })
  write_i32(bytes, 16, length)
  write_i32(bytes, 20, if contract.has_predecessor { 1 } else { 0 })
  write_u64(bytes, 24, contract.model_generation_value)
  write_u64(bytes, 32, contract.predecessor_value)
  write_digest(bytes, 40, contract.identity.content().as_hex())
  write_digest(bytes, 104, contract.identity.plan().as_hex())
  write_worker_limits(bytes, contract.worker_limits)
  write_inference_limits(bytes, contract.inference_limits)
  write_digest(bytes, 272, contract.bootstrap_digest.as_hex())
  write_i32(bytes, 340, contract.device_ordinal)
  write_digest(
    bytes,
    startup_source_digest_offset,
    contract.bootstrap_source_digest.as_hex(),
  )
  if contract.streaming is Some(capability) {
    write_i32(bytes, 408, capability.slots())
    write_i32(bytes, 412, capability.page_count())
    write_i32(bytes, 416, capability.reserve_pages())
    write_i32(bytes, 420, capability.transfer_timeout_millis())
    write_u64(bytes, 424, capability.host_bytes().reinterpret_as_uint64())
    write_u64(bytes, 432, capability.page_bytes().reinterpret_as_uint64())
    write_u64(bytes, 440, capability.restore_ns().reinterpret_as_uint64())
    write_u64(bytes, 448, capability.recompute_ns().reinterpret_as_uint64())
    write_u64(bytes, 456, contract.streaming_epoch)
    write_i32(bytes, 464, capability.shape_count())
    write_u64(bytes, 472, capability.workspace_budget().reinterpret_as_uint64())
    for i in 0..<capability.shape_count() {
      let shape = capability.shape_at(i)
      write_i32(bytes, 480 + i * 16, shape.tokens())
      write_u64(
        bytes,
        488 + i * 16,
        shape.workspace_bytes().reinterpret_as_uint64(),
      )
    }
  }
  write_u32(bytes, startup_checksum_offset, startup_checksum(bytes))
}

///|
fn validate_startup_source(
  bytes : FixedArray[Byte],
  length : Int,
) -> Unit raise WorkerWireError {
  if (length != startup_length && length != 1024) || bytes.length() < length {
    raise InvalidFrame(Length)
  }
  if read_u32(bytes, 0) != startup_magic {
    raise InvalidFrame(Magic)
  }
  if read_u32(bytes, 4) != (if length == 1024 { 5 } else { startup_version }) ||
    read_i32(bytes, 8) != @inference.ProtocolVersion::v1().wire_code() {
    raise InvalidFrame(Version)
  }
  if (read_i32(bytes, 12) != 0 && read_i32(bytes, 12) != 1) ||
    (read_i32(bytes, 20) != 0 && read_i32(bytes, 20) != 1) {
    raise InvalidFrame(Kind)
  }
  if read_i32(bytes, 16) != length || read_u32(bytes, 260) != 0 {
    raise InvalidFrame(Reserved)
  }
  if length == 1024 {
    let shapes = read_i32(bytes, 464)
    if shapes < 0 || shapes > 32 {
      raise InvalidFrame(Counts)
    }
    if read_u32(bytes, 468) != 0 {
      raise InvalidFrame(Reserved)
    }
    for i in 0..<shapes {
      if read_u32(bytes, 484 + i * 16) != 0 {
        raise InvalidFrame(Reserved)
      }
    }
    for i in (480 + shapes * 16)..<1024 {
      if bytes[i] != b'\x00' {
        raise InvalidFrame(Reserved)
      }
    }
  }
  if startup_checksum(bytes) != read_u32(bytes, startup_checksum_offset) {
    raise InvalidFrame(Checksum)
  }
}

///|
pub fn StartupFrameBuffer::encode(
  self : StartupFrameBuffer,
  kind : StartupFrameKind,
  contract : WorkerStartupContract,
) -> ValidatedStartupFrame raise WorkerWireError {
  if self.epoch == 0xffffffffffffffffUL {
    raise FrameEpochOverflow
  }
  write_startup_contract(self.storage, kind, contract)
  self.epoch += 1UL
  self.length = read_i32(self.storage, 16)
  ValidatedStartupFrame(self, self.epoch)
}

///|
pub fn StartupFrameBuffer::load(
  self : StartupFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
) -> ValidatedStartupFrame raise WorkerWireError {
  if self.epoch == 0xffffffffffffffffUL {
    raise FrameEpochOverflow
  }
  validate_startup_source(source, length)
  if self.storage.length() < length {
    raise DestinationTooSmall
  }
  for index in 0..<length {
    self.storage[index] = source[index]
  }
  self.epoch += 1UL
  self.length = length
  ValidatedStartupFrame(self, self.epoch)
}

///|
fn ValidatedStartupFrame::owner(
  self : ValidatedStartupFrame,
) -> StartupFrameBuffer raise WorkerWireError {
  if self.1 == 0UL || self.0.epoch != self.1 || self.0.length == 0 {
    raise StaleFrame
  }
  self.0
}

///|
pub fn ValidatedStartupFrame::kind(
  self : ValidatedStartupFrame,
) -> StartupFrameKind raise WorkerWireError {
  if read_i32(self.owner().storage, 12) == 0 {
    Configure
  } else {
    Ready
  }
}

///|
pub fn ValidatedStartupFrame::contract(
  self : ValidatedStartupFrame,
) -> WorkerStartupContract raise WorkerWireError {
  let bytes = self.owner().storage
  // Decoded identity is an untrusted claim until exact admission against a
  // rebuilt ModelPlan identity.
  let content = @spec.ContentDigest::from_sha256(read_digest(bytes, 40)) catch {
    _ => raise InvalidFrame(StartupIdentity)
  }
  let plan = @spec.PlanDigest::from_sha256(read_digest(bytes, 104)) catch {
    _ => raise InvalidFrame(StartupIdentity)
  }
  let identity = @spec.ModelIdentity::new(content, plan) catch {
    _ => raise InvalidFrame(StartupIdentity)
  }
  WorkerStartupContract::new(
    streaming_epoch=if self.owner().length == 1024 {
      read_u64(bytes, 456)
    } else {
      0UL
    },
    streaming?=if self.owner().length == 1024 {
      let shapes : Array[@streaming_ir.Shape] = []
      for i in 0..<read_i32(bytes, 464) {
        shapes.push(
          @streaming_ir.Shape::new(
            tokens=read_i32(bytes, 480 + i * 16),
            workspace_bytes=read_u64(bytes, 488 + i * 16).reinterpret_as_int64(),
          ) catch {
            _ => raise InvalidFrame(StartupIdentity)
          },
        )
      }
      Some(
        @streaming_ir.Capability::new(
          shapes=shapes[:],
          workspace_budget=read_u64(bytes, 472).reinterpret_as_int64(),
          transfer_timeout_millis=read_i32(bytes, 420),
          slots=read_i32(bytes, 408),
          page_count=read_i32(bytes, 412),
          reserve_pages=read_i32(bytes, 416),
          host_bytes=read_u64(bytes, 424).reinterpret_as_int64(),
          page_bytes=read_u64(bytes, 432).reinterpret_as_int64(),
          restore_ns=read_u64(bytes, 440).reinterpret_as_int64(),
          recompute_ns=read_u64(bytes, 448).reinterpret_as_int64(),
          separate_host_memory=true,
        ) catch {
          _ => raise InvalidFrame(StartupIdentity)
        },
      )
    } else {
      None
    },
    identity~,
    bootstrap_digest=WorkerBootstrapDigest::from_sha256(read_digest(bytes, 272)),
    bootstrap_source_digest=WorkerBootstrapSourceDigest::from_sha256(
      read_digest(bytes, startup_source_digest_offset),
    ) catch {
      _ => raise InvalidFrame(StartupIdentity)
    },
    device_ordinal=read_i32(bytes, 340),
    model_generation_value=read_u64(bytes, 24),
    worker_limits=read_worker_limits(bytes),
    inference_limits=read_inference_limits(bytes),
    has_predecessor=read_i32(bytes, 20) == 1,
    predecessor_value=read_u64(bytes, 32),
  )
}

///|
pub fn ValidatedStartupFrame::length(
  self : ValidatedStartupFrame,
) -> Int raise WorkerWireError {
  self.owner().length
}

///|
pub fn ValidatedStartupFrame::copy_to(
  self : ValidatedStartupFrame,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise WorkerWireError {
  let owner = self.owner()
  if destination_offset < 0 ||
    destination_offset > destination.length() - owner.length {
    raise DestinationTooSmall
  }
  for index in 0..<owner.length {
    destination[destination_offset + index] = owner.storage[index]
  }
}
