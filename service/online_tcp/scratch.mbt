///|
#warnings("-unused_value")
fn LunaOnlineTcpOutputScratch::new(
  capacity : Int,
) -> LunaOnlineTcpOutputScratch raise LunaOnlineTcpScratchError {
  if capacity <= 0 || capacity > LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES {
    raise LunaOnlineTcpInvalidScratchCapacity
  }
  let immutable = Bytes::makei(capacity, _ => b'\x00')
  let mutable = @buffer_alias.retain_bytes_as_fixed_array(immutable)
  if mutable.length() != capacity {
    raise LunaOnlineTcpInvalidScratchCapacity
  }
  {
    immutable,
    mutable,
    capacity,
    generation: 0UL,
    phase: LUNA_ONLINE_TCP_SCRATCH_IDLE,
    length: 0,
  }
}

///|
#warnings("-unused_value")
fn LunaOnlineTcpOutputScratch::begin_write(
  self : LunaOnlineTcpOutputScratch,
) -> LunaOnlineTcpOutputWrite raise LunaOnlineTcpScratchError {
  if self.phase != LUNA_ONLINE_TCP_SCRATCH_IDLE {
    raise LunaOnlineTcpScratchBusy
  }
  if self.generation == 0xffffffffffffffffUL {
    raise LunaOnlineTcpScratchEpochExhausted
  }
  self.generation += 1UL
  self.phase = LUNA_ONLINE_TCP_SCRATCH_MUTATING
  self.length = 0
  { scratch: self, generation: self.generation }
}

///|
#warnings("-unused_value")
fn LunaOnlineTcpOutputScratch::is_idle(
  self : LunaOnlineTcpOutputScratch,
) -> Bool {
  self.phase == LUNA_ONLINE_TCP_SCRATCH_IDLE
}

///|
#warnings("-unused_value")
fn LunaOnlineTcpOutputScratch::byte_capacity(
  self : LunaOnlineTcpOutputScratch,
) -> Int {
  self.capacity
}

///|
fn LunaOnlineTcpOutputWrite::require_mutating(
  self : LunaOnlineTcpOutputWrite,
) -> LunaOnlineTcpOutputScratch raise LunaOnlineTcpScratchError {
  if self.generation == 0UL ||
    self.scratch.generation != self.generation ||
    self.scratch.phase != LUNA_ONLINE_TCP_SCRATCH_MUTATING {
    raise LunaOnlineTcpScratchStale
  }
  self.scratch
}

///|
#warnings("-unused_value")
fn LunaOnlineTcpOutputWrite::write_byte(
  self : LunaOnlineTcpOutputWrite,
  index : Int,
  value : Byte,
) -> Unit raise LunaOnlineTcpScratchError {
  let scratch = self.require_mutating()
  if index < 0 || index >= scratch.capacity {
    raise LunaOnlineTcpScratchRange
  }
  scratch.mutable[index] = value
}

///|
#warnings("-unused_value")
fn LunaOnlineTcpOutputWrite::copy_from(
  self : LunaOnlineTcpOutputWrite,
  source : FixedArray[Byte],
  source_offset~ : Int,
  destination_offset~ : Int,
  length~ : Int,
) -> Unit raise LunaOnlineTcpScratchError {
  let scratch = self.require_mutating()
  if source_offset < 0 ||
    destination_offset < 0 ||
    length < 0 ||
    source_offset > source.length() - length ||
    destination_offset > scratch.capacity - length {
    raise LunaOnlineTcpScratchRange
  }
  for index in 0..<length {
    scratch.mutable[destination_offset + index] = source[source_offset + index]
  }
}

///|
#warnings("-unused_value")
fn LunaOnlineTcpOutputWrite::publish(
  self : LunaOnlineTcpOutputWrite,
  length : Int,
) -> LunaOnlineTcpOutputFlight raise LunaOnlineTcpScratchError {
  let scratch = self.require_mutating()
  if length <= 0 || length > scratch.capacity {
    raise LunaOnlineTcpScratchRange
  }
  scratch.length = length
  scratch.phase = LUNA_ONLINE_TCP_SCRATCH_IN_FLIGHT
  { scratch, generation: self.generation, length }
}

///|
#warnings("-unused_value")
fn LunaOnlineTcpOutputWrite::abort(
  self : LunaOnlineTcpOutputWrite,
) -> Unit raise LunaOnlineTcpScratchError {
  let scratch = self.require_mutating()
  scratch.length = 0
  scratch.phase = LUNA_ONLINE_TCP_SCRATCH_IDLE
}

///|
fn LunaOnlineTcpOutputFlight::require_in_flight(
  self : LunaOnlineTcpOutputFlight,
) -> LunaOnlineTcpOutputScratch raise LunaOnlineTcpScratchError {
  if self.generation == 0UL ||
    self.scratch.generation != self.generation ||
    self.scratch.phase != LUNA_ONLINE_TCP_SCRATCH_IN_FLIGHT ||
    self.scratch.length != self.length {
    raise LunaOnlineTcpScratchStale
  }
  self.scratch
}

///|
#warnings("-unused_value")
fn LunaOnlineTcpOutputFlight::byte_length(
  self : LunaOnlineTcpOutputFlight,
) -> Int raise LunaOnlineTcpScratchError {
  ignore(self.require_in_flight())
  self.length
}

///|
#warnings("-unused_value")
fn LunaOnlineTcpOutputFlight::byte_at(
  self : LunaOnlineTcpOutputFlight,
  index : Int,
) -> Byte raise LunaOnlineTcpScratchError {
  let scratch = self.require_in_flight()
  if index < 0 || index >= self.length {
    raise LunaOnlineTcpScratchRange
  }
  scratch.immutable[index]
}

///|
#warnings("-unused_value")
fn LunaOnlineTcpOutputFlight::release(
  self : LunaOnlineTcpOutputFlight,
) -> Unit raise LunaOnlineTcpScratchError {
  let scratch = self.require_in_flight()
  scratch.length = 0
  scratch.phase = LUNA_ONLINE_TCP_SCRATCH_IDLE
}
