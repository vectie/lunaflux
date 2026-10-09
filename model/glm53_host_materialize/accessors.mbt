///|
pub fn Glm53HostWeights::execution_identity(self : Self) -> @spec.ModelIdentity {
  self.execution_identity
}

///|
pub fn Glm53HostWeights::manifest_layout_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.manifest_layout
}

///|
pub fn Glm53HostWeights::numeric_schema_digest(
  self : Self,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.numeric_schema
}

///|
pub fn Glm53HostWeights::numeric_binding_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.numeric_binding
}

///|
pub fn Glm53HostWeights::source_digest(self : Self) -> Glm53HostSourceDigest {
  self.source_digest
}

///|
pub fn Glm53HostSourceDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn Glm53HostWeights::tensor_count(self : Self) -> Int {
  self.regions.length()
}

///|
pub fn Glm53HostWeights::arena_count(self : Self) -> Int {
  self.arenas.length()
}

///|
/// Report the exact length of one live host arena without exposing its bytes.
pub fn Glm53HostWeights::device_arena_size(
  self : Self,
  arena_ordinal : Int,
) -> Int raise @segmented_host_borrow.SegmentedHostBorrowError {
  if !self.open || arena_ordinal < 0 || arena_ordinal >= self.arenas.length() {
    raise SourceUnavailable(arena_ordinal)
  }
  self.arenas[arena_ordinal].bytes.length()
}

///|
/// Synchronously lend one arena to the family-neutral device copier. The
/// consumer cannot retain or reuse the borrow after this call returns.
pub fn Glm53HostWeights::provide_device_arena(
  self : Self,
  arena_ordinal : Int,
  consumer : @segmented_host_borrow.SegmentedHostArenaConsumer,
) -> Unit raise @segmented_host_borrow.SegmentedHostBorrowError {
  if !self.open || arena_ordinal < 0 || arena_ordinal >= self.arenas.length() {
    raise SourceUnavailable(arena_ordinal)
  }
  consumer.accept(self.arenas[arena_ordinal].bytes)
}

///|
pub fn Glm53HostWeights::total_bytes(self : Self) -> Int64 {
  self.total_bytes
}

///|
pub fn Glm53HostWeights::is_closed(self : Self) -> Bool {
  !self.open
}

///|
pub fn Glm53HostWeights::tensor_region(
  self : Self,
  ordinal : Int,
) -> Glm53HostTensorRegion raise Glm53HostMaterializeError {
  if !self.open {
    raise Closed
  }
  match self.regions.get(ordinal) {
    Some(region) => region
    None => raise TensorOrdinalOutOfRange
  }
}

///|
pub fn Glm53HostWeights::tensor_byte(
  self : Self,
  ordinal : Int,
  byte_offset : Int64,
) -> Byte raise Glm53HostMaterializeError {
  let region = self.tensor_region(ordinal)
  if byte_offset < 0L || byte_offset >= region.byte_length {
    raise TensorByteOutOfRange
  }
  let mut remaining = byte_offset
  for span in region.spans {
    if remaining < span.byte_length.to_int64() {
      return self.arenas[span.arena_index].bytes[span.offset_bytes +
        remaining.to_int()]
    }
    remaining -= span.byte_length.to_int64()
  }
  raise TensorByteOutOfRange
}

///|
/// Deterministically zeroizes every final arena, invalidates all access, and
/// drops the owner references. MoonBit FixedArray storage remains GC-managed;
/// this API does not claim deterministic operating-system deallocation.
pub fn Glm53HostWeights::close(self : Self) -> Unit {
  if !self.open {
    return
  }
  zero_arenas(self.arenas)
  for index in 0..<self.arenas.length() {
    self.arenas[index].bytes = FixedArray::make(0, b'\x00')
  }
  self.open = false
}
