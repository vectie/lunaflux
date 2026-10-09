///|
pub fn DeepSeekV4HostManifest::identity(self : Self) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn DeepSeekV4HostManifest::digest(
  self : Self,
) -> DeepSeekV4HostManifestDigest {
  self.digest
}

///|
pub fn DeepSeekV4HostManifest::expected_digests(
  self : Self,
) -> DeepSeekV4HostExpectedDigests {
  self.expected_digests
}

///|
pub fn DeepSeekV4HostManifest::tensor_count(self : Self) -> Int {
  self.tensors.length()
}

///|
pub fn DeepSeekV4HostManifest::arena_count(self : Self) -> Int {
  self.arena_sizes.length()
}

///|
pub fn DeepSeekV4HostManifest::arena_size(self : Self, index : Int) -> Int? {
  self.arena_sizes.get(index)
}

///|
pub fn DeepSeekV4HostManifest::total_tensor_bytes(self : Self) -> Int64 {
  self.total_tensor_bytes
}

///|
pub fn DeepSeekV4HostManifest::total_arena_bytes(self : Self) -> Int64 {
  self.total_arena_bytes
}

///|
pub fn DeepSeekV4HostTensor::binding(
  self : Self,
) -> @deepseek_v4_weights.DeepSeekV4TensorBinding {
  self.binding
}

///|
pub fn DeepSeekV4HostTensor::arena_ordinal(self : Self) -> Int {
  self.arena_ordinal
}

///|
pub fn DeepSeekV4HostTensor::arena_offset(self : Self) -> Int {
  self.arena_offset
}

///|
pub fn DeepSeekV4HostTensor::byte_count(self : Self) -> Int {
  self.byte_count
}

///|
pub fn DeepSeekV4HostTensor::quantized_parameter_index(self : Self) -> Int? {
  self.quantized_parameter_index
}

///|
pub fn DeepSeekV4HostWeights::identity(self : Self) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn DeepSeekV4HostWeights::manifest_digest(
  self : Self,
) -> DeepSeekV4HostManifestDigest {
  self.manifest_digest
}

///|
pub fn DeepSeekV4HostWeights::expected_digests(
  self : Self,
) -> DeepSeekV4HostExpectedDigests {
  self.expected_digests
}

///|
pub fn DeepSeekV4HostWeights::tensor_count(self : Self) -> Int {
  self.tensors.length()
}

///|
pub fn DeepSeekV4HostWeights::arena_count(self : Self) -> Int {
  self.arenas.length()
}

///|
/// Report the exact length of one live host arena without exposing its bytes.
pub fn DeepSeekV4HostWeights::device_arena_size(
  self : Self,
  arena_ordinal : Int,
) -> Int raise @segmented_host_borrow.SegmentedHostBorrowError {
  if self.released || arena_ordinal < 0 || arena_ordinal >= self.arenas.length() {
    raise SourceUnavailable(arena_ordinal)
  }
  guard self.arenas[arena_ordinal] is Some(arena) else {
    raise SourceUnavailable(arena_ordinal)
  }
  arena.length()
}

///|
/// Synchronously lend one arena to the family-neutral device copier. The
/// consumer cannot retain or reuse the borrow after this call returns.
pub fn DeepSeekV4HostWeights::provide_device_arena(
  self : Self,
  arena_ordinal : Int,
  consumer : @segmented_host_borrow.SegmentedHostArenaConsumer,
) -> Unit raise @segmented_host_borrow.SegmentedHostBorrowError {
  if self.released || arena_ordinal < 0 || arena_ordinal >= self.arenas.length() {
    raise SourceUnavailable(arena_ordinal)
  }
  guard self.arenas[arena_ordinal] is Some(arena) else {
    raise SourceUnavailable(arena_ordinal)
  }
  consumer.accept(arena)
}

///|
pub fn DeepSeekV4HostWeights::total_tensor_bytes(self : Self) -> Int64 {
  self.total_tensor_bytes
}

///|
pub fn DeepSeekV4HostWeights::total_arena_bytes(self : Self) -> Int64 {
  self.total_arena_bytes
}

///|
pub fn DeepSeekV4HostWeights::is_released(self : Self) -> Bool {
  self.released
}

///|
pub fn DeepSeekV4HostWeights::tensor(
  self : Self,
  index : Int,
) -> DeepSeekV4HostTensor raise DeepSeekV4HostMaterializationError {
  if self.released {
    raise ArenaReleased
  }
  match self.tensors.get(index) {
    Some(tensor) => tensor
    None => raise TensorIndexOutOfBounds(index)
  }
}

///|
pub fn DeepSeekV4HostWeights::tensor_byte(
  self : Self,
  tensor_index : Int,
  byte_offset : Int,
) -> Byte raise DeepSeekV4HostMaterializationError {
  let tensor = self.tensor(tensor_index)
  if byte_offset < 0 || byte_offset >= tensor.byte_count {
    raise TensorByteOutOfBounds(tensor_index)
  }
  guard self.arenas[tensor.arena_ordinal] is Some(arena) else {
    raise ArenaReleased
  }
  arena[tensor.arena_offset + byte_offset]
}
