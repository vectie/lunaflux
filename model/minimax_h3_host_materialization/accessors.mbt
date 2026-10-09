///|
pub fn MiniMaxH3HostComponentManifest::component(
  self : Self,
) -> @minimax_h3_weights.MiniMaxH3WeightComponent {
  self.component
}

///|
pub fn MiniMaxH3HostComponentManifest::identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn MiniMaxH3HostComponentManifest::digest(
  self : Self,
) -> MiniMaxH3HostManifestDigest {
  self.digest
}

///|
pub fn MiniMaxH3HostComponentManifest::tensor_count(self : Self) -> Int {
  self.tensors.length()
}

///|
/// Pure tensor metadata for device packing, without allocating host payloads.
pub fn MiniMaxH3HostComponentManifest::tensor(
  self : Self,
  index : Int,
) -> MiniMaxH3HostTensor raise MiniMaxH3HostMaterializationError {
  guard self.tensors.get(index) is Some(entry) else {
    raise TensorIndexOutsideArena(index)
  }
  {
    name: entry.tensor.name(),
    role: entry.role,
    shape: entry.tensor.shape(),
    arena_index: entry.arena_index,
    arena_offset: entry.arena_offset,
    byte_length: entry.tensor.byte_length(),
  }
}

///|
pub fn MiniMaxH3HostComponentManifest::shard_count(self : Self) -> Int {
  self.shards.length()
}

///|
pub fn MiniMaxH3HostComponentManifest::arena_count(self : Self) -> Int {
  self.arena_sizes.length()
}

///|
pub fn MiniMaxH3HostComponentManifest::arena_size(
  self : Self,
  index : Int,
) -> Int64? {
  self.arena_sizes.get(index)
}

///|
pub fn MiniMaxH3HostComponentManifest::total_tensor_bytes(self : Self) -> Int64 {
  self.total_tensor_bytes
}

///|
pub fn MiniMaxH3HostTensor::name(self : Self) -> String {
  self.name
}

///|
pub fn MiniMaxH3HostTensor::role(
  self : Self,
) -> @minimax_h3_weights.MiniMaxH3TensorRole {
  self.role
}

///|
pub fn MiniMaxH3HostTensor::shape(self : Self) -> ReadOnlyArray[Int64] {
  self.shape
}

///|
pub fn MiniMaxH3HostTensor::arena_index(self : Self) -> Int {
  self.arena_index
}

///|
pub fn MiniMaxH3HostTensor::arena_offset(self : Self) -> Int64 {
  self.arena_offset
}

///|
pub fn MiniMaxH3HostTensor::byte_length(self : Self) -> Int64 {
  self.byte_length
}

///|
pub fn MiniMaxH3HostComponentArena::component(
  self : Self,
) -> @minimax_h3_weights.MiniMaxH3WeightComponent {
  self.component
}

///|
pub fn MiniMaxH3HostComponentArena::identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn MiniMaxH3HostComponentArena::manifest_digest(
  self : Self,
) -> MiniMaxH3HostManifestDigest {
  self.manifest_digest
}

///|
pub fn MiniMaxH3HostComponentArena::tensor_count(self : Self) -> Int {
  self.tensors.length()
}

///|
pub fn MiniMaxH3HostComponentArena::tensor(
  self : Self,
  index : Int,
) -> MiniMaxH3HostTensor raise MiniMaxH3HostMaterializationError {
  if self.released {
    raise ArenaReleased
  }
  match self.tensors.get(index) {
    Some(tensor) => tensor
    None => raise TensorIndexOutsideArena(index)
  }
}

///|
pub fn MiniMaxH3HostComponentArena::arena_count(self : Self) -> Int {
  self.arenas.length()
}

///|
/// Report the exact length of one live host arena without exposing its bytes.
pub fn MiniMaxH3HostComponentArena::device_arena_size(
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
pub fn MiniMaxH3HostComponentArena::provide_device_arena(
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
pub fn MiniMaxH3HostComponentArena::total_tensor_bytes(self : Self) -> Int64 {
  self.total_tensor_bytes
}

///|
pub fn MiniMaxH3HostComponentArena::is_released(self : Self) -> Bool {
  self.released
}

///|
pub fn MiniMaxH3HostComponentArena::tensor_byte(
  self : Self,
  tensor_index : Int,
  byte_offset : Int64,
) -> Byte raise MiniMaxH3HostMaterializationError {
  let tensor = self.tensor(tensor_index)
  if byte_offset < 0L || byte_offset >= tensor.byte_length {
    raise TensorByteOutsideArena(tensor_index)
  }
  guard self.arenas[tensor.arena_index] is Some(arena) else {
    raise ArenaReleased
  }
  arena[(tensor.arena_offset + byte_offset).to_int()]
}

///|
/// Zeroizes every arena, drops all owner references, and invalidates access.
/// MoonBit storage remains GC-managed; this makes no operating-system release
/// timing claim.
pub fn MiniMaxH3HostComponentArena::release(self : Self) -> Bool {
  if self.released {
    return false
  }
  zero_and_drop(self.arenas)
  self.released = true
  true
}
