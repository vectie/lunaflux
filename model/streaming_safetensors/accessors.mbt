///|
pub fn StreamingSafetensorsDTypeSpec::kind(
  self : Self,
) -> StreamingSafetensorsDTypeKind {
  self.kind
}

///|
pub fn StreamingSafetensorsDTypeSpec::tag(self : Self) -> String {
  self.tag
}

///|
pub fn StreamingSafetensorsDTypeSpec::bytes_per_element(self : Self) -> Int64 {
  self.bytes_per_element
}

///|
pub fn StreamingSafetensorsFileDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn StreamingSafetensorsTensorMetadata::name(self : Self) -> String {
  self.name
}

///|
pub fn StreamingSafetensorsTensorMetadata::dtype(
  self : Self,
) -> StreamingSafetensorsDTypeSpec {
  self.dtype
}

///|
pub fn StreamingSafetensorsTensorMetadata::shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.shape
}

///|
pub fn StreamingSafetensorsTensorMetadata::data_start(self : Self) -> Int64 {
  self.data_start
}

///|
pub fn StreamingSafetensorsTensorMetadata::data_end(self : Self) -> Int64 {
  self.data_end
}

///|
pub fn StreamingSafetensorsTensorMetadata::byte_count(self : Self) -> Int64 {
  self.data_end - self.data_start
}

///|
pub fn StreamingSafetensorsShardMetadata::file_byte_count(self : Self) -> Int64 {
  self.file_byte_count
}

///|
pub fn StreamingSafetensorsShardMetadata::payload_offset(self : Self) -> Int64 {
  self.payload_offset
}

///|
pub fn StreamingSafetensorsShardMetadata::payload_byte_count(
  self : Self,
) -> Int64 {
  self.payload_byte_count
}

///|
pub fn StreamingSafetensorsShardMetadata::file_digest(
  self : Self,
) -> StreamingSafetensorsFileDigest {
  self.file_digest
}

///|
pub fn StreamingSafetensorsShardMetadata::tensor_count(self : Self) -> Int {
  self.tensors.length()
}

///|
pub fn StreamingSafetensorsShardMetadata::tensor(
  self : Self,
  ordinal : Int,
) -> StreamingSafetensorsTensorMetadata? {
  self.tensors.get(ordinal)
}

///|
pub fn StreamingSafetensorsInspection::shard_count(self : Self) -> Int {
  self.shards.length()
}

///|
pub fn StreamingSafetensorsInspection::total_tensor_count(self : Self) -> Int {
  self.total_tensor_count
}

///|
pub fn StreamingSafetensorsInspection::shard(
  self : Self,
  ordinal : Int,
) -> StreamingSafetensorsShardMetadata? {
  self.shards.get(ordinal).map(shard => shard.metadata)
}

///|
pub fn StreamingSafetensorsVisitedRange::shard_ordinal(self : Self) -> Int {
  self.shard_ordinal
}

///|
pub fn StreamingSafetensorsVisitedRange::tensor_ordinal(self : Self) -> Int {
  self.tensor_ordinal
}

///|
pub fn StreamingSafetensorsVisitedRange::tensor_byte_offset(
  self : Self,
) -> Int64 {
  self.tensor_byte_offset
}

///|
pub fn StreamingSafetensorsVisitedRange::destination_offset(self : Self) -> Int {
  self.destination_offset
}

///|
pub fn StreamingSafetensorsVisitedRange::byte_count(self : Self) -> Int {
  self.byte_count
}

///|
pub fn StreamingSafetensorsSegmentedVisitedRange::shard_ordinal(
  self : Self,
) -> Int {
  self.shard_ordinal
}

///|
pub fn StreamingSafetensorsSegmentedVisitedRange::tensor_ordinal(
  self : Self,
) -> Int {
  self.tensor_ordinal
}

///|
pub fn StreamingSafetensorsSegmentedVisitedRange::tensor_byte_offset(
  self : Self,
) -> Int64 {
  self.tensor_byte_offset
}

///|
pub fn StreamingSafetensorsSegmentedVisitedRange::destination_arena_ordinal(
  self : Self,
) -> Int {
  self.destination_arena_ordinal
}

///|
pub fn StreamingSafetensorsSegmentedVisitedRange::destination_offset(
  self : Self,
) -> Int {
  self.destination_offset
}

///|
pub fn StreamingSafetensorsSegmentedVisitedRange::byte_count(
  self : Self,
) -> Int {
  self.byte_count
}
