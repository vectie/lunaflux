///|
pub fn MiniMaxH3HostMaterializationLimits::new(
  maximum_shards~ : Int,
  maximum_tensors~ : Int,
  maximum_header_bytes~ : Int,
  maximum_payload_bytes_per_shard~ : Int64,
  maximum_total_host_bytes~ : Int64,
  maximum_arena_bytes~ : Int64,
  maximum_arenas~ : Int,
  maximum_name_bytes~ : Int,
) -> Self raise MiniMaxH3HostMaterializationError {
  if maximum_shards <= 0 ||
    maximum_shards > 256 ||
    maximum_tensors <= 0 ||
    maximum_tensors > 4096 ||
    maximum_header_bytes <= 0 ||
    maximum_header_bytes > 67108864 ||
    maximum_payload_bytes_per_shard <= 0L ||
    maximum_payload_bytes_per_shard >
    9223372036854775807L - maximum_header_bytes.to_int64() - 8L ||
    maximum_total_host_bytes <= 0L ||
    maximum_arena_bytes <= 0L ||
    maximum_arena_bytes > 2147483647L ||
    maximum_arena_bytes > maximum_total_host_bytes ||
    maximum_arenas <= 0 ||
    maximum_arenas > 4096 ||
    maximum_name_bytes <= 0 ||
    maximum_name_bytes > 1024 {
    raise InvalidLimit
  }
  {
    maximum_shards,
    maximum_tensors,
    maximum_header_bytes,
    maximum_payload_bytes_per_shard,
    maximum_total_host_bytes,
    maximum_arena_bytes,
    maximum_arenas,
    maximum_name_bytes,
  }
}

///|
pub fn MiniMaxH3ShardDeclarationInput::new(
  file_digest~ : @spec.ContentDigest,
  payload_bytes~ : Int64,
) -> Self raise MiniMaxH3HostMaterializationError {
  if payload_bytes <= 0L {
    raise InvalidShardCount
  }
  { file_digest, payload_bytes, }
}

///|
pub fn MiniMaxH3TensorIndexInput::new(
  tensor_name~ : String,
  shard_index~ : Int,
) -> Self {
  { tensor_name, shard_index, }
}

///|
pub fn MiniMaxH3ShardSource::new(
  locator~ : @approved_fs.ApprovedRelativeLocator,
) -> Self {
  { locator, }
}

///|
pub fn MiniMaxH3HostManifestDigest::as_hex(self : Self) -> String {
  self.hex
}
