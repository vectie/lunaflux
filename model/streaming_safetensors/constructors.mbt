///|
fn valid_ascii_token(value : StringView, maximum : Int) -> Bool {
  !value.is_empty() &&
  value.length() <= maximum &&
  value.all(character => {
    character.is_ascii() &&
    !character.is_ascii_control() &&
    !character.is_ascii_whitespace()
  })
}

///|
pub fn StreamingSafetensorsDTypeSpec::bfloat16() -> Self {
  { kind: BFloat16, tag: "BF16", bytes_per_element: 2L, }
}

///|
pub fn StreamingSafetensorsDTypeSpec::float32() -> Self {
  { kind: Float32, tag: "F32", bytes_per_element: 4L, }
}

///|
pub fn StreamingSafetensorsDTypeSpec::opaque_one_byte(
  tag : StringView,
) -> Self raise StreamingSafetensorsError {
  StreamingSafetensorsDTypeSpec::opaque_fixed_width(tag, bytes_per_element=1L)
}

///|
pub fn StreamingSafetensorsDTypeSpec::opaque_fixed_width(
  tag : StringView,
  bytes_per_element~ : Int64,
) -> Self raise StreamingSafetensorsError {
  if !valid_ascii_token(tag, 64) ||
    tag == "BF16" ||
    tag == "F32" ||
    bytes_per_element <= 0L ||
    bytes_per_element > 1048576L {
    raise InvalidDTypeSpec
  }
  { kind: OpaqueFixedWidth, tag: tag.to_owned(), bytes_per_element, }
}

///|
pub fn StreamingSafetensorsFileDigest::from_sha256(
  hex : String,
) -> Self raise StreamingSafetensorsError {
  if hex.length() != 64 ||
    !hex[:].all(character => {
      character.is_ascii_digit() || (character >= 'a' && character <= 'f')
    }) {
    raise InvalidDigest
  }
  { hex, }
}

///|
pub fn StreamingSafetensorsShardInput::new(
  locator~ : @approved_fs.ApprovedRelativeLocator,
  expected_digest~ : StreamingSafetensorsFileDigest,
) -> Self {
  { locator, expected_digest, }
}

///|
pub fn StreamingSafetensorsLimits::new(
  maximum_shards~ : Int,
  maximum_file_bytes~ : Int64,
  maximum_header_bytes~ : Int,
  maximum_tensors_per_shard~ : Int,
  maximum_total_tensors~ : Int,
  maximum_rank~ : Int,
  maximum_name_bytes~ : Int,
  maximum_dtype_specs~ : Int,
  authentication_chunk_bytes~ : Int,
  maximum_copy_requests~ : Int,
  maximum_copy_bytes~ : Int64,
) -> Self raise StreamingSafetensorsError {
  if maximum_shards <= 0 ||
    maximum_shards > 4096 ||
    maximum_file_bytes <= 8L ||
    maximum_header_bytes <= 0 ||
    maximum_header_bytes > 67108864 ||
    maximum_tensors_per_shard <= 0 ||
    maximum_tensors_per_shard > 1000000 ||
    maximum_total_tensors <= 0 ||
    maximum_total_tensors > 1000000 ||
    maximum_rank <= 0 ||
    maximum_rank > 64 ||
    maximum_name_bytes <= 0 ||
    maximum_name_bytes > 4096 ||
    maximum_dtype_specs <= 0 ||
    maximum_dtype_specs > 256 ||
    authentication_chunk_bytes <= 0 ||
    authentication_chunk_bytes > 67108864 ||
    maximum_copy_requests <= 0 ||
    maximum_copy_requests > 1000000 ||
    maximum_copy_bytes <= 0L {
    raise InvalidLimit
  }
  {
    maximum_shards,
    maximum_file_bytes,
    maximum_header_bytes,
    maximum_tensors_per_shard,
    maximum_total_tensors,
    maximum_rank,
    maximum_name_bytes,
    maximum_dtype_specs,
    authentication_chunk_bytes,
    maximum_copy_requests,
    maximum_copy_bytes,
  }
}

///|
pub fn StreamingSafetensorsCopyRequest::new(
  shard_ordinal~ : Int,
  tensor_ordinal~ : Int,
  tensor_byte_offset~ : Int64,
  destination_offset~ : Int,
  byte_count~ : Int,
) -> Self raise StreamingSafetensorsError {
  if shard_ordinal < 0 ||
    tensor_ordinal < 0 ||
    tensor_byte_offset < 0L ||
    destination_offset < 0 ||
    byte_count <= 0 {
    raise InvalidCopyRequest(-1)
  }
  {
    shard_ordinal,
    tensor_ordinal,
    tensor_byte_offset,
    destination_offset,
    byte_count,
  }
}

///|
pub fn StreamingSafetensorsSegmentedCopyRequest::new(
  shard_ordinal~ : Int,
  tensor_ordinal~ : Int,
  tensor_byte_offset~ : Int64,
  destination_arena_ordinal~ : Int,
  destination_offset~ : Int,
  byte_count~ : Int,
) -> Self raise StreamingSafetensorsError {
  if shard_ordinal < 0 ||
    tensor_ordinal < 0 ||
    tensor_byte_offset < 0L ||
    destination_arena_ordinal < 0 ||
    destination_offset < 0 ||
    byte_count <= 0 {
    raise InvalidCopyRequest(-1)
  }
  {
    shard_ordinal,
    tensor_ordinal,
    tensor_byte_offset,
    destination_arena_ordinal,
    destination_offset,
    byte_count,
  }
}
