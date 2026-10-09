///|
fn canonical_sha256(value : StringView) -> Bool {
  value.length() == 64 &&
  value.all(character => {
    character.is_ascii_digit() || (character >= 'a' && character <= 'f')
  })
}

///|
pub fn SegmentedHostSourceDigest::from_sha256(
  hex : String,
) -> Self raise SegmentedDeviceMaterializationError {
  if !canonical_sha256(hex) {
    raise InvalidDigest
  }
  { hex, }
}

///|
pub fn SegmentedDeviceMaterializationLimits::new(
  alignment_bytes~ : Int64,
  maximum_source_arenas~ : Int,
  maximum_regions~ : Int,
  maximum_host_arena_bytes~ : Int,
  maximum_total_source_bytes~ : Int64,
  maximum_device_arena_bytes~ : Int64,
) -> Self raise SegmentedDeviceMaterializationError {
  if alignment_bytes < 16L ||
    alignment_bytes > 4096L ||
    (alignment_bytes & (alignment_bytes - 1L)) != 0L ||
    maximum_source_arenas <= 0 ||
    maximum_source_arenas > 4096 ||
    maximum_regions <= 0 ||
    maximum_regions > 1000000 ||
    maximum_host_arena_bytes <= 0 ||
    maximum_host_arena_bytes > 2147483647 ||
    maximum_total_source_bytes <= 0L ||
    maximum_device_arena_bytes <= 0L {
    raise InvalidLimit
  }
  {
    alignment_bytes,
    maximum_source_arenas,
    maximum_regions,
    maximum_host_arena_bytes,
    maximum_total_source_bytes,
    maximum_device_arena_bytes,
  }
}

///|
pub fn SegmentedDeviceTensorRegionInput::new(
  tensor_ordinal~ : Int,
  source_arena_ordinal~ : Int,
  source_offset~ : Int,
  byte_count~ : Int,
) -> Self raise SegmentedDeviceMaterializationError {
  if tensor_ordinal < 0 ||
    source_arena_ordinal < 0 ||
    source_offset < 0 ||
    byte_count <= 0 {
    raise InvalidSourceRegion(tensor_ordinal)
  }
  { tensor_ordinal, source_arena_ordinal, source_offset, byte_count, }
}
