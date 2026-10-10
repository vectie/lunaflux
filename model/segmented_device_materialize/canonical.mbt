///|
fn hash_layout_field(hash : @crypto.SHA256, value : StringView) -> Unit {
  let prefix = @utf8.encode(value.length().to_string() + ":")
  let encoded = @utf8.encode(value)
  hash.update_from_iter(prefix.iter())
  hash.update_from_iter(encoded.iter())
  hash.update_from_iter([b'\n'].iter())
}

///|
fn canonical_layout_digest(
  identity : @spec.ModelIdentity,
  source_digest : SegmentedHostSourceDigest,
  alignment_bytes : Int64,
  device_arena_bytes : Int64,
  materialized_bytes : Int64,
  source_arena_lengths : ReadOnlyArray[Int],
  regions : ReadOnlyArray[SegmentedDeviceTensorRegion],
) -> SegmentedDeviceLayoutDigest {
  let hash = @crypto.SHA256::new()
  hash_layout_field(hash, "lunaflux/segmented-device-layout/v1")
  hash_layout_field(hash, identity.content().as_hex())
  hash_layout_field(hash, identity.plan().as_hex())
  hash_layout_field(hash, source_digest.hex)
  hash_layout_field(hash, alignment_bytes.to_string())
  hash_layout_field(hash, device_arena_bytes.to_string())
  hash_layout_field(hash, materialized_bytes.to_string())
  hash_layout_field(hash, source_arena_lengths.length().to_string())
  for index, length in source_arena_lengths {
    hash_layout_field(hash, index.to_string())
    hash_layout_field(hash, length.to_string())
  }
  hash_layout_field(hash, regions.length().to_string())
  for region in regions {
    hash_layout_field(hash, region.tensor_ordinal.to_string())
    hash_layout_field(hash, region.source_arena_ordinal.to_string())
    hash_layout_field(hash, region.source_offset.to_string())
    hash_layout_field(hash, region.device_offset.to_string())
    hash_layout_field(hash, region.byte_count.to_string())
  }
  { hex: @crypto.bytes_to_hex_string(hash.finalize()), }
}
