///|
fn hash_host_source_field(hash : @crypto.SHA256, value : StringView) -> Unit {
  let prefix = @utf8.encode(value.length().to_string() + ":")
  let encoded = @utf8.encode(value)
  hash.update_from_iter(prefix.iter())
  hash.update_from_iter(encoded.iter())
  hash.update_from_iter([b'\n'].iter())
}

///|
fn canonical_host_source_digest(
  expected : Glm53HostExpectedDigests,
  sources : ArrayView[Glm53HostShardSource],
  prepared : Glm53PreparedHostMaterialization,
) -> Glm53HostSourceDigest {
  let hash = @crypto.SHA256::new()
  hash_host_source_field(hash, "lunaflux/glm53-host-source/v1")
  hash_host_source_field(hash, expected.execution_identity.content().as_hex())
  hash_host_source_field(hash, expected.execution_identity.plan().as_hex())
  hash_host_source_field(hash, expected.manifest_layout.as_hex())
  hash_host_source_field(hash, expected.numeric_schema.as_hex())
  hash_host_source_field(hash, expected.numeric_binding.as_hex())
  hash_host_source_field(hash, sources.length().to_string())
  for ordinal, source in sources {
    hash_host_source_field(hash, ordinal.to_string())
    hash_host_source_field(hash, source.artifact_digest.as_hex())
  }
  hash_host_source_field(hash, prepared.total_bytes.to_string())
  hash_host_source_field(hash, prepared.arenas.length().to_string())
  for ordinal, arena in prepared.arenas {
    hash_host_source_field(hash, ordinal.to_string())
    hash_host_source_field(hash, arena.bytes.length().to_string())
  }
  hash_host_source_field(hash, prepared.regions.length().to_string())
  for region in prepared.regions {
    hash_host_source_field(hash, region.ordinal.to_string())
    hash_host_source_field(hash, region.byte_length.to_string())
    hash_host_source_field(hash, region.spans.length().to_string())
    for span in region.spans {
      hash_host_source_field(hash, span.arena_index.to_string())
      hash_host_source_field(hash, span.offset_bytes.to_string())
      hash_host_source_field(hash, span.byte_length.to_string())
    }
  }
  { hex: @crypto.bytes_to_hex_string(hash.finalize()), }
}
