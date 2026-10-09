///|
priv struct TokenHashSidecarTableMetadata {
  layer_index : Int
  digest : String
  byte_count : Int
}

///|
priv struct TokenHashSidecarMetadata {
  identity : @spec.ModelIdentity
  source_manifest_digest : String
  tables : ReadOnlyArray[TokenHashSidecarTableMetadata]
  released : Bool
}

///|
let official_token_hash_layer_count : Int = 3

///|
let official_token_hash_table_bytes : Int = 129280 * 6 * 4

///|
fn canonical_digest_text(value : StringView) -> Bool {
  value.length() == 64 &&
  value.all(character => {
    character.is_ascii_digit() || (character >= 'a' && character <= 'f')
  })
}

///|
fn hash_sidecar_field(hash : @crypto.SHA256, value : StringView) -> Unit {
  let prefix = @utf8.encode(value.length().to_string() + ":")
  let encoded = @utf8.encode(value)
  hash.update_from_iter(prefix.iter())
  hash.update_from_iter(encoded.iter())
  hash.update_from_iter([b'\n'].iter())
}

///|
fn validate_token_hash_metadata(
  metadata : TokenHashSidecarMetadata,
) -> Unit raise DeepSeekV4DeviceUploadError {
  if metadata.released {
    raise TokenHashSourceReleased
  }
  if !canonical_digest_text(metadata.source_manifest_digest) {
    raise TokenHashMetadataDigestInvalid(-1)
  }
  if metadata.tables.length() != official_token_hash_layer_count {
    raise TokenHashLayerCountMismatch(metadata.tables.length())
  }
  for ordinal, table in metadata.tables {
    if table.layer_index != ordinal {
      raise TokenHashLayerOrderMismatch(ordinal~, layer=table.layer_index)
    }
    if table.byte_count != official_token_hash_table_bytes {
      raise TokenHashTableByteCountMismatch(ordinal~, actual=table.byte_count)
    }
    if !canonical_digest_text(table.digest) {
      raise TokenHashMetadataDigestInvalid(ordinal)
    }
  }
}

///|
fn token_hash_metadata(
  tables : @deepseek_v4_host_materialization.DeepSeekV4TokenHashTables,
) -> TokenHashSidecarMetadata raise DeepSeekV4DeviceUploadError {
  if tables.is_released() {
    raise TokenHashSourceReleased
  }
  let entries : Array[TokenHashSidecarTableMetadata] = []
  for ordinal in 0..<tables.table_count() {
    entries.push({
      layer_index: from_token_hash(() => tables.layer_index(ordinal)),
      digest: from_token_hash(() => tables.table_digest(ordinal)).as_hex(),
      byte_count: from_token_hash(() => tables.table_byte_count(ordinal)),
    })
  }
  let metadata = {
    identity: tables.identity(),
    source_manifest_digest: tables.source_manifest_digest().as_hex(),
    tables: ReadOnlyArray::from_array(entries),
    released: false,
  }
  validate_token_hash_metadata(metadata)
  metadata
}

///|
fn token_hash_source_digest(
  metadata : TokenHashSidecarMetadata,
) -> @segmented_device_materialize.SegmentedHostSourceDigest raise DeepSeekV4DeviceUploadError {
  validate_token_hash_metadata(metadata)
  let hash = @crypto.SHA256::new()
  hash_sidecar_field(hash, "lunaflux/deepseek-v4-token-hash-sidecars/v1")
  hash_sidecar_field(hash, metadata.identity.content().as_hex())
  hash_sidecar_field(hash, metadata.identity.plan().as_hex())
  hash_sidecar_field(hash, metadata.source_manifest_digest)
  hash_sidecar_field(hash, metadata.tables.length().to_string())
  for ordinal, table in metadata.tables {
    hash_sidecar_field(hash, ordinal.to_string())
    hash_sidecar_field(hash, table.layer_index.to_string())
    hash_sidecar_field(hash, table.digest)
    hash_sidecar_field(hash, table.byte_count.to_string())
  }
  from_device(() => {
    @segmented_device_materialize.SegmentedHostSourceDigest::from_sha256(
      @crypto.bytes_to_hex_string(hash.finalize()),
    )
  })
}

///|
fn plan_token_hash_metadata_upload(
  metadata : TokenHashSidecarMetadata,
  limits : @segmented_device_materialize.SegmentedDeviceMaterializationLimits,
) -> @segmented_device_materialize.SegmentedDeviceLayout raise DeepSeekV4DeviceUploadError {
  let digest = token_hash_source_digest(metadata)
  let arena_lengths : Array[Int] = []
  let regions : Array[
    @segmented_device_materialize.SegmentedDeviceTensorRegionInput,
  ] = []
  for ordinal, table in metadata.tables {
    arena_lengths.push(table.byte_count)
    regions.push(
      from_device(() => {
        @segmented_device_materialize.SegmentedDeviceTensorRegionInput::new(
          tensor_ordinal=table.layer_index,
          source_arena_ordinal=ordinal,
          source_offset=0,
          byte_count=table.byte_count,
        )
      }),
    )
  }
  from_device(() => {
    @segmented_device_materialize.plan_layout(
      source_identity=metadata.identity,
      source_digest=digest,
      source_arena_lengths=arena_lengths,
      regions~,
      limits~,
    )
  })
}

///|
fn expected_aligned_offset(value : Int64, alignment : Int64) -> Int64 {
  let remainder = value % alignment
  if remainder == 0L {
    value
  } else {
    value + alignment - remainder
  }
}

///|
fn validate_token_hash_layout(
  metadata : TokenHashSidecarMetadata,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
) -> Unit raise DeepSeekV4DeviceUploadError {
  let digest = token_hash_source_digest(metadata)
  if layout.source_identity() != metadata.identity {
    raise TokenHashIdentityMismatch
  }
  if layout.source_digest() != digest {
    raise TokenHashSourceDigestMismatch
  }
  if layout.source_arena_count() != metadata.tables.length() ||
    layout.region_count() != metadata.tables.length() {
    raise TokenHashLayoutMismatch
  }
  let mut device_end = 0L
  let mut materialized = 0L
  for ordinal, table in metadata.tables {
    if layout.source_arena_length(ordinal) != Some(table.byte_count) {
      raise TokenHashLayoutMismatch
    }
    guard layout.region(ordinal) is Some(region) else {
      raise TokenHashLayoutMismatch
    }
    let expected_offset = expected_aligned_offset(
      device_end,
      layout.alignment_bytes(),
    )
    if region.tensor_ordinal() != table.layer_index ||
      region.source_arena_ordinal() != ordinal ||
      region.source_offset() != 0 ||
      region.byte_count() != table.byte_count.to_int64() ||
      region.device_offset() != expected_offset {
      raise TokenHashLayoutMismatch
    }
    device_end = expected_offset + table.byte_count.to_int64()
    materialized += table.byte_count.to_int64()
  }
  if layout.materialized_bytes() != materialized ||
    layout.device_arena_bytes() != device_end {
    raise TokenHashLayoutMismatch
  }
}

///|
/// Plans one isolated I32 device region for each official hash-routed layer.
/// The source digest binds model identity, source manifest, ordered layer ids,
/// checked table digests, and exact byte counts.
pub fn plan_token_hash_sidecar_upload(
  tables : @deepseek_v4_host_materialization.DeepSeekV4TokenHashTables,
  limits : @segmented_device_materialize.SegmentedDeviceMaterializationLimits,
) -> @segmented_device_materialize.SegmentedDeviceLayout raise DeepSeekV4DeviceUploadError {
  plan_token_hash_metadata_upload(token_hash_metadata(tables), limits)
}

///|
/// Validates the live, synchronously borrowed I32 sidecars against their exact
/// family-owned layout without allocating device memory.
pub fn validate_token_hash_sidecar_source(
  tables : @deepseek_v4_host_materialization.DeepSeekV4TokenHashTables,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
) -> Unit raise DeepSeekV4DeviceUploadError {
  let metadata = token_hash_metadata(tables)
  validate_token_hash_metadata_source(metadata, layout, provide=(
    ordinal,
    consumer,
  ) => tables.provide_table_arena(ordinal, consumer))
}

///|
fn validate_token_hash_metadata_source(
  metadata : TokenHashSidecarMetadata,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
  provide~ : (Int, @segmented_host_borrow.SegmentedHostArenaConsumer) -> Unit raise @segmented_host_borrow.SegmentedHostBorrowError,
) -> Unit raise DeepSeekV4DeviceUploadError {
  validate_token_hash_layout(metadata, layout)
  from_device(() => {
    @segmented_device_materialize.validate_borrowed_source(layout, provide~)
  })
}

///|
/// Uploads only checked I32 token-hash sidecars. Raw model weights remain in a
/// separate allocation and this result grants no kernel launch authority.
pub fn materialize_token_hash_sidecars(
  context : @device.Context,
  tables : @deepseek_v4_host_materialization.DeepSeekV4TokenHashTables,
  layout : @segmented_device_materialize.SegmentedDeviceLayout,
) -> @segmented_device_materialize.SegmentedDevicePreparation raise DeepSeekV4DeviceUploadError {
  let metadata = token_hash_metadata(tables)
  validate_token_hash_layout(metadata, layout)
  let digest = token_hash_source_digest(metadata)
  from_device(() => {
    @segmented_device_materialize.materialize_from_borrowed_source(
      context,
      layout,
      expected_identity=metadata.identity,
      expected_source_digest=digest,
      expected_layout_digest=layout.digest(),
      provide=(ordinal, consumer) => {
        tables.provide_table_arena(ordinal, consumer)
      },
    )
  })
}
