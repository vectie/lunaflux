///|
fn sidecar_test_identity(seed : Char) -> @spec.ModelIdentity {
  let plan_seed = if seed == 'a' { 'b' } else { 'e' }
  try! @spec.ModelIdentity::new(
    try! @spec.ContentDigest::from_sha256(String::make(64, seed)),
    try! @spec.PlanDigest::from_sha256(String::make(64, plan_seed)),
  )
}

///|
fn sidecar_test_metadata(
  identity : @spec.ModelIdentity,
  manifest_seed : Char,
  layers : ArrayView[Int],
  byte_counts : ArrayView[Int],
  table_digest_seed : Char,
  released : Bool,
) -> TokenHashSidecarMetadata {
  let tables : Array[TokenHashSidecarTableMetadata] = []
  for ordinal, layer in layers {
    tables.push({
      layer_index: layer,
      digest: String::make(64, table_digest_seed),
      byte_count: byte_counts[ordinal],
    })
  }
  {
    identity,
    source_manifest_digest: String::make(64, manifest_seed),
    tables: ReadOnlyArray::from_array(tables),
    released,
  }
}

///|
fn exact_sidecar_test_metadata() -> TokenHashSidecarMetadata {
  sidecar_test_metadata(
    sidecar_test_identity('a'),
    'c',
    [0, 1, 2],
    [
      official_token_hash_table_bytes, official_token_hash_table_bytes, official_token_hash_table_bytes,
    ],
    'd',
    false,
  )
}

///|
fn sidecar_test_limits(
  maximum_source_arenas? : Int = 3,
  maximum_regions? : Int = 3,
  maximum_host_arena_bytes? : Int = official_token_hash_table_bytes,
  maximum_total_source_bytes? : Int64 = 9308160L,
  maximum_device_arena_bytes? : Int64 = 9308160L,
) -> @segmented_device_materialize.SegmentedDeviceMaterializationLimits {
  try! @segmented_device_materialize.SegmentedDeviceMaterializationLimits::new(
    alignment_bytes=256L,
    maximum_source_arenas~,
    maximum_regions~,
    maximum_host_arena_bytes~,
    maximum_total_source_bytes~,
    maximum_device_arena_bytes~,
  )
}

///|
fn exact_sidecar_regions(
  source_order : ArrayView[Int],
) -> Array[@segmented_device_materialize.SegmentedDeviceTensorRegionInput] {
  let regions : Array[
    @segmented_device_materialize.SegmentedDeviceTensorRegionInput,
  ] = []
  for ordinal, source in source_order {
    regions.push(
      try! @segmented_device_materialize.SegmentedDeviceTensorRegionInput::new(
        tensor_ordinal=ordinal,
        source_arena_ordinal=source,
        source_offset=0,
        byte_count=official_token_hash_table_bytes,
      ),
    )
  }
  regions
}

///|
test "token-hash sidecar layout binds deterministic exact official regions" {
  let metadata = exact_sidecar_test_metadata()
  let first = plan_token_hash_metadata_upload(metadata, sidecar_test_limits())
  let repeated = plan_token_hash_metadata_upload(
    metadata,
    sidecar_test_limits(),
  )
  assert_eq(first, repeated)
  assert_eq(first.source_identity(), metadata.identity)
  assert_eq(first.source_arena_count(), 3)
  assert_eq(first.region_count(), 3)
  assert_eq(first.materialized_bytes(), 9308160L)
  assert_eq(first.device_arena_bytes(), 9308160L)
  for ordinal in 0..<3 {
    assert_eq(
      first.source_arena_length(ordinal),
      Some(official_token_hash_table_bytes),
    )
    guard first.region(ordinal) is Some(region) else {
      fail("official sidecar region missing")
    }
    assert_eq(region.tensor_ordinal(), ordinal)
    assert_eq(region.source_arena_ordinal(), ordinal)
    assert_eq(region.source_offset(), 0)
    assert_eq(
      region.device_offset(),
      ordinal.to_int64() * official_token_hash_table_bytes.to_int64(),
    )
    assert_eq(region.byte_count(), official_token_hash_table_bytes.to_int64())
  }
  let changed_manifest = sidecar_test_metadata(
    metadata.identity,
    'f',
    [0, 1, 2],
    [
      official_token_hash_table_bytes, official_token_hash_table_bytes, official_token_hash_table_bytes,
    ],
    'd',
    false,
  )
  let changed = plan_token_hash_metadata_upload(
    changed_manifest,
    sidecar_test_limits(),
  )
  assert_true(first.source_digest() != changed.source_digest())
  assert_true(first.digest() != changed.digest())
}

///|
test "released order count truncation and digest metadata fail closed" {
  let identity = sidecar_test_identity('a')
  let exact_bytes = official_token_hash_table_bytes
  let hostile = [
    sidecar_test_metadata(
      identity,
      'c',
      [0, 1, 2],
      [exact_bytes, exact_bytes, exact_bytes],
      'd',
      true,
    ),
    sidecar_test_metadata(
      identity,
      'c',
      [0, 1],
      [exact_bytes, exact_bytes],
      'd',
      false,
    ),
    sidecar_test_metadata(
      identity,
      'c',
      [0, 2, 1],
      [exact_bytes, exact_bytes, exact_bytes],
      'd',
      false,
    ),
    sidecar_test_metadata(
      identity,
      'c',
      [0, 1, 2],
      [exact_bytes, exact_bytes - 4, exact_bytes],
      'd',
      false,
    ),
    sidecar_test_metadata(
      identity,
      'c',
      [0, 1, 2],
      [exact_bytes, exact_bytes, exact_bytes],
      'z',
      false,
    ),
  ]
  for index, metadata in hostile {
    try plan_token_hash_metadata_upload(metadata, sidecar_test_limits()) catch {
      TokenHashSourceReleased if index == 0 => ()
      TokenHashLayerCountMismatch(2) if index == 1 => ()
      TokenHashLayerOrderMismatch(ordinal=1, layer=2) if index == 2 => ()
      TokenHashTableByteCountMismatch(ordinal=1, actual~) if index == 3 =>
        assert_eq(actual, exact_bytes - 4)
      TokenHashMetadataDigestInvalid(0) if index == 4 => ()
      _ => fail("unexpected hostile sidecar metadata rejection")
    } noraise {
      _ => fail("hostile sidecar metadata was admitted")
    }
  }
}

///|
test "sidecar validation rejects identity source layout and live truncation" {
  let metadata = exact_sidecar_test_metadata()
  let limits = sidecar_test_limits()
  let layout = plan_token_hash_metadata_upload(metadata, limits)
  let foreign_identity = sidecar_test_metadata(
    sidecar_test_identity('d'),
    'c',
    [0, 1, 2],
    [
      official_token_hash_table_bytes, official_token_hash_table_bytes, official_token_hash_table_bytes,
    ],
    'd',
    false,
  )
  try validate_token_hash_layout(foreign_identity, layout) catch {
    TokenHashIdentityMismatch => ()
    _ => fail("unexpected token-hash identity rejection")
  } noraise {
    _ => fail("foreign sidecar identity was admitted")
  }
  let changed_table = sidecar_test_metadata(
    metadata.identity,
    'c',
    [0, 1, 2],
    [
      official_token_hash_table_bytes, official_token_hash_table_bytes, official_token_hash_table_bytes,
    ],
    'f',
    false,
  )
  let wrong_source_layout = try! @segmented_device_materialize.plan_layout(
    source_identity=metadata.identity,
    source_digest=token_hash_source_digest(changed_table),
    source_arena_lengths=[
      official_token_hash_table_bytes, official_token_hash_table_bytes, official_token_hash_table_bytes,
    ],
    regions=exact_sidecar_regions([0, 1, 2]),
    limits~,
  )
  try validate_token_hash_layout(metadata, wrong_source_layout) catch {
    TokenHashSourceDigestMismatch => ()
    _ => fail("unexpected token-hash source-digest rejection")
  } noraise {
    _ => fail("foreign sidecar source digest was admitted")
  }
  let wrong_regions = try! @segmented_device_materialize.plan_layout(
    source_identity=metadata.identity,
    source_digest=token_hash_source_digest(metadata),
    source_arena_lengths=[
      official_token_hash_table_bytes, official_token_hash_table_bytes, official_token_hash_table_bytes,
    ],
    regions=exact_sidecar_regions([1, 0, 2]),
    limits~,
  )
  try validate_token_hash_layout(metadata, wrong_regions) catch {
    TokenHashLayoutMismatch => ()
    _ => fail("unexpected token-hash layout rejection")
  } noraise {
    _ => fail("reordered sidecar source regions were admitted")
  }
  validate_token_hash_metadata_source(metadata, layout, provide=(_, consumer) => {
    consumer.accept(FixedArray::make(official_token_hash_table_bytes, b'\x00'))
  })
  try
    validate_token_hash_metadata_source(metadata, layout, provide=(
      ordinal,
      consumer,
    ) => {
      consumer.accept(
        FixedArray::make(
          if ordinal == 1 {
            official_token_hash_table_bytes - 4
          } else {
            official_token_hash_table_bytes
          },
          b'\x00',
        ),
      )
    })
  catch {
    Device(HostArenaLengthMismatch(1)) => ()
    _ => fail("unexpected live sidecar truncation rejection")
  } noraise {
    _ => fail("truncated live sidecar arena was admitted")
  }
}

///|
test "sidecar source region and byte budgets fail before layout escapes" {
  let metadata = exact_sidecar_test_metadata()
  let limits = [
    sidecar_test_limits(maximum_source_arenas=2),
    sidecar_test_limits(maximum_regions=2),
    sidecar_test_limits(
      maximum_host_arena_bytes=official_token_hash_table_bytes - 1,
    ),
    sidecar_test_limits(maximum_total_source_bytes=9308159L),
    sidecar_test_limits(maximum_device_arena_bytes=9308159L),
  ]
  for index, limit in limits {
    try plan_token_hash_metadata_upload(metadata, limit) catch {
      Device(TooManySourceArenas) if index == 0 => ()
      Device(TooManyRegions) if index == 1 => ()
      Device(InvalidSourceArena(0)) if index == 2 => ()
      Device(SourceByteBudgetExceeded) if index == 3 => ()
      Device(DeviceArenaTooLarge) if index == 4 => ()
      _ => fail("unexpected token-hash layout-budget rejection")
    } noraise {
      _ => fail("undersized token-hash layout budget was admitted")
    }
  }
}
