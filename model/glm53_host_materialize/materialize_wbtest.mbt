///|
fn fixture_limits(max_arena_bytes? : Int = 5) -> Glm53HostMaterializationLimits {
  try! Glm53HostMaterializationLimits::new(
    max_header_bytes=4096,
    max_tensor_count=8,
    max_rank=4,
    max_name_bytes=128,
    max_source_count=4,
    max_file_bytes=8192L,
    max_arena_bytes~,
    max_total_arena_bytes=8192L,
  )
}

///|
fn fixture_file(header : StringView, payload : BytesView) -> Bytes {
  let encoded = @utf8.encode(header)
  let output : Array[Byte] = []
  let length = encoded.length().to_uint64()
  for shift in 0..<8 {
    output.push(((length >> (shift * 8)) & 0xffUL).to_byte())
  }
  for byte in encoded {
    output.push(byte)
  }
  for byte in payload {
    output.push(byte)
  }
  Bytes::from_array(output)
}

///|
fn fixture_digest(
  bytes : Bytes,
) -> @streaming_safetensors.StreamingSafetensorsFileDigest {
  try! @streaming_safetensors.StreamingSafetensorsFileDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn fixture_source(name : StringView, bytes : Bytes) -> Glm53HostShardSource {
  Glm53HostShardSource::new(
    locator=try! @approved_fs.ApprovedRelativeLocator::new(name),
    artifact_digest=fixture_digest(bytes),
  )
}

///|
fn mixed_expected() -> Array[Glm53ExpectedHostTensor] {
  [
    {
      ordinal: 0,
      name: "weight",
      dtype: BFloat16,
      shape: ReadOnlyArray::from_array([2L]),
    },
    {
      ordinal: 1,
      name: "route",
      dtype: Float32,
      shape: ReadOnlyArray::from_array([1L]),
    },
  ]
}

///|
fn fixture_expected_digests() -> Glm53HostExpectedDigests {
  let content = try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = try! @spec.PlanDigest::from_sha256(
    "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  Glm53HostExpectedDigests::new(
    execution_identity=try! @spec.ModelIdentity::new(content, plan),
    manifest_layout=try! @spec.PlanDigest::from_sha256(
      "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    numeric_schema=try! @numeric_contract.ModelNumericSchemaDigest::from_sha256(
      "3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    numeric_binding=try! @spec.PlanDigest::from_sha256(
      "4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
async test "neutral segmented copy fills canonical mixed arenas and close drops without scrubbing" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-glm53-host")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let route = fixture_file(
      (
        #|{"route":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
      ),
      b"\x05\x06\x07\x08",
    )
    let weight = fixture_file(
      (
        #|{"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}
      ),
      b"\x01\x02\x03\x04",
    )
    @fs.write_file(canonical + "/route", route, create_mode=CreateNew)
    @fs.write_file(canonical + "/weight", weight, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let limits = fixture_limits()
    let inspection = inspect_host_shards(
      root,
      [fixture_source("route", route), fixture_source("weight", weight)],
      limits,
    )
    let prepared = prepare_expected_host_materialization(
      mixed_expected(),
      inspection,
      limits,
    )
    assert_eq(prepared.total_bytes, 8L)
    assert_eq(prepared.arenas.length(), 2)
    copy_prepared_materialization(root, inspection, prepared, limits)
    let expected = fixture_expected_digests()
    let captured = prepared.arenas[0].bytes
    let weights : Glm53HostWeights = {
      execution_identity: expected.execution_identity,
      manifest_layout: expected.manifest_layout,
      numeric_schema: expected.numeric_schema,
      numeric_binding: expected.numeric_binding,
      source_digest: { hex: String::make(64, 'a'), },
      regions: prepared.regions,
      arenas: prepared.arenas,
      total_bytes: prepared.total_bytes,
      open: true,
    }
    assert_eq(weights.tensor_byte(0, 0L), b'\x01')
    assert_eq(weights.tensor_byte(0, 3L), b'\x04')
    assert_eq(weights.tensor_byte(1, 0L), b'\x05')
    assert_eq(weights.tensor_byte(1, 3L), b'\x08')
    assert_eq(weights.tensor_region(1).span_count(), 2)
    weights.close()
    assert_true(weights.is_closed())
    assert_eq(captured[0], b'\x01')
    assert_true(weights.arenas.iter().all(arena => arena.bytes.length() == 0))
    weights.close()
    try weights.tensor_byte(0, 0L) catch {
      Closed => ()
      _ => fail("unexpected closed-state failure")
    } noraise {
      _ => fail("closed materialization remained readable")
    }
    root.close()
  }
}

///|
async test "adapter rejects name dtype shape byte-range and payload-gap mutations" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-glm53-hostile")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let fixtures = [
      (
        "name",
        fixture_file(
          (
            #|{"other":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"route":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}
          ),
          b"12345678",
        ),
      ),
      (
        "dtype",
        fixture_file(
          (
            #|{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"route":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}
          ),
          b"12345678",
        ),
      ),
      (
        "shape",
        fixture_file(
          (
            #|{"weight":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]},"route":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}
          ),
          b"12345678",
        ),
      ),
      (
        "short",
        fixture_file(
          (
            #|{"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,2]},"route":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}
          ),
          b"123456",
        ),
      ),
      (
        "gap",
        fixture_file(
          (
            #|{"weight":{"dtype":"BF16","shape":[2],"data_offsets":[1,5]},"route":{"dtype":"F32","shape":[1],"data_offsets":[5,9]}}
          ),
          b"x12345678",
        ),
      ),
    ]
    for item in fixtures {
      let (name, bytes) = item
      @fs.write_file(canonical + "/" + name, bytes, create_mode=CreateNew)
    }
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let limits = fixture_limits()
    for index, item in fixtures {
      let (name, bytes) = item
      let admitted : Result[
        @streaming_safetensors.StreamingSafetensorsInspection,
        Glm53HostMaterializeError,
      ] = Ok(inspect_host_shards(root, [fixture_source(name, bytes)], limits)) catch {
        error => Err(error)
      }
      match (index, admitted) {
        (3, Err(StreamingAdmission(RangeSizeMismatch(shard=0, tensor=0)))) => ()
        (3, _) => fail("short byte range had unexpected result")
        (_, Err(_)) => fail("metadata fixture failed before adapter validation")
        (_, Ok(inspection)) =>
          try
            prepare_expected_host_materialization(
              mixed_expected(),
              inspection,
              limits,
            )
          catch {
            TensorNameMismatch(ordinal=0) if index == 0 => ()
            TensorDTypeMismatch(ordinal=0) if index == 1 => ()
            TensorShapeMismatch(ordinal=0) if index == 2 => ()
            PayloadCoverageMismatch(shard=0) if index == 4 => ()
            _ => fail("unexpected adapter mutation rejection")
          } noraise {
            _ => fail("hostile adapter mutation was admitted")
          }
      }
    }
    root.close()
  }
}

///|
async test "neutral duplicate rejection and declared inventory labels stay typed through adapter" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-glm53-auth")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let bytes = fixture_file(
      (
        #|{"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}
      ),
      b"1234",
    )
    @fs.write_file(canonical + "/a", bytes, create_mode=CreateNew)
    @fs.write_file(canonical + "/b", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let limits = fixture_limits()
    try
      inspect_host_shards(
        root,
        [fixture_source("a", bytes), fixture_source("b", bytes)],
        limits,
      )
    catch {
      StreamingAdmission(DuplicateTensorName(..)) => ()
      _ => fail("unexpected duplicate rejection")
    } noraise {
      _ => fail("cross-shard duplicate was admitted")
    }
    let forged = Glm53HostShardSource::new(
      locator=try! @approved_fs.ApprovedRelativeLocator::new("a"),
      artifact_digest=try! @streaming_safetensors.StreamingSafetensorsFileDigest::from_sha256(
        String::make(64, 'f'),
      ),
    )
    ignore(inspect_host_shards(root, [forged], limits))
    root.close()
  }
}

///|
test "neutral close failure drops all final arenas" {
  let arenas : Array[Glm53HostArena] = [
    { bytes: FixedArray::make(4, b'Z'), },
    { bytes: FixedArray::make(3, b'X'), },
  ]
  try
    run_segmented_copy_with_cleanup(arenas, () => raise FileCloseFailed(0))
  catch {
    StreamingAdmission(FileCloseFailed(0)) => ()
    _ => fail("unexpected close failure")
  } noraise {
    _ => fail("close failure was suppressed")
  }
  for arena in arenas {
    assert_eq(arena.bytes.length(), 0)
  }
}

///|
test "official manifest and numeric identities are exact and FP8 stays unavailable" {
  let metadata = @glm53_spec.Glm53FullMetadata::from_verified_content(
    @glm53_spec.Glm53FullSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
  let manifest = @glm53_weights.build_full_tensor_manifest(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let numeric = @glm53_numeric_plan.build_full_numeric_plan(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let exact = Glm53HostExpectedDigests::new(
    execution_identity=manifest.execution_identity(),
    manifest_layout=manifest.layout_digest(),
    numeric_schema=numeric.numeric_schema_digest(),
    numeric_binding=numeric.binding_digest(),
  )
  validate_expected_digests(manifest, numeric, exact)
  let forgeries = [
    Glm53HostExpectedDigests::new(
      execution_identity=manifest.execution_identity(),
      manifest_layout=try! @spec.PlanDigest::from_sha256(String::make(64, 'd')),
      numeric_schema=numeric.numeric_schema_digest(),
      numeric_binding=numeric.binding_digest(),
    ),
    Glm53HostExpectedDigests::new(
      execution_identity=manifest.execution_identity(),
      manifest_layout=manifest.layout_digest(),
      numeric_schema=try! @numeric_contract.ModelNumericSchemaDigest::from_sha256(
        String::make(64, 'e'),
      ),
      numeric_binding=numeric.binding_digest(),
    ),
    Glm53HostExpectedDigests::new(
      execution_identity=manifest.execution_identity(),
      manifest_layout=manifest.layout_digest(),
      numeric_schema=numeric.numeric_schema_digest(),
      numeric_binding=try! @spec.PlanDigest::from_sha256(String::make(64, 'f')),
    ),
  ]
  for index, forged in forgeries {
    try validate_expected_digests(manifest, numeric, forged) catch {
      DigestMismatch(ManifestLayout) if index == 0 => ()
      DigestMismatch(NumericSchema) if index == 1 => ()
      DigestMismatch(NumericBinding) if index == 2 => ()
      _ => fail("unexpected forged digest rejection")
    } noraise {
      _ => fail("forged host binding digest was admitted")
    }
  }
  let forged_identity = Glm53HostExpectedDigests::new(
    execution_identity=try! @spec.ModelIdentity::new(
      try! @spec.ContentDigest::from_sha256(String::make(64, 'c')),
      manifest.execution_identity().plan(),
    ),
    manifest_layout=manifest.layout_digest(),
    numeric_schema=numeric.numeric_schema_digest(),
    numeric_binding=numeric.binding_digest(),
  )
  try validate_expected_digests(manifest, numeric, forged_identity) catch {
    DigestMismatch(ExecutionIdentity) => ()
    _ => fail("unexpected execution identity rejection")
  } noraise {
    _ => fail("forged execution identity was admitted")
  }
  try
    @glm53_weights.build_full_tensor_manifest(metadata, Fp8E4M3Block128Dynamic)
  catch {
    UnsupportedPhysicalEncoding(_) => ()
    _ => fail("unexpected block-FP8 failure")
  } noraise {
    _ => fail("official block-FP8 was admitted")
  }
  let flash_metadata = @glm53_spec.Glm53FlashMetadata::from_verified_content(
    @glm53_spec.Glm53FlashSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "6123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
  let flash_manifest = @glm53_weights.build_flash_tensor_manifest(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  let flash_numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  validate_expected_digests(
    flash_manifest,
    flash_numeric,
    Glm53HostExpectedDigests::new(
      execution_identity=flash_manifest.execution_identity(),
      manifest_layout=flash_manifest.layout_digest(),
      numeric_schema=flash_numeric.numeric_schema_digest(),
      numeric_binding=flash_numeric.binding_digest(),
    ),
  )
}
