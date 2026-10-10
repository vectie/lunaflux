///|
test "distinct shard ordinals may share a declared checkpoint label" {
  let label = fixture_digest(b"declared checkpoint")
  let shards = [
    MiniMaxH3ShardDeclarationInput::new(file_digest=label, payload_bytes=6L),
    MiniMaxH3ShardDeclarationInput::new(file_digest=label, payload_bytes=7L),
  ]
  validate_shard_declarations(shards, fixture_limits())
  try
    validate_shard_declarations(
      [
        shards[0],
        MiniMaxH3ShardDeclarationInput::new(
          file_digest=label,
          payload_bytes=8193L,
        ),
      ],
      fixture_limits(),
    )
  catch {
    ShardPayloadMismatch(1) => ()
    _ => fail("wrong shard-size failure")
  } noraise {
    _ => fail("oversized shard was accepted")
  }
}

///|
fn fixture_limits() -> MiniMaxH3HostMaterializationLimits {
  try! MiniMaxH3HostMaterializationLimits::new(
    maximum_shards=4,
    maximum_tensors=8,
    maximum_header_bytes=4096,
    maximum_payload_bytes_per_shard=8192L,
    maximum_total_host_bytes=8192L,
    maximum_arena_bytes=1024L,
    maximum_arenas=8,
    maximum_name_bytes=128,
  )
}

///|
fn fixture_envelope(header : StringView, payload : ArrayView[Byte]) -> Bytes {
  let encoded = @utf8.encode(header)
  let output : Array[Byte] = []
  let length = encoded.length().to_uint64()
  for shift in 0..<8 {
    output.push((length >> (shift * 8)).to_byte())
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
fn fixture_header() -> String {
  (
    #|{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}}
  )
}

///|
fn fixture_digest(bytes : Bytes) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn fixture_identity(foreign? : Bool = false) -> @spec.ModelIdentity {
  let content = if foreign {
    "1111111111111111111111111111111111111111111111111111111111111111"
  } else {
    "0000000000000000000000000000000000000000000000000000000000000000"
  }
  try! @spec.ModelIdentity::new(
    try! @spec.ContentDigest::from_sha256(content),
    try! @spec.PlanDigest::from_sha256(
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    ),
  )
}

///|
fn fixture_metadata(envelope : Bytes) -> @safetensors.SafetensorsMetadata {
  let limits = try! @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=8,
    max_rank=8,
    max_name_length=128,
    max_payload_bytes=8192L,
    max_materialized_tensor_bytes=8192L,
  )
  try! @safetensors_reader.read_metadata(envelope, limits~)
}

///|
fn fixture_manifest(envelope : Bytes) -> MiniMaxH3HostComponentManifest {
  let metadata = fixture_metadata(envelope)
  guard metadata.find_tensor("a") is Some(a) else { abort("missing a") }
  guard metadata.find_tensor("b") is Some(b) else { abort("missing b") }
  let identity = fixture_identity()
  let shards = ReadOnlyArray::from_array([
    try! MiniMaxH3ShardDeclarationInput::new(
      file_digest=fixture_digest(envelope),
      payload_bytes=6L,
    ),
  ])
  let tensors = ReadOnlyArray::from_array([
    {
      binding_index: 0,
      role: DenoiserProjection,
      tensor: a,
      shard_index: 0,
      arena_index: 0,
      arena_offset: 0L,
    },
    {
      binding_index: 1,
      role: DenoiserAttention,
      tensor: b,
      shard_index: 0,
      arena_index: 0,
      arena_offset: 64L,
    },
  ])
  let arena_sizes = ReadOnlyArray::from_array([66L])
  let digest = canonical_manifest_digest(
    Ref2vaDenoiser,
    identity,
    shards,
    tensors,
    arena_sizes,
    6L,
  )
  {
    component: Ref2vaDenoiser,
    identity,
    digest,
    shards,
    tensors,
    arena_sizes,
    total_tensor_bytes: 6L,
  }
}

///|
async test "streaming adapter materializes once and releases without scrubbing" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-minimax-h3-streaming")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let envelope = fixture_envelope(fixture_header(), [1, 2, 3, 4, 5, 6])
    @fs.write_file(canonical + "/weights", envelope, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let manifest = fixture_manifest(envelope)
    let host = materialize_safetensors_shards(
      root,
      [
        MiniMaxH3ShardSource::new(
          locator=try! @approved_fs.ApprovedRelativeLocator::new("weights"),
        ),
      ],
      manifest,
      expected_identity=manifest.identity,
      expected_manifest_digest=manifest.digest,
      limits=fixture_limits(),
    )
    guard host.arenas[0] is Some(captured) else { abort("missing arena") }
    assert_eq(host.tensor_byte(0, 0L), b'\x01')
    assert_eq(host.tensor_byte(0, 3L), b'\x04')
    assert_eq(host.tensor_byte(1, 0L), b'\x05')
    assert_eq(host.tensor_byte(1, 1L), b'\x06')
    assert_true(host.release())
    assert_true(host.is_released())
    assert_eq(captured[0], b'\x01')
    assert_true(host.arenas.iter().all(arena => arena is None))
    assert_false(host.release())
    try host.tensor_byte(0, 0L) catch {
      ArenaReleased => ()
      _ => fail("unexpected released-state error")
    } noraise {
      _ => fail("released arena remained readable")
    }
    root.close()
  }
}

///|
async test "adapter rejects authenticated foreign streaming metadata" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-minimax-h3-foreign")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let expected = fixture_envelope(fixture_header(), [1, 2, 3, 4, 5, 6])
    let foreign = fixture_envelope(
      (
        #|{"a":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}}
      ),
      [1, 2, 3, 4, 5, 6],
    )
    @fs.write_file(canonical + "/foreign", foreign, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let original = fixture_manifest(expected)
    let foreign_shards = ReadOnlyArray::from_array([
      try! MiniMaxH3ShardDeclarationInput::new(
        file_digest=fixture_digest(foreign),
        payload_bytes=6L,
      ),
    ])
    let foreign_manifest_digest = canonical_manifest_digest(
      original.component,
      original.identity,
      foreign_shards,
      original.tensors,
      original.arena_sizes,
      original.total_tensor_bytes,
    )
    let forged : MiniMaxH3HostComponentManifest = {
      component: original.component,
      identity: original.identity,
      digest: foreign_manifest_digest,
      shards: foreign_shards,
      tensors: original.tensors,
      arena_sizes: original.arena_sizes,
      total_tensor_bytes: original.total_tensor_bytes,
    }
    let input = @streaming_safetensors.StreamingSafetensorsShardInput::new(
      locator=try! @approved_fs.ApprovedRelativeLocator::new("foreign"),
      expected_digest=try! @streaming_safetensors.StreamingSafetensorsFileDigest::from_sha256(
        fixture_digest(foreign).as_hex(),
      ),
    )
    let inspection = @streaming_safetensors.inspect_shards(
      root,
      [input],
      dtype_specs=[
        @streaming_safetensors.StreamingSafetensorsDTypeSpec::bfloat16(),
        @streaming_safetensors.StreamingSafetensorsDTypeSpec::float32(),
      ],
      limits=try! streaming_limits(fixture_limits()),
    )
    try validate_streaming_inspection(inspection, forged) catch {
      StreamingTensorMismatch(0) => ()
      _ => fail("unexpected foreign metadata rejection")
    } noraise {
      _ => fail("foreign streaming shape was admitted")
    }
    root.close()
  }
}

///|
test "identity manifest and VAE component boundaries remain exact" {
  let envelope = fixture_envelope(fixture_header(), [1, 2, 3, 4, 5, 6])
  let manifest = fixture_manifest(envelope)
  validate_materialization_admission(
    manifest,
    manifest.identity,
    manifest.digest,
    fixture_limits(),
  )
  try
    validate_materialization_admission(
      manifest,
      fixture_identity(foreign=true),
      manifest.digest,
      fixture_limits(),
    )
  catch {
    IdentityMismatch => ()
    _ => fail("unexpected identity mismatch")
  } noraise {
    _ => fail("foreign identity admitted")
  }
  let forged : MiniMaxH3HostManifestDigest = {
    hex: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  }
  try
    validate_materialization_admission(
      manifest,
      manifest.identity,
      forged,
      fixture_limits(),
    )
  catch {
    ManifestDigestMismatch => ()
    _ => fail("unexpected manifest mismatch")
  } noraise {
    _ => fail("foreign manifest digest admitted")
  }
  require_materializable_component(VideoVae)
  require_materializable_component(AudioVae)
}

///|
test "materialization consumes manifest labels without rehashing" {
  let envelope = fixture_envelope(fixture_header(), [1, 2, 3, 4, 5, 6])
  let original = fixture_manifest(envelope)
  let label : MiniMaxH3HostManifestDigest = {
    hex: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  }
  // A matching opaque label associates the already-built immutable manifest.
  // It deliberately does not assert a checksum of its contents or payloads.
  let manifest : MiniMaxH3HostComponentManifest = { ..original, digest: label, }
  validate_materialization_admission(
    manifest,
    manifest.identity,
    label,
    fixture_limits(),
  )
  // Removing authentication must not remove allocation bounds.
  let oversized : MiniMaxH3HostComponentManifest = {
    ..manifest,
    arena_sizes: ReadOnlyArray::from_array([2147483648L]),
  }
  try
    validate_materialization_admission(
      oversized,
      oversized.identity,
      label,
      fixture_limits(),
    )
  catch {
    HostBudgetExceeded => ()
    _ => fail("unexpected budget error")
  } noraise {
    _ => fail("oversized arena admitted")
  }
}

///|
test "official Qwen3-VL shapes remain exact" {
  assert_true(
    exact_official_shape(
      TextConditioner,
      "model.language_model.layers.63.self_attn.q_proj.weight",
      ReadOnlyArray::from_array([8192L, 5120L]),
    ),
  )
  assert_false(
    exact_official_shape(
      TextConditioner,
      "model.language_model.layers.64.self_attn.q_proj.weight",
      ReadOnlyArray::from_array([8192L, 5120L]),
    ),
  )
  assert_false(
    exact_official_shape(
      TextConditioner,
      "model.language_model.layers.63.self_attn.q_proj.weight",
      ReadOnlyArray::from_array([8192L, 4096L]),
    ),
  )
}

///|
test "official AudioVae host shapes remain exact" {
  assert_true(
    exact_official_shape(
      AudioVae,
      "decoder.resblocks.20.convs2.2.weight_v",
      ReadOnlyArray::from_array([8L, 8L, 11L]),
    ),
  )
  assert_false(
    exact_official_shape(
      AudioVae,
      "decoder.resblocks.21.convs2.2.weight_v",
      ReadOnlyArray::from_array([8L, 8L, 11L]),
    ),
  )
  assert_false(
    exact_official_shape(
      AudioVae,
      "decoder.resblocks.20.convs2.2.weight_v",
      ReadOnlyArray::from_array([8L, 8L, 7L]),
    ),
  )
}
