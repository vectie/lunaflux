///|
fn video_vae_model_metadata() -> @minimax_h3_spec.MiniMaxH3ModelMetadata {
  @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
    @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
      variant=Ref2va,
      checkpoint_format=DiffusersV1,
    ),
    try! @spec.ContentDigest::from_sha256(
      "4444444444444444444444444444444444444444444444444444444444444444",
    ),
  )
}

///|
fn video_vae_manifest_fixture(
  original? : Bool = false,
) -> MiniMaxH3HostComponentManifest {
  let schema = @minimax_h3_weights.MiniMaxH3VideoVaeSchema::official_v1()
  let source = @minimax_h3_weights.MiniMaxH3OriginalVideoVaeSchema::official_v1()
  let count = if original {
    source.tensor_count()
  } else {
    schema.tensor_count()
  }
  let raw : Array[@safetensors.RawTensorMetadata] = []
  let index : Array[MiniMaxH3TensorIndexInput] = []
  let mut offset = 0L
  for tensor_index in 0..<count {
    guard (if original {
        source.tensor_name(tensor_index)
      } else {
        schema.tensor_name(tensor_index)
      })
      is Some(name) else {
      abort("missing video-VAE name")
    }
    guard (if original {
        source.tensor_shape(tensor_index)
      } else {
        schema.tensor_shape(tensor_index)
      })
      is Some(shape) else {
      abort("missing video-VAE shape")
    }
    let mut elements = 1L
    for dimension in shape {
      elements *= dimension
    }
    let next = offset + elements * 4L
    raw.push(
      @safetensors.RawTensorMetadata::new(
        name~,
        dtype="F32",
        shape~,
        data_start=offset,
        data_end=next,
      ),
    )
    index.push(MiniMaxH3TensorIndexInput::new(tensor_name=name, shard_index=0))
    offset = next
  }
  if offset != (if original { 10415484128L } else { schema.tensor_bytes() }) {
    abort("video-VAE fixture byte mismatch")
  }
  let tensors = try! @safetensors.validate_metadata(
    raw,
    encoded_header_bytes=131072,
    payload_bytes=offset,
    limits=try! @safetensors.HeaderLimits::new(
      max_header_bytes=1048576,
      max_tensor_count=703,
      max_rank=5,
      max_name_length=128,
      max_payload_bytes=11000000000L,
      max_materialized_tensor_bytes=11000000000L,
    ),
  )
  let metadata = video_vae_model_metadata()
  let plan = try! @minimax_h3.build(metadata)
  let partial = try! @minimax_h3_weights.bind(metadata, plan, VideoVae, tensors)
  let complete = try! @minimax_h3_weights.join_complete_component(VideoVae, [
    partial,
  ])
  let identity = try! @spec.ModelIdentity::new(
    metadata.content_digest(),
    plan.plan_digest(),
  )
  try! build_manifest(
    complete,
    expected_identity=identity,
    shards=[
      try! MiniMaxH3ShardDeclarationInput::new(
        file_digest=try! @spec.ContentDigest::from_sha256(
          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        ),
        payload_bytes=offset,
      ),
    ],
    index~,
    limits=try! MiniMaxH3HostMaterializationLimits::new(
      maximum_shards=3,
      maximum_tensors=703,
      maximum_header_bytes=1048576,
      maximum_payload_bytes_per_shard=11000000000L,
      maximum_total_host_bytes=10416000000L,
      maximum_arena_bytes=2000000000L,
      maximum_arenas=8,
      maximum_name_bytes=128,
    ),
  )
}

///|
test "original VideoVAE manifest covers fused QKV and mask buffer without dropping bytes" {
  let original = video_vae_manifest_fixture(original=true)
  let converted = video_vae_manifest_fixture()
  assert_eq(original.tensor_count(), 560)
  assert_eq(
    original.total_tensor_bytes(),
    converted.total_tensor_bytes() + 8192L,
  )
  assert_true(original.digest() != converted.digest())
}

///|
test "complete video-VAE metadata reaches bounded host manifest admission" {
  let manifest = video_vae_manifest_fixture()
  assert_eq(manifest.component(), VideoVae)
  assert_eq(manifest.tensor_count(), 703)
  assert_eq(manifest.total_tensor_bytes(), 10415475936L)
  assert_true(manifest.arena_count() >= 6)
  assert_true(manifest.arena_count() <= 8)
}

///|
test "video-VAE manifest remains component-scoped metadata authority" {
  let manifest = video_vae_manifest_fixture()
  validate_materialization_admission(
    manifest,
    manifest.identity(),
    manifest.digest(),
    try! MiniMaxH3HostMaterializationLimits::new(
      maximum_shards=3,
      maximum_tensors=703,
      maximum_header_bytes=1048576,
      maximum_payload_bytes_per_shard=11000000000L,
      maximum_total_host_bytes=10416000000L,
      maximum_arena_bytes=2000000000L,
      maximum_arenas=8,
      maximum_name_bytes=128,
    ),
  )
  assert_eq(manifest.component(), VideoVae)
}
