///|
fn audio_vae_manifest_fixture() -> MiniMaxH3HostComponentManifest {
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  let raw : Array[@safetensors.RawTensorMetadata] = []
  let index : Array[MiniMaxH3TensorIndexInput] = []
  let mut offset = 0L
  for tensor_index in 0..<schema.tensor_count() {
    guard schema.tensor_name(tensor_index) is Some(name) else {
      abort("missing AudioVae name")
    }
    guard schema.tensor_shape(tensor_index) is Some(shape) else {
      abort("missing AudioVae shape")
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
  if offset != schema.tensor_bytes() {
    abort("AudioVae fixture byte mismatch")
  }
  let tensors = try! @safetensors.validate_metadata(
    raw,
    encoded_header_bytes=schema.header_bytes(),
    payload_bytes=offset,
    limits=try! @safetensors.HeaderLimits::new(
      max_header_bytes=1048576,
      max_tensor_count=1087,
      max_rank=3,
      max_name_length=128,
      max_payload_bytes=700000000L,
      max_materialized_tensor_bytes=700000000L,
    ),
  )
  let metadata = video_vae_model_metadata()
  let plan = try! @minimax_h3.build(metadata)
  let partial = try! @minimax_h3_weights.bind(metadata, plan, AudioVae, tensors)
  let complete = try! @minimax_h3_weights.join_complete_component(AudioVae, [
    partial,
  ])
  try! build_manifest(
    complete,
    expected_identity=complete.identity(),
    shards=[
      try! MiniMaxH3ShardDeclarationInput::new(
        file_digest=try! @spec.ContentDigest::from_sha256(
          "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        ),
        payload_bytes=offset,
      ),
    ],
    index~,
    limits=try! MiniMaxH3HostMaterializationLimits::new(
      maximum_shards=1,
      maximum_tensors=1087,
      maximum_header_bytes=1048576,
      maximum_payload_bytes_per_shard=700000000L,
      maximum_total_host_bytes=606000000L,
      maximum_arena_bytes=200000000L,
      maximum_arenas=4,
      maximum_name_bytes=128,
    ),
  )
}

///|
test "complete AudioVae metadata reaches bounded host manifest admission" {
  let manifest = audio_vae_manifest_fixture()
  assert_eq(manifest.component(), AudioVae)
  assert_eq(manifest.tensor_count(), 1087)
  assert_eq(manifest.total_tensor_bytes(), 605306340L)
  assert_true(manifest.arena_count() >= 3)
  assert_true(manifest.arena_count() <= 4)
  validate_materialization_admission(
    manifest,
    manifest.identity(),
    manifest.digest(),
    try! MiniMaxH3HostMaterializationLimits::new(
      maximum_shards=1,
      maximum_tensors=1087,
      maximum_header_bytes=1048576,
      maximum_payload_bytes_per_shard=700000000L,
      maximum_total_host_bytes=606000000L,
      maximum_arena_bytes=200000000L,
      maximum_arenas=4,
      maximum_name_bytes=128,
    ),
  )
}
