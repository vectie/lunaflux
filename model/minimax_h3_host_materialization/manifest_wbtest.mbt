///|
fn append_named_shape(
  entries : Array[@safetensors.RawTensorMetadata],
  indexes : Array[MiniMaxH3TensorIndexInput],
  name : String,
  shape : ArrayView[Int64],
  offset : Int64,
) -> Int64 {
  let mut elements = 1L
  for dimension in shape {
    elements *= dimension
  }
  let f32 = name == "proj_in.weight" ||
    name == "proj_in.bias" ||
    name == "audio_proj_in.weight" ||
    name == "audio_proj_in.bias" ||
    name == "proj_out.weight" ||
    name == "proj_out.bias" ||
    name == "audio_proj_out.weight" ||
    name == "audio_proj_out.bias" ||
    name.has_prefix("time_embedder.")
  let byte_length = elements * (if f32 { 4L } else { 2L })
  entries.push(
    @safetensors.RawTensorMetadata::new(
      name~,
      dtype=if f32 { "F32" } else { "BF16" },
      shape~,
      data_start=offset,
      data_end=offset + byte_length,
    ),
  )
  indexes.push(MiniMaxH3TensorIndexInput::new(tensor_name=name, shard_index=0))
  offset + byte_length
}

///|
fn denoiser_block_shape(suffix : StringView) -> Array[Int64] {
  match suffix {
    "adaln_proj.linear.weight" => [96768L, 2688L]
    "adaln_proj.linear.bias" => [96768L]
    "attn.norm_k.weight" | "attn.norm_q.weight" => [128L]
    "attn.to_k.weight" | "attn.to_q.weight" | "attn.to_v.weight" =>
      [7168L, 5376L]
    "attn.to_out.0.weight" => [5376L, 7168L]
    "ff.net.0.proj.weight" => [28672L, 5376L]
    "ff.net.2.weight" => [5376L, 14336L]
    "norm1.weight" | "norm2.weight" => [5376L]
    _ => abort("unknown denoiser fixture suffix")
  }
}

///|
fn append_denoiser_block(
  entries : Array[@safetensors.RawTensorMetadata],
  indexes : Array[MiniMaxH3TensorIndexInput],
  prefix : String,
  refiner : Bool,
  offset : Int64,
) -> Int64 {
  let suffixes = if refiner {
    [
      "attn.norm_k.weight", "attn.norm_q.weight", "attn.to_k.weight", "attn.to_out.0.weight",
      "attn.to_q.weight", "attn.to_v.weight", "ff.net.0.proj.weight", "ff.net.2.weight",
      "norm1.weight", "norm2.weight",
    ]
  } else {
    [
      "adaln_proj.linear.bias", "adaln_proj.linear.weight", "attn.norm_k.weight",
      "attn.norm_q.weight", "attn.to_k.weight", "attn.to_out.0.weight", "attn.to_q.weight",
      "attn.to_v.weight", "ff.net.0.proj.weight", "ff.net.2.weight", "norm1.weight",
      "norm2.weight",
    ]
  }
  let mut next = offset
  for suffix in suffixes {
    next = append_named_shape(
      entries,
      indexes,
      "\{prefix}.\{suffix}",
      denoiser_block_shape(suffix),
      next,
    )
  }
  next
}

///|
fn complete_denoiser_fixture() -> (
  @safetensors.SafetensorsMetadata,
  Array[MiniMaxH3TensorIndexInput],
  Int64,
) {
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let indexes : Array[MiniMaxH3TensorIndexInput] = []
  let mut offset = 0L
  for
    fixed in [
      ("audio_proj_in.weight", [5376L, 32L]),
      ("audio_proj_in.bias", [5376L]),
      ("audio_proj_out.weight", [32L, 5376L]),
      ("audio_proj_out.bias", [32L]),
      ("context_embedder.weight", [5376L, 5120L]),
      ("context_embedder.bias", [5376L]),
      ("proj_in.weight", [5376L, 96L]),
      ("proj_in.bias", [5376L]),
      ("proj_out.weight", [96L, 5376L]),
      ("proj_out.bias", [96L]),
      ("time_embedder.linear_1.weight", [5376L, 256L]),
      ("time_embedder.linear_1.bias", [5376L]),
      ("time_embedder.linear_2.weight", [2688L, 5376L]),
      ("time_embedder.linear_2.bias", [2688L]),
      ("norm_out.linear.weight", [10752L, 2688L]),
      ("norm_out.linear.bias", [10752L]),
      ("norm_out.norm.weight", [5376L]),
      ("token_refiner.final_norm.weight", [5376L]),
    ] {
    offset = append_named_shape(entries, indexes, fixed.0, fixed.1, offset)
  }
  for layer in 0..<2 {
    offset = append_denoiser_block(
      entries,
      indexes,
      "token_refiner.refiner_blocks.\{layer}",
      true,
      offset,
    )
  }
  for layer in 0..<50 {
    offset = append_denoiser_block(
      entries,
      indexes,
      "transformer_blocks.\{layer}",
      false,
      offset,
    )
  }
  let limits = try! @safetensors.HeaderLimits::new(
    max_header_bytes=1048576,
    max_tensor_count=700,
    max_rank=8,
    max_name_length=256,
    max_payload_bytes=1000000000000L,
    max_materialized_tensor_bytes=1000000000000L,
  )
  (
    try! @safetensors.validate_metadata(
      entries,
      encoded_header_bytes=65536,
      payload_bytes=offset,
      limits~,
    ),
    indexes,
    offset,
  )
}

///|
fn complete_denoiser_binding(
  tensors : @safetensors.SafetensorsMetadata,
) -> @minimax_h3_weights.MiniMaxH3CompleteComponentBinding {
  let metadata = @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
    @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
      variant=Ref2va,
      checkpoint_format=DiffusersV1,
    ),
    try! @spec.ContentDigest::from_sha256(
      "2222222222222222222222222222222222222222222222222222222222222222",
    ),
  )
  let partial = try! @minimax_h3_weights.bind(
    metadata,
    @minimax_h3.build(metadata),
    Ref2vaDenoiser,
    tensors,
  )
  try! @minimax_h3_weights.join_complete_component(Ref2vaDenoiser, [partial])
}

///|
fn complete_manifest_limits() -> MiniMaxH3HostMaterializationLimits {
  try! MiniMaxH3HostMaterializationLimits::new(
    maximum_shards=8,
    maximum_tensors=700,
    maximum_header_bytes=1048576,
    maximum_payload_bytes_per_shard=1000000000000L,
    maximum_total_host_bytes=1000000000000L,
    maximum_arena_bytes=1073741824L,
    maximum_arenas=1024,
    maximum_name_bytes=256,
  )
}

///|
test "complete 638-tensor denoiser manifest joins exact membership and layout" {
  let (tensors, indexes, payload_bytes) = complete_denoiser_fixture()
  assert_eq(tensors.tensor_count(), 638)
  let binding = complete_denoiser_binding(tensors)
  let manifest = try! build_manifest(
    binding,
    expected_identity=binding.identity(),
    shards=[
      try! MiniMaxH3ShardDeclarationInput::new(
        file_digest=try! @spec.ContentDigest::from_sha256(
          "3333333333333333333333333333333333333333333333333333333333333333",
        ),
        payload_bytes~,
      ),
    ],
    index=indexes,
    limits=complete_manifest_limits(),
  )
  assert_eq(manifest.component(), Ref2vaDenoiser)
  assert_eq(manifest.tensor_count(), 638)
  assert_eq(manifest.shard_count(), 1)
  assert_eq(manifest.total_tensor_bytes(), payload_bytes)
  assert_true(manifest.arena_count() > 1)
  assert_eq(manifest.identity(), binding.identity())
}

///|
test "complete denoiser manifest rejects duplicate missing and foreign membership" {
  let (tensors, indexes, payload_bytes) = complete_denoiser_fixture()
  let binding = complete_denoiser_binding(tensors)
  let shard = try! MiniMaxH3ShardDeclarationInput::new(
    file_digest=try! @spec.ContentDigest::from_sha256(
      "3333333333333333333333333333333333333333333333333333333333333333",
    ),
    payload_bytes~,
  )
  let missing = indexes.copy()
  ignore(missing.pop())
  try
    build_manifest(
      binding,
      expected_identity=binding.identity(),
      shards=[shard],
      index=missing,
      limits=complete_manifest_limits(),
    )
  catch {
    InvalidIndexCount(expected=638, actual=637) => ()
    _ => fail("unexpected missing-index error")
  } noraise {
    _ => fail("missing index admitted")
  }
  let duplicated = indexes.copy()
  duplicated[637] = duplicated[0]
  try
    build_manifest(
      binding,
      expected_identity=binding.identity(),
      shards=[shard],
      index=duplicated,
      limits=complete_manifest_limits(),
    )
  catch {
    DuplicateIndex(0) => ()
    _ => fail("unexpected duplicate-index error")
  } noraise {
    _ => fail("duplicate index admitted")
  }
  let foreign = indexes.copy()
  foreign[0] = MiniMaxH3TensorIndexInput::new(
    tensor_name=foreign[0].tensor_name,
    shard_index=1,
  )
  try
    build_manifest(
      binding,
      expected_identity=binding.identity(),
      shards=[shard],
      index=foreign,
      limits=complete_manifest_limits(),
    )
  catch {
    ShardIndexOutsideManifest(0) => ()
    _ => fail("unexpected foreign-shard error")
  } noraise {
    _ => fail("foreign shard membership admitted")
  }
}
