///|
fn append_conditioner_tensor(
  entries : Array[@safetensors.RawTensorMetadata],
  indexes : Array[MiniMaxH3TensorIndexInput],
  name : String,
  shape : ArrayView[Int64],
  offset : Int64,
) -> Int64 {
  append_named_shape(entries, indexes, name, shape, offset)
}

///|
fn append_language_layer(
  entries : Array[@safetensors.RawTensorMetadata],
  indexes : Array[MiniMaxH3TensorIndexInput],
  layer : Int,
  offset : Int64,
) -> Int64 {
  let prefix = "model.language_model.layers.\{layer}"
  let mut next = offset
  for
    named in [
      ("input_layernorm.weight", [5120L]),
      ("post_attention_layernorm.weight", [5120L]),
      ("self_attn.q_proj.weight", [8192L, 5120L]),
      ("self_attn.k_proj.weight", [1024L, 5120L]),
      ("self_attn.v_proj.weight", [1024L, 5120L]),
      ("self_attn.o_proj.weight", [5120L, 8192L]),
      ("self_attn.q_norm.weight", [128L]),
      ("self_attn.k_norm.weight", [128L]),
      ("mlp.gate_proj.weight", [25600L, 5120L]),
      ("mlp.up_proj.weight", [25600L, 5120L]),
      ("mlp.down_proj.weight", [5120L, 25600L]),
    ] {
    next = append_conditioner_tensor(
      entries,
      indexes,
      "\{prefix}.\{named.0}",
      named.1,
      next,
    )
  }
  next
}

///|
fn append_vision_block(
  entries : Array[@safetensors.RawTensorMetadata],
  indexes : Array[MiniMaxH3TensorIndexInput],
  layer : Int,
  offset : Int64,
) -> Int64 {
  let prefix = "model.visual.blocks.\{layer}"
  let mut next = offset
  for
    named in [
      ("norm1.weight", [1152L]),
      ("norm1.bias", [1152L]),
      ("norm2.weight", [1152L]),
      ("norm2.bias", [1152L]),
      ("attn.qkv.weight", [3456L, 1152L]),
      ("attn.qkv.bias", [3456L]),
      ("attn.proj.weight", [1152L, 1152L]),
      ("attn.proj.bias", [1152L]),
      ("mlp.linear_fc1.weight", [4304L, 1152L]),
      ("mlp.linear_fc1.bias", [4304L]),
      ("mlp.linear_fc2.weight", [1152L, 4304L]),
      ("mlp.linear_fc2.bias", [1152L]),
    ] {
    next = append_conditioner_tensor(
      entries,
      indexes,
      "\{prefix}.\{named.0}",
      named.1,
      next,
    )
  }
  next
}

///|
fn append_merger(
  entries : Array[@safetensors.RawTensorMetadata],
  indexes : Array[MiniMaxH3TensorIndexInput],
  prefix : String,
  offset : Int64,
) -> Int64 {
  let mut next = offset
  for
    named in [
      ("norm.weight", [1152L]),
      ("norm.bias", [1152L]),
      ("linear_fc1.weight", [4608L, 4608L]),
      ("linear_fc1.bias", [4608L]),
      ("linear_fc2.weight", [5120L, 4608L]),
      ("linear_fc2.bias", [5120L]),
    ] {
    next = append_conditioner_tensor(
      entries,
      indexes,
      "\{prefix}.\{named.0}",
      named.1,
      next,
    )
  }
  next
}

///|
fn complete_conditioner_fixture() -> (
  @safetensors.SafetensorsMetadata,
  Array[MiniMaxH3TensorIndexInput],
  Int64,
) {
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let indexes : Array[MiniMaxH3TensorIndexInput] = []
  let mut offset = 0L
  for
    named in [
      ("lm_head.weight", [151936L, 5120L]),
      ("model.language_model.embed_tokens.weight", [151936L, 5120L]),
      ("model.language_model.norm.weight", [5120L]),
      ("model.visual.patch_embed.proj.weight", [1152L, 3L, 2L, 16L, 16L]),
      ("model.visual.patch_embed.proj.bias", [1152L]),
      ("model.visual.pos_embed.weight", [2304L, 1152L]),
    ] {
    offset = append_conditioner_tensor(
      entries,
      indexes,
      named.0,
      named.1,
      offset,
    )
  }
  for layer in 0..<64 {
    offset = append_language_layer(entries, indexes, layer, offset)
  }
  for layer in 0..<27 {
    offset = append_vision_block(entries, indexes, layer, offset)
  }
  offset = append_merger(entries, indexes, "model.visual.merger", offset)
  for merger in 0..<3 {
    offset = append_merger(
      entries,
      indexes,
      "model.visual.deepstack_merger_list.\{merger}",
      offset,
    )
  }
  let limits = try! @safetensors.HeaderLimits::new(
    max_header_bytes=1048576,
    max_tensor_count=1100,
    max_rank=8,
    max_name_length=256,
    max_payload_bytes=1000000000000L,
    max_materialized_tensor_bytes=1000000000000L,
  )
  (
    try! @safetensors.validate_metadata(
      entries,
      encoded_header_bytes=131072,
      payload_bytes=offset,
      limits~,
    ),
    indexes,
    offset,
  )
}

///|
test "complete 1058-tensor Qwen3-VL conditioner builds an exact host manifest" {
  let (tensors, indexes, payload_bytes) = complete_conditioner_fixture()
  assert_eq(tensors.tensor_count(), 1058)
  let metadata = @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
    @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
      variant=Ref2va,
      checkpoint_format=DiffusersV1,
    ),
    try! @spec.ContentDigest::from_sha256(
      "4444444444444444444444444444444444444444444444444444444444444444",
    ),
  )
  let partial = try! @minimax_h3_weights.bind(
    metadata,
    @minimax_h3.build(metadata),
    TextConditioner,
    tensors,
  )
  let binding = try! @minimax_h3_weights.join_complete_component(
    TextConditioner,
    [partial],
  )
  let limits = try! MiniMaxH3HostMaterializationLimits::new(
    maximum_shards=8,
    maximum_tensors=1100,
    maximum_header_bytes=1048576,
    maximum_payload_bytes_per_shard=1000000000000L,
    maximum_total_host_bytes=1000000000000L,
    maximum_arena_bytes=2000000000L,
    maximum_arenas=1024,
    maximum_name_bytes=256,
  )
  let manifest = try! build_manifest(
    binding,
    expected_identity=binding.identity(),
    shards=[
      try! MiniMaxH3ShardDeclarationInput::new(
        file_digest=try! @spec.ContentDigest::from_sha256(
          "5555555555555555555555555555555555555555555555555555555555555555",
        ),
        payload_bytes~,
      ),
    ],
    index=indexes,
    limits~,
  )
  assert_eq(manifest.component(), TextConditioner)
  assert_eq(manifest.tensor_count(), 1058)
  assert_eq(manifest.total_tensor_bytes(), payload_bytes)
  assert_true(manifest.arena_count() > 1)
}
