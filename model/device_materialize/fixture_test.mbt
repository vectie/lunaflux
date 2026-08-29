///|
const TEST_HEADER_BYTES : Int = 16

///|
fn test_model() -> @spec.LlamaModelMetadata raise {
  let spec = @spec.LlamaModelSpec::new(
    vocabulary_size=8,
    hidden_size=4,
    intermediate_size=6,
    layer_count=1,
    attention_head_count=2,
    kv_head_count=1,
    context_length=16,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let content = @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  @spec.LlamaModelMetadata::from_verified_content(spec, content)
}

///|
fn tensor_bytes(shape : ArrayView[Int64]) -> Int64 {
  shape.fold(init=2L, (size, dimension) => size * dimension)
}

///|
fn append_raw(
  entries : Array[@safetensors.RawTensorMetadata],
  name : String,
  shape : ArrayView[Int64],
  offset : Ref[Int64],
) -> Unit {
  let start = offset.val
  let end = start + tensor_bytes(shape)
  entries.push(
    @safetensors.RawTensorMetadata::new(
      name~,
      dtype="BF16",
      shape~,
      data_start=start,
      data_end=end,
    ),
  )
  offset.val = end
}

///|
fn test_bindings() -> @llama_weights.LlamaWeightBindings raise {
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let offset = Ref(0L)
  append_raw(entries, "model.embed_tokens.weight", [8L, 4L], offset)
  append_raw(entries, "model.layers.0.input_layernorm.weight", [4L], offset)
  append_raw(
    entries,
    "model.layers.0.self_attn.q_proj.weight",
    [4L, 4L],
    offset,
  )
  append_raw(
    entries,
    "model.layers.0.self_attn.k_proj.weight",
    [2L, 4L],
    offset,
  )
  append_raw(
    entries,
    "model.layers.0.self_attn.v_proj.weight",
    [2L, 4L],
    offset,
  )
  append_raw(
    entries,
    "model.layers.0.self_attn.o_proj.weight",
    [4L, 4L],
    offset,
  )
  append_raw(
    entries,
    "model.layers.0.post_attention_layernorm.weight",
    [4L],
    offset,
  )
  append_raw(entries, "model.layers.0.mlp.gate_proj.weight", [6L, 4L], offset)
  append_raw(entries, "model.layers.0.mlp.up_proj.weight", [6L, 4L], offset)
  append_raw(entries, "model.layers.0.mlp.down_proj.weight", [4L, 6L], offset)
  append_raw(entries, "model.norm.weight", [4L], offset)
  append_raw(entries, "lm_head.weight", [8L, 4L], offset)
  let limits = @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=32,
    max_rank=2,
    max_name_length=128,
    max_payload_bytes=4096L,
    max_materialized_tensor_bytes=4096L,
  )
  let metadata = @safetensors.validate_metadata(
    entries,
    encoded_header_bytes=TEST_HEADER_BYTES,
    payload_bytes=offset.val,
    limits~,
  )
  let model = test_model()
  let plan = @llama.build(model)
  @llama_weights.bind(model, plan, metadata)
}
