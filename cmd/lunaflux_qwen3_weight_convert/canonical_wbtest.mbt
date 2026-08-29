///|
fn converter_qwen3_model() -> @qwen3_spec.Qwen3ModelMetadata raise {
  let spec = try! @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    layer_count=28,
    attention_head_count=16,
    kv_head_count=8,
    head_dimension=128,
    context_length=40960,
    rope_theta=1000000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
  @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
    spec,
    try! @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
fn converter_tensor_bytes(shape : ArrayView[Int64]) -> Int64 {
  shape.fold(init=2L, (size, dimension) => size * dimension)
}

///|
fn converter_append_tensor(
  tensors : Array[@safetensors.RawTensorMetadata],
  name : String,
  shape : ArrayView[Int64],
  offset : Ref[Int64],
) -> Unit {
  let start = offset.val
  let end = start + converter_tensor_bytes(shape)
  tensors.push(
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
fn converter_qwen3_bindings(
  model : @qwen3_spec.Qwen3ModelMetadata,
) -> @qwen3_weights.Qwen3WeightBindings raise {
  let spec = model.spec()
  let hidden = spec.hidden_size().to_int64()
  let intermediate = spec.intermediate_size().to_int64()
  let vocabulary = spec.vocabulary_size().to_int64()
  let query_width = spec.query_projection_width().to_int64()
  let kv_width = spec.kv_projection_width().to_int64()
  let head_dimension = spec.head_dimension().to_int64()
  let tensors : Array[@safetensors.RawTensorMetadata] = []
  let offset = Ref(0L)
  converter_append_tensor(
    tensors,
    "model.embed_tokens.weight",
    [vocabulary, hidden],
    offset,
  )
  for layer in 0..<spec.layer_count() {
    let prefix = "model.layers.\{layer}"
    converter_append_tensor(
      tensors,
      "\{prefix}.input_layernorm.weight",
      [hidden],
      offset,
    )
    converter_append_tensor(
      tensors,
      "\{prefix}.self_attn.q_proj.weight",
      [query_width, hidden],
      offset,
    )
    converter_append_tensor(
      tensors,
      "\{prefix}.self_attn.k_proj.weight",
      [kv_width, hidden],
      offset,
    )
    converter_append_tensor(
      tensors,
      "\{prefix}.self_attn.v_proj.weight",
      [kv_width, hidden],
      offset,
    )
    converter_append_tensor(
      tensors,
      "\{prefix}.self_attn.q_norm.weight",
      [head_dimension],
      offset,
    )
    converter_append_tensor(
      tensors,
      "\{prefix}.self_attn.k_norm.weight",
      [head_dimension],
      offset,
    )
    converter_append_tensor(
      tensors,
      "\{prefix}.self_attn.o_proj.weight",
      [hidden, query_width],
      offset,
    )
    converter_append_tensor(
      tensors,
      "\{prefix}.post_attention_layernorm.weight",
      [hidden],
      offset,
    )
    converter_append_tensor(
      tensors,
      "\{prefix}.mlp.gate_proj.weight",
      [intermediate, hidden],
      offset,
    )
    converter_append_tensor(
      tensors,
      "\{prefix}.mlp.up_proj.weight",
      [intermediate, hidden],
      offset,
    )
    converter_append_tensor(
      tensors,
      "\{prefix}.mlp.down_proj.weight",
      [hidden, intermediate],
      offset,
    )
  }
  converter_append_tensor(tensors, "model.norm.weight", [hidden], offset)
  assert_eq(tensors.length(), 310)
  assert_eq(offset.val, 1192099840L)
  let metadata = try! @safetensors.validate_metadata(
    tensors,
    encoded_header_bytes=65536,
    payload_bytes=offset.val,
    limits=try! @safetensors.HeaderLimits::new(
      max_header_bytes=131072,
      max_tensor_count=311,
      max_rank=2,
      max_name_length=160,
      max_payload_bytes=4294967296L,
      max_materialized_tensor_bytes=4294967296L,
    ),
  )
  try! @qwen3_weights.bind(model, metadata)
}

///|
test "converter emits the exact 310-tensor canonical ordinal header and regions" {
  let model = converter_qwen3_model()
  let bindings = converter_qwen3_bindings(model)
  let plan = try! @qwen3.build_paged(model)
  let conversion = prepare_qwen3_numeric_conversion(plan, bindings)
  assert_eq(conversion.payload_bytes, 1192099840L)
  assert_eq(conversion.source_regions.length(), 310 * 16)
  assert_eq(conversion.header.length() % 8, 0)
  let header = try! @utf8.decode(conversion.header)
  assert_true(header.has_prefix("{\"__metadata__\""))
  assert_true(header.contains("\"lunaflux.tensor.v1.0\""))
  assert_true(header.contains("\"lunaflux.tensor.v1.309\""))
  assert_false(header.contains("lm_head.weight"))
  assert_true(header.has_suffix(" "))
  let artifact = "a".repeat(64)
  let first = qwen3_route_manifest_sha256(plan, bindings, artifact)
  let second = qwen3_route_manifest_sha256(plan, bindings, artifact)
  assert_eq(first, second)
  assert_eq(first.length(), 64)
}
