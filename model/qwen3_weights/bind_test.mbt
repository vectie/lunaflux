///|
fn test_qwen3_model(
  layer_count? : Int = 1,
  vocabulary_size? : Int = 8,
  hidden_size? : Int = 4,
  intermediate_size? : Int = 6,
  attention_head_count? : Int = 2,
  kv_head_count? : Int = 1,
  head_dimension? : Int = 4,
) -> @qwen3_spec.Qwen3ModelMetadata raise {
  let spec = @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size~,
    hidden_size~,
    intermediate_size~,
    layer_count~,
    attention_head_count~,
    kv_head_count~,
    head_dimension~,
    context_length=128,
    rope_theta=1000000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
  @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
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
  dtype? : String = "BF16",
) -> Unit {
  let start = offset.val
  let end = start + tensor_bytes(shape)
  entries.push(
    @safetensors.RawTensorMetadata::new(
      name~,
      dtype~,
      shape~,
      data_start=start,
      data_end=end,
    ),
  )
  offset.val = end
}

///|
fn qwen3_entries(
  model : @qwen3_spec.Qwen3ModelMetadata,
  redundant_head? : Bool = false,
) -> (Array[@safetensors.RawTensorMetadata], Int64) {
  let spec = model.spec()
  let hidden = Int64::from_int(spec.hidden_size())
  let intermediate = Int64::from_int(spec.intermediate_size())
  let vocabulary = Int64::from_int(spec.vocabulary_size())
  let query_width = Int64::from_int(spec.query_projection_width())
  let kv_width = Int64::from_int(spec.kv_projection_width())
  let head_dimension = Int64::from_int(spec.head_dimension())
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let offset = Ref(0L)
  if redundant_head {
    append_raw(entries, "lm_head.weight", [vocabulary, hidden], offset)
  }
  append_raw(entries, "model.embed_tokens.weight", [vocabulary, hidden], offset)
  for layer in 0..<spec.layer_count() {
    let prefix = "model.layers.\{layer}"
    append_raw(entries, "\{prefix}.input_layernorm.weight", [hidden], offset)
    append_raw(
      entries,
      "\{prefix}.self_attn.q_proj.weight",
      [query_width, hidden],
      offset,
    )
    append_raw(
      entries,
      "\{prefix}.self_attn.k_proj.weight",
      [kv_width, hidden],
      offset,
    )
    append_raw(
      entries,
      "\{prefix}.self_attn.v_proj.weight",
      [kv_width, hidden],
      offset,
    )
    append_raw(
      entries,
      "\{prefix}.self_attn.q_norm.weight",
      [head_dimension],
      offset,
    )
    append_raw(
      entries,
      "\{prefix}.self_attn.k_norm.weight",
      [head_dimension],
      offset,
    )
    append_raw(
      entries,
      "\{prefix}.self_attn.o_proj.weight",
      [hidden, query_width],
      offset,
    )
    append_raw(
      entries,
      "\{prefix}.post_attention_layernorm.weight",
      [hidden],
      offset,
    )
    append_raw(
      entries,
      "\{prefix}.mlp.gate_proj.weight",
      [intermediate, hidden],
      offset,
    )
    append_raw(
      entries,
      "\{prefix}.mlp.up_proj.weight",
      [intermediate, hidden],
      offset,
    )
    append_raw(
      entries,
      "\{prefix}.mlp.down_proj.weight",
      [hidden, intermediate],
      offset,
    )
  }
  append_raw(entries, "model.norm.weight", [hidden], offset)
  (entries, offset.val)
}

///|
fn validated(
  entries : ArrayView[@safetensors.RawTensorMetadata],
  payload_bytes : Int64,
) -> @safetensors.SafetensorsMetadata raise {
  @safetensors.validate_metadata(
    entries,
    encoded_header_bytes=65536,
    payload_bytes~,
    limits=@safetensors.HeaderLimits::new(
      max_header_bytes=131072,
      max_tensor_count=512,
      max_rank=2,
      max_name_length=128,
      max_payload_bytes=2147483648L,
      max_materialized_tensor_bytes=2147483648L,
    ),
  )
}

///|
test "Qwen3 tied output aliases unique embedding storage in exact semantic order" {
  let model = test_qwen3_model()
  let (entries, payload_bytes) = qwen3_entries(model)
  let bindings = @qwen3_weights.bind(model, validated(entries, payload_bytes))
  assert_eq(bindings.length(), 13)
  assert_eq(bindings.content_digest(), model.content_digest())
  assert_eq(bindings.source_tensor_count(), 13)
  assert_eq(bindings.source_bytes(), payload_bytes)
  assert_eq(bindings.materialized_bytes(), payload_bytes)
  let embedding = bindings.embedding_binding()
  let lm_head = bindings.lm_head_binding()
  assert_eq(embedding, lm_head)
  assert_eq(lm_head.reference().as_int(), 0)
  assert_eq(lm_head.tensor().name(), "model.embed_tokens.weight")
  assert_true(bindings.redundant_lm_head_tensor() is None)
  let q_norm = bindings.binding(try! @plan.TensorRef::new(5))
  guard q_norm is Some(binding) else { fail("expected Qwen3 q_norm binding") }
  assert_eq(binding.tensor().name(), "model.layers.0.self_attn.q_norm.weight")
  let k_norm = bindings.binding(try! @plan.TensorRef::new(6))
  guard k_norm is Some(binding) else { fail("expected Qwen3 k_norm binding") }
  assert_eq(binding.tensor().name(), "model.layers.0.self_attn.k_norm.weight")
}

///|
test "Qwen3 canonical tied artifact may omit redundant LM-head bytes" {
  let model = test_qwen3_model()
  let (entries, payload_bytes) = qwen3_entries(model, redundant_head=false)
  let bindings = @qwen3_weights.bind(model, validated(entries, payload_bytes))
  assert_eq(bindings.length(), 13)
  assert_eq(bindings.source_tensor_count(), 13)
  assert_eq(bindings.materialized_bytes(), payload_bytes)
  assert_true(bindings.redundant_lm_head_tensor() is None)
  assert_eq(bindings.embedding_binding(), bindings.lm_head_binding())
}

///|
test "pinned Qwen3-0.6B dimensions bind all 310 unique storage references" {
  let model = test_qwen3_model(
    layer_count=28,
    vocabulary_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    attention_head_count=16,
    kv_head_count=8,
    head_dimension=128,
  )
  let (entries, payload_bytes) = qwen3_entries(model)
  assert_eq(entries.length(), 310)
  assert_eq(payload_bytes, 1192099840L)
  let bindings = @qwen3_weights.bind(model, validated(entries, payload_bytes))
  assert_eq(bindings.length(), 310)
  assert_eq(bindings.source_tensor_count(), 310)
  assert_eq(bindings.materialized_bytes(), 1192099840L)
  guard bindings.binding(try! @plan.TensorRef::new(309)) is Some(final_norm) else {
    fail("expected final norm")
  }
  assert_eq(final_norm.tensor().name(), "model.norm.weight")
}

///|
test "Qwen3 binding order matches the admitted numeric tensor schema" {
  let model = test_qwen3_model()
  let (entries, payload_bytes) = qwen3_entries(model)
  let bindings = @qwen3_weights.bind(model, validated(entries, payload_bytes))
  let schema = model.spec().numeric_schema()
  assert_eq(schema.tensors().length(), bindings.length())
  for descriptor in schema.tensors() {
    let reference = try! @plan.TensorRef::new(descriptor.ordinal().as_int())
    guard bindings.binding(reference) is Some(binding) else {
      fail("numeric schema tensor has no Qwen3 source binding")
    }
    assert_eq(binding.tensor().shape(), descriptor.shape().dimensions())
  }
}

///|
test "Qwen3 binding rejects unknown and missing tensor names" {
  let model = test_qwen3_model()
  let (unknown, payload_bytes) = qwen3_entries(model)
  let original = unknown[5]
  unknown[5] = @safetensors.RawTensorMetadata::new(
    name="model.layers.0.self_attn.unknown_norm.weight",
    dtype="BF16",
    shape=[4L],
    data_start=original.data_start,
    data_end=original.data_end,
  )
  try @qwen3_weights.bind(model, validated(unknown, payload_bytes)) catch {
    @qwen3_weights.Qwen3WeightsError::UnknownTensor(index=5) => ()
    _ => fail("unexpected unknown-tensor error")
  } noraise {
    _ => fail("unknown Qwen3 tensor was admitted")
  }
  let (missing, missing_payload) = qwen3_entries(model, redundant_head=false)
  ignore(missing.pop())
  try @qwen3_weights.bind(model, validated(missing, missing_payload)) catch {
    @qwen3_weights.Qwen3WeightsError::MissingTensor(reference=12) => ()
    _ => fail("unexpected missing-tensor error")
  } noraise {
    _ => fail("missing Qwen3 final norm was admitted")
  }
}

///|
test "Qwen3 binding rejects wrong per-head norm and output shapes" {
  let model = test_qwen3_model()
  let (wrong_norm, payload_bytes) = qwen3_entries(model)
  let norm = wrong_norm[5]
  wrong_norm[5] = @safetensors.RawTensorMetadata::new(
    name="model.layers.0.self_attn.q_norm.weight",
    dtype="BF16",
    shape=[2L, 2L],
    data_start=norm.data_start,
    data_end=norm.data_end,
  )
  try @qwen3_weights.bind(model, validated(wrong_norm, payload_bytes)) catch {
    @qwen3_weights.Qwen3WeightsError::InvalidTensor(reference=5, TensorRank) =>
      ()
    _ => fail("unexpected q_norm shape error")
  } noraise {
    _ => fail("wrong Qwen3 q_norm shape was admitted")
  }
  let (wrong_output, output_payload) = qwen3_entries(model)
  let output = wrong_output[7]
  wrong_output[7] = @safetensors.RawTensorMetadata::new(
    name="model.layers.0.self_attn.o_proj.weight",
    dtype="BF16",
    shape=[8L, 4L],
    data_start=output.data_start,
    data_end=output.data_end,
  )
  try
    @qwen3_weights.bind(model, validated(wrong_output, output_payload))
  catch {
    @qwen3_weights.Qwen3WeightsError::InvalidTensor(
      reference=7,
      TensorDimension
    ) => ()
    _ => fail("unexpected output shape error")
  } noraise {
    _ => fail("transposed Qwen3 output projection was admitted")
  }
  let (wrong_head, head_payload) = qwen3_entries(model, redundant_head=true)
  let head = wrong_head[0]
  wrong_head[0] = @safetensors.RawTensorMetadata::new(
    name="lm_head.weight",
    dtype="BF16",
    shape=[4L, 8L],
    data_start=head.data_start,
    data_end=head.data_end,
  )
  try @qwen3_weights.bind(model, validated(wrong_head, head_payload)) catch {
    @qwen3_weights.Qwen3WeightsError::InvalidTensor(reference=0, TiedOutput) =>
      ()
    _ => fail("unexpected tied-output shape error")
  } noraise {
    _ => fail("wrong Qwen3 tied-output shape was admitted")
  }
}

///|
test "safetensors rejects Qwen3 duplicate dtype and range attacks first" {
  let model = test_qwen3_model()
  let (duplicate, payload_bytes) = qwen3_entries(model)
  let copy = duplicate[0]
  duplicate.push(
    @safetensors.RawTensorMetadata::new(
      name="model.embed_tokens.weight",
      dtype="BF16",
      shape=[8L, 4L],
      data_start=payload_bytes,
      data_end=payload_bytes + 64L,
    ),
  )
  try validated(duplicate, payload_bytes + 64L) catch {
    @safetensors.SafetensorsError::DuplicateTensor(_, _) => ()
    _ => fail("unexpected duplicate error")
  } noraise {
    _ => fail("duplicate Qwen3 tensor was admitted")
  }
  let (wrong_dtype, dtype_payload) = qwen3_entries(model)
  wrong_dtype[0] = @safetensors.RawTensorMetadata::new(
    name=copy.name,
    dtype="F16",
    shape=[8L, 4L],
    data_start=copy.data_start,
    data_end=copy.data_end,
  )
  try validated(wrong_dtype, dtype_payload) catch {
    @safetensors.SafetensorsError::InvalidDType(_) => ()
    _ => fail("unexpected dtype error")
  } noraise {
    _ => fail("non-BF16 Qwen3 tensor was admitted")
  }
  let (outside, outside_payload) = qwen3_entries(model)
  let embedding = outside[0]
  outside[0] = @safetensors.RawTensorMetadata::new(
    name=embedding.name,
    dtype="BF16",
    shape=[8L, 4L],
    data_start=embedding.data_start,
    data_end=outside_payload + 1L,
  )
  try validated(outside, outside_payload) catch {
    @safetensors.SafetensorsError::RangeOutsidePayload(_) => ()
    _ => fail("unexpected range error")
  } noraise {
    _ => fail("out-of-range Qwen3 tensor was admitted")
  }
}
