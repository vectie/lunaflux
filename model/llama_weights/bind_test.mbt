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
  let plan = @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  @spec.LlamaModelMetadata::new(spec, @spec.ModelIdentity::new(content, plan))
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
fn valid_entries() -> (Array[@safetensors.RawTensorMetadata], Int64) {
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
  (entries, offset.val)
}

///|
fn validated(
  entries : ArrayView[@safetensors.RawTensorMetadata],
  payload_bytes : Int64,
) -> @safetensors.SafetensorsMetadata raise {
  let limits = @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=32,
    max_rank=2,
    max_name_length=128,
    max_payload_bytes=4096L,
    max_materialized_tensor_bytes=4096L,
  )
  @safetensors.validate_metadata(
    entries,
    encoded_header_bytes=1024,
    payload_bytes~,
    limits~,
  )
}

///|
test "complete dense Llama weights bind in semantic TensorRef order" {
  let (entries, payload_bytes) = valid_entries()
  let tensors = validated(entries, payload_bytes)
  let bindings = @llama_weights.bind(test_model(), tensors)
  assert_eq(bindings.length(), 12)
  assert_eq(bindings.materialized_bytes(), payload_bytes)
  assert_eq(bindings.identity(), test_model().identity())
  let q_reference = try! @plan.TensorRef::new(2)
  guard bindings.binding(q_reference) is Some(binding) else {
    fail("expected q projection binding")
  }
  assert_eq(binding.reference().as_int(), 2)
  assert_eq(binding.tensor().name(), "model.layers.0.self_attn.q_proj.weight")
}

///|
test "weight bindings cover every TensorRef emitted by the Llama plan" {
  let model = test_model()
  let (entries, payload_bytes) = valid_entries()
  let bindings = @llama_weights.bind(model, validated(entries, payload_bytes))
  let referenced = FixedArray::make(bindings.length(), false)
  for operation in @llama.build(model).operations() {
    for input in operation.inputs() {
      match input {
        Tensor(reference) => referenced[reference.as_int()] = true
        ModelInput(_) | Activation(_) => ()
      }
    }
  }
  for present in referenced {
    assert_true(present)
  }
}

///|
test "tensor count mismatch fails before name lookup" {
  let (entries, payload_bytes) = valid_entries()
  ignore(entries.pop())
  try
    @llama_weights.bind(test_model(), validated(entries, payload_bytes))
  catch {
    @llama_weights.LlamaWeightsError::TensorCountMismatch(
      expected=12,
      actual=11
    ) => ()
    _ => fail("unexpected binding error")
  } noraise {
    _ => fail("expected tensor count mismatch")
  }
}

///|
test "missing and shape-mismatched tensors fail with bounded references" {
  let (missing, payload_bytes) = valid_entries()
  let original = missing[2]
  missing[2] = @safetensors.RawTensorMetadata::new(
    name="unexpected.weight",
    dtype="BF16",
    shape=[4L, 4L],
    data_start=original.data_start,
    data_end=original.data_end,
  )
  try
    @llama_weights.bind(test_model(), validated(missing, payload_bytes))
  catch {
    @llama_weights.LlamaWeightsError::MissingTensor(reference=2) => ()
    _ => fail("unexpected missing tensor error")
  } noraise {
    _ => fail("expected missing tensor")
  }
  let (wrong_shape, wrong_payload) = valid_entries()
  let q = wrong_shape[2]
  wrong_shape[2] = @safetensors.RawTensorMetadata::new(
    name="model.layers.0.self_attn.q_proj.weight",
    dtype="BF16",
    shape=[2L, 8L],
    data_start=q.data_start,
    data_end=q.data_end,
  )
  try
    @llama_weights.bind(test_model(), validated(wrong_shape, wrong_payload))
  catch {
    @llama_weights.LlamaWeightsError::ShapeDimensionMismatch(
      reference=2,
      axis=0
    ) => ()
    _ => fail("unexpected shape error")
  } noraise {
    _ => fail("expected shape mismatch")
  }
}
