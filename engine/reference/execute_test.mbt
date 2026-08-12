///|
fn test_model() -> @spec.LlamaModelMetadata raise {
  let spec = @spec.LlamaModelSpec::new(
    vocabulary_size=3,
    hidden_size=2,
    intermediate_size=2,
    layer_count=1,
    attention_head_count=1,
    kv_head_count=1,
    context_length=4,
    rope_theta=10000.0,
    rms_norm_epsilon=0.000001,
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
fn append_bfloat16(output : Array[Byte], values : ArrayView[Float]) -> Unit {
  for value in values {
    let bits = value.reinterpret_as_uint()
    output.push((bits >> 16).to_byte())
    output.push((bits >> 24).to_byte())
  }
}

///|
fn append_tensor(
  entries : Array[@safetensors.RawTensorMetadata],
  payload : Array[Byte],
  name : String,
  shape : ArrayView[Int64],
  values : ArrayView[Float],
) -> Unit {
  let start = payload.length().to_int64()
  append_bfloat16(payload, values)
  entries.push(
    @safetensors.RawTensorMetadata::new(
      name~,
      dtype="BF16",
      shape~,
      data_start=start,
      data_end=payload.length().to_int64(),
    ),
  )
}

///|
fn test_weights(
  non_degenerate? : Bool = false,
) -> @materialize.HostMaterialization raise {
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let payload : Array[Byte] = []
  append_tensor(entries, payload, "model.embed_tokens.weight", [3L, 2L], [
    1.0F,
    0.0F,
    0.0F,
    1.0F,
    -1.0F,
    -1.0F,
  ])
  append_tensor(
    entries,
    payload,
    "model.layers.0.input_layernorm.weight",
    [2L],
    [1.0F, 1.0F],
  )
  for
    name in [
      "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
      "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight",
    ] {
    let values : Array[Float] = if non_degenerate {
      [1.0F, 0.0F, 0.0F, 1.0F]
    } else {
      [0.0F, 0.0F, 0.0F, 0.0F]
    }
    append_tensor(entries, payload, name, [2L, 2L], values)
  }
  append_tensor(
    entries,
    payload,
    "model.layers.0.post_attention_layernorm.weight",
    [2L],
    [1.0F, 1.0F],
  )
  for
    name in [
      "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight",
      "model.layers.0.mlp.down_proj.weight",
    ] {
    let values : Array[Float] = if non_degenerate {
      [1.0F, 0.0F, 0.0F, 1.0F]
    } else {
      [0.0F, 0.0F, 0.0F, 0.0F]
    }
    append_tensor(entries, payload, name, [2L, 2L], values)
  }
  append_tensor(entries, payload, "model.norm.weight", [2L], [1.0F, 1.0F])
  append_tensor(entries, payload, "lm_head.weight", [3L, 2L], [
    1.0F,
    0.0F,
    0.0F,
    1.0F,
    -1.0F,
    -1.0F,
  ])
  let header_bytes = 16
  let header_limits = @safetensors.HeaderLimits::new(
    max_header_bytes=128,
    max_tensor_count=16,
    max_rank=2,
    max_name_length=128,
    max_payload_bytes=1024L,
    max_materialized_tensor_bytes=1024L,
  )
  let metadata = @safetensors.validate_metadata(
    entries,
    encoded_header_bytes=header_bytes,
    payload_bytes=payload.length().to_int64(),
    limits=header_limits,
  )
  let bindings = @llama_weights.bind(test_model(), metadata)
  let file : Array[Byte] = [header_bytes.to_byte(), 0, 0, 0, 0, 0, 0, 0]
  file.push(b'{')
  file.push(b'}')
  for _ in 2..<header_bytes {
    file.push(b' ')
  }
  for byte in payload {
    file.push(byte)
  }
  let limits = @materialize.MaterializationLimits::new(
    max_header_bytes=128,
    max_tensor_count=16,
    max_payload_bytes=1024L,
    max_host_bytes=1024L,
  )
  @materialize.copy_to_host(bindings, Bytes::from_array(file), limits~)
}

///|
fn execution_limits() -> @reference.ExecutionLimits raise {
  @reference.ExecutionLimits::new(
    kernels=@refkernels.ReferenceLimits::new(
      max_rows=4,
      max_columns=8,
      max_elements=32,
    ),
    sampling=@sampling.SamplingLimits::new(1, 3),
    max_operations=16,
    max_tensors=12,
    max_input_tokens=4,
  )
}

///|
test "immutable Llama plan produces pinned logits and greedy token" {
  let plan = @llama.build(test_model())
  let prepared = @reference.prepare(
    plan,
    test_weights(),
    limits=execution_limits(),
  )
  assert_eq(prepared.identity(), plan.identity())
  assert_eq(prepared.tensor_count(), 12)
  let first = @reference.execute(plan, prepared, [0], limits=execution_limits())
  assert_eq(first.operation_count(), 12)
  assert_eq(first.identity(), plan.identity())
  assert_eq(first.logits().rows(), 1)
  assert_eq(first.logits().columns(), 3)
  assert_true(first.logits().at(row=0, column=0).is_close(1.4142121F))
  assert_true(first.logits().at(row=0, column=1).is_close(0.0F))
  assert_true(first.logits().at(row=0, column=2).is_close(-1.4142121F))
  assert_eq(first.selection().token_id(), 0)
  let second = @reference.execute(
    plan,
    prepared,
    [0, 1],
    limits=execution_limits(),
  )
  assert_true(second.logits().at(row=1, column=1).is_close(1.4142121F))
  assert_eq(second.selection().token_id(), 1)
}

///|
test "non-degenerate plan execution changes logits through attention and MLP" {
  let plan = @llama.build(test_model())
  let prepared = @reference.prepare(
    plan,
    test_weights(non_degenerate=true),
    limits=execution_limits(),
  )
  let result = @reference.execute(
    plan,
    prepared,
    [0, 1],
    limits=execution_limits(),
  )
  // Independently derived from split-half RoPE, causal softmax, identity
  // Q/K/V/O matrices, identity SiLU-gated MLP matrices, and both residuals.
  assert_true(
    result
    .logits()
    .at(row=1, column=0)
    .is_close(
      0.03575761F,
      relative_tolerance=0.00001F,
      absolute_tolerance=0.00001F,
    ),
  )
  assert_true(
    result
    .logits()
    .at(row=1, column=1)
    .is_close(
      1.4137614F,
      relative_tolerance=0.00001F,
      absolute_tolerance=0.00001F,
    ),
  )
  assert_true(
    result
    .logits()
    .at(row=1, column=2)
    .is_close(
      -1.449519F,
      relative_tolerance=0.00001F,
      absolute_tolerance=0.00001F,
    ),
  )
  assert_eq(result.selection().token_id(), 1)
}

///|
test "reference execution rejects identity and caller-bound violations" {
  let plan = @llama.build(test_model())
  let prepared = @reference.prepare(
    plan,
    test_weights(),
    limits=execution_limits(),
  )
  try @reference.execute(plan, prepared, [], limits=execution_limits()) catch {
    @reference.ReferenceExecutionError::InvalidGraph(InputLimitExceeded) => ()
    _ => fail("unexpected execution error")
  } noraise {
    _ => fail("expected empty-input rejection")
  }
  let narrow = @reference.ExecutionLimits::new(
    kernels=@refkernels.ReferenceLimits::new(
      max_rows=4,
      max_columns=8,
      max_elements=32,
    ),
    sampling=@sampling.SamplingLimits::new(1, 3),
    max_operations=11,
    max_tensors=12,
    max_input_tokens=4,
  )
  try @reference.execute(plan, prepared, [0], limits=narrow) catch {
    InvalidGraph(OperationLimitExceeded) => ()
    _ => fail("unexpected execution error")
  } noraise {
    _ => fail("expected operation-limit rejection")
  }
}
