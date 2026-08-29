///|
fn valid_spec() -> @spec.LlamaModelSpec raise {
  @spec.LlamaModelSpec::new(
    vocabulary_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    layer_count=32,
    attention_head_count=32,
    kv_head_count=8,
    context_length=4096,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
}

///|
test "valid dense Llama metadata is plan-ready and immutable" {
  let spec = valid_spec()
  assert_eq(spec.head_dimension(), 128)
  assert_eq(spec.kv_head_count(), 8)
  assert_eq(spec.context_length(), 4096)
  assert_eq(spec.rms_norm_epsilon(), 0.000001)
  let numeric = spec.numeric_schema()
  assert_eq(numeric.tensors().length(), 9 * spec.layer_count() + 3)
  assert_eq(numeric.operations().length(), 9 * spec.layer_count() + 3)
  assert_eq(numeric.digest().as_hex().length(), 64)
  let content = @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let metadata = @spec.LlamaModelMetadata::from_verified_content(spec, content)
  assert_eq(metadata.spec().layer_count(), 32)
  assert_eq(metadata.content_digest().as_hex(), content.as_hex())
}

///|
test "metadata carries content without minting a plan identity" {
  let base = @spec.LlamaModelSpec::new(
    vocabulary_size=32000,
    hidden_size=16,
    intermediate_size=32,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=2,
    context_length=128,
    rope_theta=10000.0,
    rms_norm_epsilon=0.000001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let content = @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let metadata = @spec.LlamaModelMetadata::from_verified_content(base, content)
  assert_eq(metadata.content_digest(), content)
  assert_eq(base.numeric_schema(), base.numeric_schema())
}

///|
test "RMS normalization epsilon is explicit and bounded" {
  let spec = @spec.LlamaModelSpec::new(
    vocabulary_size=32,
    hidden_size=8,
    intermediate_size=16,
    layer_count=1,
    attention_head_count=2,
    kv_head_count=1,
    context_length=16,
    rope_theta=10000.0,
    rms_norm_epsilon=0.00001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  assert_eq(spec.rms_norm_epsilon(), 0.00001)
  try
    @spec.LlamaModelSpec::new(
      vocabulary_size=32,
      hidden_size=8,
      intermediate_size=16,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      context_length=16,
      rope_theta=10000.0,
      rms_norm_epsilon=0.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    )
  catch {
    InvalidRmsNormEpsilon => ()
    _ => fail("unexpected model error")
  } noraise {
    _ => fail("expected invalid RMS normalization epsilon")
  }
}

///|
test "invalid dimensions and head relationships are rejected" {
  try
    @spec.LlamaModelSpec::new(
      vocabulary_size=32000,
      hidden_size=4097,
      intermediate_size=11008,
      layer_count=32,
      attention_head_count=32,
      kv_head_count=8,
      context_length=4096,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    )
  catch {
    InvalidHeadRelationship(HiddenNotDivisibleByAttentionHeads) => ()
    _ => fail("unexpected model error")
  } noraise {
    _ => fail("expected invalid head relationship")
  }
  try
    @spec.LlamaModelSpec::new(
      vocabulary_size=0,
      hidden_size=4096,
      intermediate_size=11008,
      layer_count=32,
      attention_head_count=32,
      kv_head_count=8,
      context_length=4096,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    )
  catch {
    InvalidDimension(Vocabulary) => ()
    _ => fail("unexpected model error")
  } noraise {
    _ => fail("expected invalid vocabulary")
  }
}

///|
test "Llama layer authority fits the numeric and plan operation ceilings" {
  let maximum = @spec.LlamaModelSpec::new(
    vocabulary_size=32,
    hidden_size=8,
    intermediate_size=16,
    layer_count=454,
    attention_head_count=2,
    kv_head_count=1,
    context_length=16,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  assert_eq(3 + 9 * maximum.layer_count(), 4089)
  try
    @spec.LlamaModelSpec::new(
      vocabulary_size=32,
      hidden_size=8,
      intermediate_size=16,
      layer_count=455,
      attention_head_count=2,
      kv_head_count=1,
      context_length=16,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    )
  catch {
    InvalidDimension(LayerCount) => ()
    _ => fail("unexpected layer ceiling failure")
  } noraise {
    _ => fail("layer count above the plan operation ceiling was admitted")
  }
}

///|
test "unsupported precision and malformed digest fail closed" {
  try
    @spec.LlamaModelSpec::new(
      vocabulary_size=32000,
      hidden_size=4096,
      intermediate_size=11008,
      layer_count=32,
      attention_head_count=32,
      kv_head_count=8,
      context_length=4096,
      rope_theta=10000.0,
      weight_dtype=Float16,
      kv_dtype=BFloat16,
    )
  catch {
    UnsupportedWeightDType => ()
    _ => fail("unexpected model error")
  } noraise {
    _ => fail("expected unsupported dtype")
  }
  try @spec.ContentDigest::from_sha256("not-a-digest") catch {
    InvalidDigest => ()
    _ => fail("unexpected digest error")
  } noraise {
    _ => fail("expected invalid digest")
  }
}
