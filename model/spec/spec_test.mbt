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
  let content = @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let metadata = @spec.LlamaModelMetadata::from_verified_content(spec, content)
  assert_eq(metadata.spec().layer_count(), 32)
  assert_eq(metadata.identity().content().as_hex(), content.as_hex())
}

///|
test "canonical plan identity is derived deterministically from semantic fields" {
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
  assert_eq(
    base.canonical_plan_digest().as_hex(),
    "27324d458a6df9e95d2dc38b7ae70b0010b1970d6d47c3c1c8c6c5bee5af6993",
  )
  assert_eq(
    base.canonical_paged_plan_digest().as_hex(),
    "e1eef676085ba63a306e1ccd1c7e82f7d32d8a49a4ea4d07b44c02560effcc01",
  )
  assert_true(
    base.canonical_paged_plan_digest() != base.canonical_plan_digest(),
  )
  let changed = @spec.LlamaModelSpec::new(
    vocabulary_size=32000,
    hidden_size=16,
    intermediate_size=32,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=2,
    context_length=128,
    rope_theta=20000.0,
    rms_norm_epsilon=0.000001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  assert_true(base.canonical_plan_digest() != changed.canonical_plan_digest())
  let content = @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  assert_eq(
    @spec.LlamaModelMetadata::from_verified_content(base, content)
    .identity()
    .plan(),
    base.canonical_plan_digest(),
  )
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
