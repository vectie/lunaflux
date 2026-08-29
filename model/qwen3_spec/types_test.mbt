///|
fn pinned_qwen3_spec() -> @qwen3_spec.Qwen3ModelSpec raise {
  @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    layer_count=28,
    attention_head_count=16,
    kv_head_count=8,
    head_dimension=128,
    context_length=40960,
    rope_theta=1000000.0,
    rms_norm_epsilon=0.000001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
}

///|
test "pinned Qwen3 geometry retains explicit wide query heads" {
  let spec = pinned_qwen3_spec()
  assert_eq(spec.hidden_size(), 1024)
  assert_eq(spec.head_dimension(), 128)
  assert_eq(spec.query_projection_width(), 2048)
  assert_eq(spec.kv_projection_width(), 1024)
  assert_true(spec.has_query_key_norm())
  assert_true(spec.tied_embeddings())
}

///|
test "Qwen3 rejects hostile head geometry and untied embeddings" {
  try
    @qwen3_spec.Qwen3ModelSpec::new(
      vocabulary_size=151936,
      hidden_size=1024,
      intermediate_size=3072,
      layer_count=28,
      attention_head_count=16,
      kv_head_count=8,
      head_dimension=127,
      context_length=40960,
      rope_theta=1000000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
      tied_embeddings=true,
    )
  catch {
    InvalidHeadRelationship(OddHeadDimension) => ()
    _ => fail("unexpected odd-head rejection")
  } noraise {
    _ => fail("odd Qwen3 head dimension was admitted")
  }
  try
    @qwen3_spec.Qwen3ModelSpec::new(
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
      tied_embeddings=false,
    )
  catch {
    UntiedEmbeddings => ()
    _ => fail("unexpected tied-embedding rejection")
  } noraise {
    _ => fail("untied Qwen3 embeddings were admitted")
  }
}
