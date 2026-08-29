///|
test "Mistral context is capped before sliding attention would diverge" {
  let spec = @mistral_spec.MistralModelSpec::new(
    vocabulary_size=32000,
    hidden_size=4096,
    intermediate_size=14336,
    layer_count=32,
    attention_head_count=32,
    kv_head_count=8,
    max_position_embeddings=32768,
    sliding_window=4096,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  assert_eq(spec.max_position_embeddings(), 32768)
  assert_eq(spec.sliding_window(), Some(4096))
  assert_eq(spec.context_length(), 4096)
}

///|
test "Mistral without sliding attention retains the position ceiling" {
  let spec = @mistral_spec.MistralModelSpec::new(
    vocabulary_size=128,
    hidden_size=16,
    intermediate_size=32,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=2,
    max_position_embeddings=64,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  assert_eq(spec.sliding_window(), None)
  assert_eq(spec.context_length(), 64)
}

///|
test "Mistral rejects an empty sliding window" {
  try
    @mistral_spec.MistralModelSpec::new(
      vocabulary_size=128,
      hidden_size=16,
      intermediate_size=32,
      layer_count=2,
      attention_head_count=4,
      kv_head_count=2,
      max_position_embeddings=64,
      sliding_window=0,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    )
  catch {
    InvalidDimension(ContextLength) => ()
    _ => fail("unexpected Mistral validation error")
  } noraise {
    _ => fail("empty Mistral sliding window was admitted")
  }
}
