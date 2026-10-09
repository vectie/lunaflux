///|
test "I32 token-hash source is deterministic bounded and model-neutral" {
  let source = token_hash_lookup_i32_source(
    vocabulary_size=129280,
    experts_per_token=6,
    routed_expert_count=256,
    maximum_rows=32,
    function_symbol="lunaflux_advanced_decoder_token_hash_i32_v1",
  )
  assert_eq(source.operation(), TokenHashLookupI32)
  assert_eq(source.numerical_contract(), BitExactInt32LookupV1)
  assert_eq(source.vocabulary_size(), 129280)
  assert_eq(source.experts_per_token(), 6)
  assert_eq(source.routed_expert_count(), 256)
  assert_eq(source.maximum_rows(), 32)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("const int32_t *table"))
  assert_true(text.contains("int32_t *expert_indices"))
  assert_true(text.contains("live_rows > 32"))
  assert_true(text.contains("token >= 129280"))
  assert_true(text.contains("slot >= 6"))
  assert_true(text.contains("expert < 256"))
  assert_true(text.contains("expert_indices[output_index] = -1"))
  let repeated = token_hash_lookup_i32_source(
    vocabulary_size=129280,
    experts_per_token=6,
    routed_expert_count=256,
    maximum_rows=32,
    function_symbol="lunaflux_advanced_decoder_token_hash_i32_v1",
  )
  assert_eq(source.source_bytes(), repeated.source_bytes())
}

///|
test "I32 token-hash source rejects hostile geometry and symbol" {
  for invalid in [0, 257] {
    try
      token_hash_lookup_i32_source(
        vocabulary_size=129280,
        experts_per_token=invalid,
        routed_expert_count=256,
        maximum_rows=1,
        function_symbol="token_hash_i32",
      )
    catch {
      InvalidExpertsPerToken => ()
      _ => fail("unexpected selected-expert geometry rejection")
    } noraise {
      _ => fail("invalid selected-expert geometry was admitted")
    }
  }
  try
    token_hash_lookup_i32_source(
      vocabulary_size=129280,
      experts_per_token=6,
      routed_expert_count=5,
      maximum_rows=1,
      function_symbol="token_hash_i32",
    )
  catch {
    InvalidRoutedExpertCount => ()
    _ => fail("unexpected routed-expert geometry rejection")
  } noraise {
    _ => fail("selected count larger than routed width was admitted")
  }
  try
    token_hash_lookup_i32_source(
      vocabulary_size=0,
      experts_per_token=6,
      routed_expert_count=256,
      maximum_rows=1,
      function_symbol="9bad",
    )
  catch {
    InvalidVocabularySize => ()
    _ => fail("unexpected vocabulary rejection")
  } noraise {
    _ => fail("zero vocabulary was admitted")
  }
  try
    token_hash_lookup_i32_source(
      vocabulary_size=1,
      experts_per_token=1,
      routed_expert_count=1,
      maximum_rows=1048577,
      function_symbol="9bad",
    )
  catch {
    InvalidMaximumRows => ()
    _ => fail("unexpected maximum-row rejection")
  } noraise {
    _ => fail("oversized row envelope was admitted")
  }
  try
    token_hash_lookup_i32_source(
      vocabulary_size=1,
      experts_per_token=1,
      routed_expert_count=1,
      maximum_rows=1,
      function_symbol="9bad",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("invalid symbol was admitted")
  }
}
