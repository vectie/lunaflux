///|
test "shared embedding and RMSNorm source is deterministic" {
  let embedding = token_embedding_bf16_source(
    vocabulary_size=128,
    hidden_size=16,
    function_symbol="lunaflux_advanced_decoder_embedding_bf16_v1",
  )
  assert_eq(embedding.operation(), TokenEmbeddingBf16)
  assert_eq(embedding.numerical_contract(), BitExactBf16Copy)
  let repeated = token_embedding_bf16_source(
    vocabulary_size=128,
    hidden_size=16,
    function_symbol="lunaflux_advanced_decoder_embedding_bf16_v1",
  )
  assert_eq(embedding.source_bytes(), repeated.source_bytes())
  let norm = rms_norm_bf16_source(
    hidden_size=16,
    epsilon=0.000001,
    function_symbol="lunaflux_advanced_decoder_final_rms_norm_bf16_v1",
  )
  assert_eq(norm.operation(), RmsNormBf16)
  assert_eq(norm.numerical_contract(), OrderedFloat32ReductionBf16RoundV1)
  let norm_text = @utf8.decode(norm.source_bytes())
  assert_true(norm_text.contains("__fmul_rn"))
  assert_true(norm_text.contains("__fadd_rn"))
  assert_true(norm_text.contains("__fdiv_rn"))
}

///|
test "shared source validation fails closed" {
  try
    rms_norm_bf16_source(
      hidden_size=16,
      epsilon=0.0,
      function_symbol="valid_symbol",
    )
  catch {
    InvalidEpsilon => ()
    _ => fail("unexpected epsilon rejection")
  } noraise {
    _ => fail("invalid epsilon was accepted")
  }
  try
    token_embedding_bf16_source(
      vocabulary_size=1,
      hidden_size=1,
      function_symbol="invalid-symbol",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("invalid symbol was accepted")
  }
  try
    rms_norm_bf16_source(
      hidden_size=16,
      epsilon=0.000001,
      function_symbol="1invalid_symbol",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected leading-digit symbol rejection")
  } noraise {
    _ => fail("leading-digit symbol was accepted")
  }
}
