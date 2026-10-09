///|
test "DSA pooling renderer fixes standalone semantic and ABI boundaries" {
  let rendered = dsa_key_pool_compression_bf16_source(
    head_dimension=128,
    pool_size=4,
    maximum_tokens=32,
    function_symbol="lunaflux_glm_hybrid_dsa_key_pool_bf16_v1",
  )
  assert_eq(rendered.operation(), DsaKeyPoolCompressionBf16)
  assert_eq(rendered.head_dimension(), 128)
  assert_eq(rendered.pool_size(), 4)
  assert_eq(rendered.pool_capacity(), 8)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const int live_tokens = counts[3];"))
  assert_true(source.contains("__float2bfloat16_rn(__fdividef"))
  assert_true(source.contains("raw_indices"))
  assert_false(source.contains("topk"))
  assert_false(source.contains("tail"))
}

///|
test "DSA pooling renderer rejects non-poolable envelopes" {
  try
    dsa_key_pool_compression_bf16_source(
      head_dimension=128,
      pool_size=4,
      maximum_tokens=3,
      function_symbol="lunaflux_glm_hybrid_dsa_key_pool_bf16_v1",
    )
  catch {
    InvalidMaximumTokens => ()
    _ => fail("unexpected renderer error")
  } noraise {
    _ => fail("sub-pool token envelope accepted")
  }
}
