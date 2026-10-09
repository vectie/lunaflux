///|
test "scaled YaRN renderer binds adjacent BF16 pair geometry" {
  let rendered = scaled_yarn_rope_bf16_source(
    query_head_count=64,
    key_value_head_count=1,
    rope_dimension=64,
    maximum_rows=32,
    maximum_positions=1048576,
    original_positions=65536,
    compressed_theta=160000.0,
    scaling_factor=16.0,
    beta_fast=32.0,
    beta_slow=1.0,
    function_symbol="rope_test",
  )
  assert_eq(rendered.numerical_contract(), AdjacentPairF32YarnBf16RoundV1)
  assert_eq(rendered.query_head_count(), 64)
  assert_eq(rendered.rope_dimension(), 64)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("live_rows > 32"))
  assert_true(source.contains("position >= 1048576"))
  assert_true(source.contains("pair * 2"))
  assert_true(source.contains("powf(160000.0f"))
  assert_true(source.contains("cosf(angle)"))
  assert_true(source.contains("sin_f(angle)") == false)
  assert_true(source.contains("__float2bfloat16_rn"))
}

///|
test "scaled YaRN renderer rejects hostile geometry and symbol" {
  try
    scaled_yarn_rope_bf16_source(
      query_head_count=1,
      key_value_head_count=1,
      rope_dimension=63,
      maximum_rows=1,
      maximum_positions=2,
      original_positions=1,
      compressed_theta=10000.0,
      scaling_factor=1.0,
      beta_fast=32.0,
      beta_slow=1.0,
      function_symbol="rope",
    )
  catch {
    InvalidRopeDimension => ()
    _ => fail("wrong dimension error")
  } noraise {
    _ => fail("odd RoPE width was admitted")
  }
  try
    scaled_yarn_rope_bf16_source(
      query_head_count=1,
      key_value_head_count=1,
      rope_dimension=64,
      maximum_rows=1,
      maximum_positions=2,
      original_positions=3,
      compressed_theta=10000.0,
      scaling_factor=1.0,
      beta_fast=32.0,
      beta_slow=1.0,
      function_symbol="9rope",
    )
  catch {
    InvalidOriginalPositions => ()
    _ => fail("wrong positions error")
  } noraise {
    _ => fail("oversized original positions were admitted")
  }
  try
    scaled_yarn_rope_bf16_source(
      query_head_count=1,
      key_value_head_count=1,
      rope_dimension=64,
      maximum_rows=1,
      maximum_positions=2,
      original_positions=1,
      compressed_theta=0.0,
      scaling_factor=1.0,
      beta_fast=32.0,
      beta_slow=1.0,
      function_symbol="rope",
    )
  catch {
    InvalidTheta => ()
    _ => fail("wrong theta error")
  } noraise {
    _ => fail("zero compressed theta was admitted")
  }
  try
    scaled_yarn_rope_bf16_source(
      query_head_count=1,
      key_value_head_count=1,
      rope_dimension=64,
      maximum_rows=1,
      maximum_positions=2,
      original_positions=1,
      compressed_theta=10000.0,
      scaling_factor=0.0,
      beta_fast=32.0,
      beta_slow=1.0,
      function_symbol="rope",
    )
  catch {
    InvalidScaling => ()
    _ => fail("wrong scaling error")
  } noraise {
    _ => fail("zero scaling factor was admitted")
  }
}
