///|
test "inverse scaled RoPE source copies prefix and conjugates suffix" {
  let rendered = inverse_scaled_yarn_attention_output_bf16_source(
    head_count=2,
    head_dimension=8,
    rope_dimension=4,
    maximum_rows=3,
    maximum_positions=1024,
    original_positions=128,
    compressed_theta=20000.0,
    scaling_factor=8.0,
    beta_fast=32.0,
    beta_slow=1.0,
    function_symbol="inverse_rope_test",
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("if (pair < 2) { return; }"))
  assert_true(source.contains("__fmul_rn(right, sine)"))
  assert_true(source.contains("-__fmul_rn(left, sine)"))
  assert_false(source.contains("query_input"))
  assert_false(source.contains("__restrict__"))
  assert_false(source.contains("attention_output_words"))
}

///|
test "panic inverse scaled RoPE rejects incompatible dimensions" {
  try
    inverse_scaled_yarn_attention_output_bf16_source(
      head_count=1,
      head_dimension=7,
      rope_dimension=4,
      maximum_rows=1,
      maximum_positions=16,
      original_positions=8,
      compressed_theta=10000.0,
      scaling_factor=1.0,
      beta_fast=32.0,
      beta_slow=1.0,
      function_symbol="valid_symbol",
    )
  catch {
    InvalidHeadDimension => ()
    _ => fail("wrong head-dimension failure")
  } noraise {
    _ => fail("odd head dimension was accepted")
  }
}
