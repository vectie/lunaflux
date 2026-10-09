///|
test "interleaved RoPE emits split-half outputs and bit-exact query prefix copy" {
  let rendered = interleaved_pair_rope_with_query_prefix_copy_bf16_source(
    query_head_count=64,
    key_head_count=1,
    query_head_dimension=256,
    rope_dimension=64,
    maximum_rows=32,
    maximum_positions=1048576,
    theta=8000000.0,
    function_symbol="interleaved_rope",
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("column < 192"))
  assert_true(source.contains("pair >= 32"))
  assert_true(source.contains("output_base + 32ULL"))
  assert_true(
    source.contains("query_output[query_base + column] = query_input"),
  )
  assert_true(source.contains("cosine_bf16 = __float2bfloat16_rn(cosf(angle))"))
  assert_true(source.contains("left_cos_bf16 = __float2bfloat16_rn"))
  assert_true(source.contains("position < 0 || position >= 1048576"))
  assert_true(source.contains("__float2bfloat16_rn"))
}

///|
test "interleaved RoPE rejects invalid geometry and theta" {
  try
    interleaved_pair_rope_with_query_prefix_copy_bf16_source(
      query_head_count=64,
      key_head_count=1,
      query_head_dimension=256,
      rope_dimension=63,
      maximum_rows=32,
      maximum_positions=128,
      theta=8000000.0,
      function_symbol="x",
    )
  catch {
    InvalidRopeDimension => ()
    _ => fail("wrong rope error")
  } noraise {
    _ => fail("odd rope admitted")
  }
  try
    interleaved_pair_rope_with_query_prefix_copy_bf16_source(
      query_head_count=64,
      key_head_count=1,
      query_head_dimension=256,
      rope_dimension=64,
      maximum_rows=32,
      maximum_positions=128,
      theta=0.0,
      function_symbol="x",
    )
  catch {
    InvalidTheta => ()
    _ => fail("wrong theta error")
  } noraise {
    _ => fail("zero theta admitted")
  }
}
