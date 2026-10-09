///|
test "two-stage projection emits explicit intermediate BF16 round" {
  let rendered = bias_free_bf16_low_rank_projection_source(
    input_width=4096,
    intermediate_width=128,
    output_width=8192,
    maximum_rows=32,
    function_symbol="test_two_stage_projection",
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(
    source.contains(
      "intermediate[intermediate_base + out] = __float2bfloat16_rn(sum)",
    ),
  )
  assert_true(
    source.contains("output[output_base + out] = __float2bfloat16_rn(sum)"),
  )
}

///|
test "two-stage projection rejects invalid intermediate and symbol" {
  try
    bias_free_bf16_low_rank_projection_source(
      input_width=1,
      intermediate_width=0,
      output_width=1,
      maximum_rows=1,
      function_symbol="ok",
    )
  catch {
    InvalidIntermediateWidth => ()
    _ => fail("wrong width error")
  } noraise {
    _ => fail("zero intermediate admitted")
  }
}
