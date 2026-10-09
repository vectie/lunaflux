///|
test "recurrent projection source preserves explicit BF16 control intermediates" {
  let rendered = qkv_forget_input_gate_bf16_source(
    hidden_width=4096,
    head_count=64,
    head_dimension=128,
    control_rank=128,
    maximum_rows=32,
    function_symbol="test_recurrent_projection",
  )
  assert_eq(rendered.hidden_width(), 4096)
  assert_eq(rendered.head_count(), 64)
  assert_eq(rendered.head_dimension(), 128)
  assert_eq(rendered.control_rank(), 128)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("__nv_bfloat16 *forget_low_rank"))
  assert_true(source.contains("__nv_bfloat16 *raw_forget"))
  assert_true(
    source.contains("const __nv_bfloat16 logit = __float2bfloat16_rn(sum)"),
  )
  assert_true(
    source.contains("beta[gate_base + out] = __float2bfloat16_rn(sigmoid)"),
  )
  assert_false(source.contains("dt_bias"))
  assert_false(source.contains("A_log"))
}

///|
test "recurrent projection source rejects invalid shape and symbol" {
  try
    qkv_forget_input_gate_bf16_source(
      hidden_width=1,
      head_count=1,
      head_dimension=1,
      control_rank=0,
      maximum_rows=1,
      function_symbol="ok",
    )
  catch {
    InvalidControlRank => ()
    _ => fail("wrong rank error")
  } noraise {
    _ => fail("zero control rank admitted")
  }
  try
    qkv_forget_input_gate_bf16_source(
      hidden_width=1,
      head_count=1,
      head_dimension=1,
      control_rank=1,
      maximum_rows=1,
      function_symbol="bad-symbol",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong symbol error")
  } noraise {
    _ => fail("bad symbol admitted")
  }
}
