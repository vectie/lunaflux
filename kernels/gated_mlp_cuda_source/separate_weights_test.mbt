///|
test "separate gate up down renderer authenticates order and live rows" {
  let rendered = separate_gate_up_swiglu_dense_bf16_source(
    hidden_size=6144,
    intermediate_size=12288,
    maximum_rows=32,
    function_symbol="glm_dense_swiglu",
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("counts, const __nv_bfloat16 *input"))
  assert_true(source.contains("*gate_weight, const __nv_bfloat16 *up_weight"))
  assert_true(source.contains("live_rows > 32"))
  assert_true(source.contains("gate_weight[weight_base + column]"))
  assert_true(source.contains("up_weight[weight_base + column]"))
  assert_true(source.contains("down_weight[(unsigned long long)out * 12288ULL"))
  assert_eq(
    rendered.numerical_contract(),
    OrderedF32AccumulationBf16SwiGluDownBf16V1,
  )
}

///|
test "separate gate up down renderer rejects row and width mismatches" {
  try
    separate_gate_up_swiglu_dense_bf16_source(
      hidden_size=0,
      intermediate_size=12288,
      maximum_rows=32,
      function_symbol="x",
    )
  catch {
    InvalidWidth => ()
    _ => fail("wrong width error")
  } noraise {
    _ => fail("zero hidden width admitted")
  }
  try
    separate_gate_up_swiglu_dense_bf16_source(
      hidden_size=4096,
      intermediate_size=12288,
      maximum_rows=0,
      function_symbol="x",
    )
  catch {
    InvalidRowCount => ()
    _ => fail("wrong row error")
  } noraise {
    _ => fail("zero maximum rows admitted")
  }
}
