///|
test "packed SwiGLU source preserves official halves and BF16 stages" {
  let rendered = packed_swiglu_dense_bf16_rows_source(
    rows=3,
    hidden_size=4,
    intermediate_size=8,
    function_symbol="swiglu_rows_v1",
  )
  assert_eq(
    rendered.numerical_contract(),
    OrderedF32AccumulationBf16SwiGluDownBf16V1,
  )
  assert_eq(rendered.row_count(), 3)
  assert_eq(rendered.hidden_size(), 4)
  assert_eq(rendered.intermediate_size(), 8)
  let source = try! @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const __nv_bfloat16 *input"))
  assert_true(source.contains("const __nv_bfloat16 *packed_up_gate_weight"))
  assert_true(source.contains("inner + 8"))
  assert_true(source.contains("const __nv_bfloat16 up = __float2bfloat16_rn"))
  assert_true(source.contains("const __nv_bfloat16 gate = __float2bfloat16_rn"))
  assert_true(source.contains("expf(-gate_f32)"))
  assert_true(
    source.contains("const __nv_bfloat16 product = __float2bfloat16_rn"),
  )
  assert_true(source.contains("output[i] = __float2bfloat16_rn(down_sum)"))
}

///|
test "packed SwiGLU source rejects hostile geometry and symbol" {
  for geometry in [(0, 4, 8), (1, 0, 8), (1, 4, 0)] {
    let (rows, hidden, intermediate) = geometry
    try
      packed_swiglu_dense_bf16_rows_source(
        rows~,
        hidden_size=hidden,
        intermediate_size=intermediate,
        function_symbol="ok",
      )
    catch {
      InvalidRowCount | InvalidWidth => ()
      _ => fail("unexpected geometry rejection")
    } noraise {
      _ => fail("hostile geometry admitted")
    }
  }
  try
    packed_swiglu_dense_bf16_rows_source(
      rows=1073741824,
      hidden_size=2,
      intermediate_size=8,
      function_symbol="ok",
    )
  catch {
    DimensionOverflow => ()
    _ => fail("unexpected overflow rejection")
  } noraise {
    _ => fail("overflowing rows admitted")
  }
  try
    packed_swiglu_dense_bf16_rows_source(
      rows=1,
      hidden_size=4,
      intermediate_size=8,
      function_symbol="9bad",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("leading digit symbol admitted")
  }
}
