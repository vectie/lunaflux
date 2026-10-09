///|
test "router projection renders F32 input BF16 row-major weight F32 output" {
  let rendered = f32_input_bf16_weight_projection_source(
    input_width=6144,
    output_width=256,
    maximum_rows=32,
    function_symbol="router_projection",
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_eq(rendered.operation(), Float32InputBf16WeightFloat32OutputProjection)
  assert_true(source.contains("const float *input"))
  assert_true(source.contains("const __nv_bfloat16 *weight"))
  assert_true(source.contains("float *output"))
  assert_true(source.contains("weight_base = column * 6144ULL"))
  assert_true(source.contains("__fmul_rn(input[input_base + inner]"))
  assert_true(source.contains("__fadd_rn(accumulator, product)"))
  assert_false(source.contains("sigmoid"))
  assert_false(source.contains("bias"))
}

///|
test "router projection rejects invalid geometry and symbol" {
  try
    f32_input_bf16_weight_projection_source(
      input_width=0,
      output_width=256,
      maximum_rows=1,
      function_symbol="x",
    )
  catch {
    InvalidInputWidth => ()
    _ => fail("wrong width error")
  } noraise {
    _ => fail("zero width admitted")
  }
  try
    f32_input_bf16_weight_projection_source(
      input_width=1,
      output_width=1,
      maximum_rows=0,
      function_symbol="x",
    )
  catch {
    InvalidMaximumRows => ()
    _ => fail("wrong row error")
  } noraise {
    _ => fail("zero rows admitted")
  }
  try
    f32_input_bf16_weight_projection_source(
      input_width=1,
      output_width=1,
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
