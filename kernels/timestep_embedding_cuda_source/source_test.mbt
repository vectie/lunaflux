///|
test "timestep rows render exact flipped cos sin F32 ABI" {
  let rendered = cosine_sine_timestep_rows_source(
    rows=3,
    frequency_width=256,
    function_symbol="time_rows_v1",
  )
  assert_eq(rendered.numerical_contract(), Float32CosSinExpfPeriod10000V1)
  assert_eq(rendered.row_count(), 3)
  let source = try! @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const float *timestep"))
  assert_true(source.contains("expf(exponent)"))
  assert_true(source.contains("cosf(angle) : sinf(angle)"))
}

///|
test "timestep rows reject hostile dimensions and symbols" {
  try
    cosine_sine_timestep_rows_source(
      rows=0,
      frequency_width=256,
      function_symbol="ok",
    )
  catch {
    InvalidRowCount => ()
    _ => fail("unexpected row rejection")
  } noraise {
    _ => fail("zero rows admitted")
  }
  try
    cosine_sine_timestep_rows_source(
      rows=1,
      frequency_width=255,
      function_symbol="ok",
    )
  catch {
    InvalidFrequencyWidth => ()
    _ => fail("unexpected width rejection")
  } noraise {
    _ => fail("odd width admitted")
  }
  try
    cosine_sine_timestep_rows_source(
      rows=1,
      frequency_width=256,
      function_symbol="9bad",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("bad symbol admitted")
  }
}

///|
test "dense SiLU dense rows render the exact six-operand F32 ABI" {
  let rendered = dense_silu_dense_f32_rows_source(
    rows=3,
    input_width=256,
    hidden_width=5376,
    output_width=2688,
    function_symbol="timestep_mlp_v1",
  )
  assert_eq(rendered.numerical_contract(), OrderedFloat32DenseSiluDenseV1)
  assert_eq(rendered.row_count(), 3)
  assert_eq(rendered.frequency_width(), 2688)
  let source = try! @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const float *input"))
  assert_true(source.contains("const float *weight1"))
  assert_true(source.contains("const float *bias1"))
  assert_true(source.contains("const float *weight2"))
  assert_true(source.contains("const float *bias2"))
  assert_true(source.contains("float *output"))
  assert_true(source.contains("__fadd_rn"))
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fdiv_rn"))
  assert_true(source.contains("expf(-first)"))
}

///|
test "dense SiLU dense rows reject hostile widths" {
  for widths in [(0, 4, 2), (4, 0, 2), (4, 8, 0)] {
    let (input, hidden, output) = widths
    try
      dense_silu_dense_f32_rows_source(
        rows=1,
        input_width=input,
        hidden_width=hidden,
        output_width=output,
        function_symbol="ok",
      )
    catch {
      InvalidFrequencyWidth => ()
      _ => fail("unexpected dense width rejection")
    } noraise {
      _ => fail("hostile dense width admitted")
    }
  }
}
