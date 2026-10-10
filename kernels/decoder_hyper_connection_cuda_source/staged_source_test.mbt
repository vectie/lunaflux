///|
test "staged function preserves BF16 checkpoint storage and explicit outputs" {
  let rendered = staged_function_and_collapse_bf16_f32_source(
    stream_width=4,
    hidden_width=4096,
    maximum_rows=8,
    control_epsilon=0.000001,
    normalization_epsilon=0.00001,
    function_symbol="lunaflux_hc_function_v1",
  )
  assert_eq(rendered.operation(), StagedFunctionAndCollapseV1)
  assert_eq(rendered.numerical_contract(), OrderedF32FunctionBf16CollapseV1)
  assert_eq(rendered.stream_width(), 4)
  assert_eq(rendered.hidden_width(), 4096)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const __nv_bfloat16 *function_weights"))
  assert_true(source.contains("float *post, float *initial_combination"))
  assert_true(source.contains("__nv_bfloat16 *collapsed"))
  assert_true(source.contains("normalized = __fmul_rn"))
  assert_true(source.contains("initial_combination[index] = __fadd_rn"))
  assert_true(source.contains("__float2bfloat16_rn(sum)"))
  assert_true(source.contains("const void *buffers[8]"))
  assert_true(source.contains("buffers[left] == buffers[right]"))
}

///|
test "F32 controls normalize the projected value rather than individual operands" {
  let rendered = staged_function_and_collapse_bf16_f32_source(
    stream_width=4,
    hidden_width=4096,
    maximum_rows=8,
    control_epsilon=0.000001,
    normalization_epsilon=0.00001,
    function_symbol="f32_control",
    control_law=F32ProjectionThenNormalize,
  )
  assert_eq(
    rendered.numerical_contract(),
    OrderedF32ProjectionThenNormalizeBf16CollapseV1,
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const float *function_weights"))
  assert_true(source.contains("mixes[output] = __fmul_rn(dot, inverse);"))
  assert_false(source.contains("normalized = __fmul_rn"))
}

///|
test "staged Sinkhorn preserves initial-column then nineteen row-column passes" {
  let rendered = staged_positive_sinkhorn_f32_source(
    stream_width=4,
    maximum_rows=8,
    sinkhorn_iterations=20,
    control_epsilon=0.000001,
    function_symbol="lunaflux_hc_sinkhorn_v1",
  )
  assert_eq(rendered.operation(), StagedSinkhornV1)
  assert_eq(rendered.numerical_contract(), OrderedF32PositiveSinkhornV1)
  assert_eq(rendered.sinkhorn_iterations(), 20)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("initial_combination == combination"))
  assert_true(source.contains("value <= 0.0f"))
  assert_true(source.contains("iteration = 1; iteration < 20"))
  assert_true(source.contains("destination * 4ULL"))
  assert_true(source.contains("__fadd_rn(sum, 0.000001f)"))
}

///|
test "staged sources reject hostile geometry and policy" {
  try
    staged_function_and_collapse_bf16_f32_source(
      stream_width=1,
      hidden_width=4096,
      maximum_rows=8,
      control_epsilon=0.000001,
      normalization_epsilon=0.00001,
      function_symbol="valid",
    )
  catch {
    InvalidStreamWidth => ()
    _ => fail("wrong function geometry rejection")
  } noraise {
    _ => fail("single stream admitted")
  }
  try
    staged_positive_sinkhorn_f32_source(
      stream_width=4,
      maximum_rows=8,
      sinkhorn_iterations=65,
      control_epsilon=0.000001,
      function_symbol="valid",
    )
  catch {
    InvalidIterationCount => ()
    _ => fail("wrong Sinkhorn iteration rejection")
  } noraise {
    _ => fail("unbounded iterations admitted")
  }
}
