///|
test "decay control renders official bias rate sigmoid lower-bound order" {
  let rendered = safe_lower_bound_sigmoid_decay_f32_source(
    head_count=64,
    head_dimension=128,
    lower_bound=-5.0,
    maximum_rows=32,
    function_symbol="decay_control",
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_eq(rendered.operation(), SafeLowerBoundSigmoidDecayF32)
  assert_true(
    source.contains("__bfloat162float(raw_forget[index]), dt_bias[component]"),
  )
  assert_true(source.contains("decay_rate = expf(a_log[head])"))
  assert_true(source.contains("__fmul_rn(decay_rate, biased_logit)"))
  assert_true(source.contains("expf(-scaled)"))
  assert_true(source.contains("__fmul_rn(-5.0f, sigmoid)"))
  assert_false(source.contains("query"))
  assert_false(source.contains("initial_state"))
}

///|
test "decay control rejects invalid geometry bound rows and symbol" {
  try
    safe_lower_bound_sigmoid_decay_f32_source(
      head_count=0,
      head_dimension=128,
      lower_bound=-5.0,
      maximum_rows=1,
      function_symbol="x",
    )
  catch {
    InvalidHeadCount => ()
    _ => fail("wrong head error")
  } noraise {
    _ => fail("zero heads admitted")
  }
  try
    safe_lower_bound_sigmoid_decay_f32_source(
      head_count=64,
      head_dimension=128,
      lower_bound=0.0,
      maximum_rows=1,
      function_symbol="x",
    )
  catch {
    InvalidLowerBound => ()
    _ => fail("wrong bound error")
  } noraise {
    _ => fail("nonnegative bound admitted")
  }
  try
    safe_lower_bound_sigmoid_decay_f32_source(
      head_count=64,
      head_dimension=128,
      lower_bound=-5.0,
      maximum_rows=0,
      function_symbol="x",
    )
  catch {
    InvalidMaximumRows => ()
    _ => fail("wrong rows error")
  } noraise {
    _ => fail("zero rows admitted")
  }
}
