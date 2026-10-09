///|
test "source distinguishes causal encoder from noncausal DiT and omits final norm" {
  let plan = @text_plan.TextEncoderPlan::new(rows=7, epsilon=0.000001)
  let source = build_source(plan, prefix="h3text")
  let text = @utf8.decode(source.source)
  assert_eq(source.symbols.length(), 13)
  assert_true(text.contains("key<=row"))
  assert_true(text.contains("const int kh=h/8;"))
  assert_true(text.contains("rsqrtf(128.0f)"))
  assert_true(text.contains("h3text_rotary"))
  assert_false(text.contains("final_norm"))
  assert_false(text.contains("lm_head"))
}

///|
test "current shared MLP row ceiling is explicit at source construction" {
  let maximum = @text_plan.TextEncoderPlan::new(rows=83886, epsilon=0.000001)
  let source = build_source(maximum, prefix="largest_encoder")
  assert_eq(source.plan.rows, 83886)
  let next = @text_plan.TextEncoderPlan::new(rows=83887, epsilon=0.000001)
  try build_source(next, prefix="too_large_encoder") catch {
    @gated_mlp_cuda_source.GatedMlpCudaSourceError::DimensionOverflow => ()
    _ => fail("wrong row-limit error")
  } noraise {
    _ => fail("unsupported row extent admitted")
  }
}
