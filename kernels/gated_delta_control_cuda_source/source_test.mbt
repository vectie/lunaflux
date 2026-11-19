///|
test "controls keep FP32 unbounded decay and BF16 beta" {
  let plan = @precision.GatedDeltaControlPrecision::new(
    head_count=32,
    maximum_rows=32,
  )
  let source = @gated_delta_control_cuda_source.gated_delta_control_source(
    plan,
    symbol="controls",
  )
  let text = @utf8.decode(source.source_bytes()[:])
  assert_eq(source.launch(), (32, 1, 128))
  assert_eq(source.parameter_count(), 7)
  assert_true(text.contains("float *log_decay"))
  assert_true(text.contains("log1pf"))
  assert_true(text.contains("__float2bfloat16_rn"))
  assert_true(plan.expression(Beta).root() is Convert(BFloat16, _))
  assert_true(
    plan.expression(LogDecay).root()
    is Binary(
      MultiplyRN,
      Unary(Negate, Unary(NaturalExp, _)),
      Select(GreaterThan, _, Float32Bits(0x41a00000U), _, _)
    ),
  )
}
