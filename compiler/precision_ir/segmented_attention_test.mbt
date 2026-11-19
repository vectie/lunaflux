///|
test "segmented attention retains score and normalized probability rounds" {
  let plan = @precision_ir.SegmentedAttentionPrecision::new(
    maximum_rows=1024,
    maximum_segment=1024,
    head_count=16,
    head_width=72,
  )
  assert_eq(plan.geometry(), (1024, 1024, 16, 72))
  assert_true(
    plan.expression(ScaledScore).root()
    is Convert(
      Float32,
      Convert(
        BFloat16,
        Binary(
          MultiplyRN,
          Convert(Float32, Convert(BFloat16, Input(0))),
          Input(1)
        )
      )
    ),
  )
  assert_true(
    plan.expression(ProbabilityStorage).root()
    is Convert(Float32, Convert(BFloat16, Binary(DivideRN, Input(0), Input(1)))),
  )
  assert_true(plan.expression(Publish).root() is Convert(BFloat16, Input(0)))
  assert_true(
    (try? @precision_ir.SegmentedAttentionPrecision::new(
      maximum_rows=16,
      maximum_segment=17,
      head_count=1,
      head_width=72,
    ))
    is Err(_),
  )
  assert_true(
    (try? @precision_ir.SegmentedAttentionPrecision::new(
      maximum_rows=2147483647,
      maximum_segment=1,
      head_count=16,
      head_width=72,
    ))
    is Err(_),
  )
}
