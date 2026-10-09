///|
test "quantized low-rank prefix retains physical bytes and distinct rounding" {
  let plan = @precision_ir.QuantizedQueryKvPrecision::new(
    hidden_width=256,
    query_rank=128,
    head_count=2,
    head_width=128,
    maximum_rows=3,
    normalization_epsilon=1.0e-6,
    activation_amax_floor=0.0001,
    maximum_workspace_bytes=6144L,
  )
  assert_eq(plan.geometry(), (256, 128, 2, 128, 3))
  assert_eq(plan.frame_bytes()[:].to_owned(), [
    768L, 768L, 1536L, 1536L, 768L, 768L,
  ])
  assert_eq(plan.weight_bytes()[:].to_owned(), [
    32768L, 2L, 256L, 32768L, 2L, 32768L, 2L, 256L,
  ])
  assert_eq(plan.workspace_bytes(), 6144L)
  assert_eq(plan.activation_quantization().amax_floor(), 0.0001)
  assert_eq(plan.head_normalization(), Bf16SquareMeanEpsilonInverseProduct)
  try
    @precision_ir.QuantizedQueryKvPrecision::new(
      hidden_width=256,
      query_rank=128,
      head_count=2,
      head_width=128,
      maximum_rows=3,
      normalization_epsilon=1.0e-6,
      activation_amax_floor=0.0001,
      maximum_workspace_bytes=6143L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    error => raise error
  } noraise {
    _ => fail("undersized aggregate workspace accepted")
  }
}

///|
test "quantized query geometry refuses overflow and unsupported matrix shapes" {
  for (hidden, heads) in [(255, 2), (256, 513), (256, 512)] {
    try
      @precision_ir.QuantizedQueryKvPrecision::new(
        hidden_width=hidden,
        query_rank=128,
        head_count=heads,
        head_width=256,
        maximum_rows=3,
        normalization_epsilon=1.0e-6,
        activation_amax_floor=0.0001,
        maximum_workspace_bytes=9223372036854775807L,
      )
    catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement => ()
      error => raise error
    } noraise {
      _ => fail("invalid head/matrix extent accepted")
    }
  }
}
