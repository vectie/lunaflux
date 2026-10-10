///|
test "projection norm preserves packed representation and separate BF16 boundaries" {
  let plan = @precision_ir.RowProjectionNormPrecision::new(
    input_width=12288,
    output_width=4096,
    maximum_rows=3,
    normalization_epsilon=1.0e-6,
    activation_amax_floor=0.0001,
    maximum_workspace_bytes=49152L,
  )
  assert_eq(plan.geometry(), (12288, 4096, 3))
  assert_eq(plan.input_bytes(), 73728L)
  assert_eq(plan.frame_bytes(), 24576L)
  assert_eq(plan.workspace_bytes(), 49152L)
  assert_eq(plan.weight_representation().payload_bytes(), 50331648L)
  assert_eq(plan.weight_representation().scale_bytes(), 3072L)
  assert_eq(plan.activation_quantization().amax_floor(), 0.0001)
  try
    @precision_ir.RowProjectionNormPrecision::new(
      input_width=12288,
      output_width=4096,
      maximum_rows=3,
      normalization_epsilon=1.0e-6,
      activation_amax_floor=0.0001,
      maximum_workspace_bytes=49151L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    e => raise e
  } noraise {
    _ => fail("budget allowed one extra byte")
  }
  for width in [0, 127, 65537] {
    try
      @precision_ir.RowProjectionNormPrecision::new(
        input_width=width,
        output_width=128,
        maximum_rows=1,
        normalization_epsilon=1.0e-6,
        activation_amax_floor=0.0001,
        maximum_workspace_bytes=512L,
      )
    catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement => ()
      e => raise e
    } noraise {
      _ => fail("invalid matrix width")
    }
  }
}
