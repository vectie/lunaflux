///|
test "dense SwiGLU owns one shared product frame and preserves separate weights" {
  let plan = @precision_ir.DenseSwiGluPrecision::new(
    hidden_width=4096,
    intermediate_width=12288,
    maximum_rows=32,
  )
  assert_eq(plan.geometry(), (4096, 12288, 32))
  assert_eq(plan.weight_bytes(), 100663296L)
  assert_eq(plan.workspace_bytes(), 786432L)
  try
    @precision_ir.DenseSwiGluPrecision::new(
      hidden_width=0,
      intermediate_width=1,
      maximum_rows=1,
    )
  catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    e => raise e
  } noraise {
    _ => fail("zero width accepted")
  }
}
