///|
test "rotary Hadamard FP4 plan has exact two-stage BF16 frames without device names" {
  let plan = @precision_ir.RotaryHadamardFp4Precision::new(
    head_count=2,
    head_width=128,
    rotary_width=64,
    maximum_rows=3,
    maximum_positions=1024,
    frequency=Base(theta=10000.0),
    simulation_block_width=32,
    maximum_workspace_bytes=3072L,
  )
  assert_eq(plan.rotary().geometry(), (2, 128, 64, 3, 1024))
  assert_eq(plan.frame_bytes().length(), 2)
  assert_eq(plan.frame_bytes()[0], 1536L)
  assert_eq(plan.frame_bytes()[1], 1536L)
  assert_eq(plan.workspace_bytes(), 3072L)
  assert_eq(plan.block_width(), 32)
  assert_eq(plan.effect(), @precision_ir.NumericEffect::StepWrite)
  for
    (width, block, budget) in [
      (96, 32, 3072L),
      (128, 24, 3072L),
      (128, 32, 3071L),
    ] {
    try
      @precision_ir.RotaryHadamardFp4Precision::new(
        head_count=2,
        head_width=width,
        rotary_width=64,
        maximum_rows=3,
        maximum_positions=1024,
        frequency=Base(theta=10000.0),
        simulation_block_width=block,
        maximum_workspace_bytes=budget,
      )
    catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement if width != 128 ||
        block != 32 => ()
      @precision_ir.PrecisionPlanError::MemoryBudgetExceeded if budget == 3071L =>
        ()
      error => raise error
    } noraise {
      _ => fail("illegal geometry or short aggregate budget accepted")
    }
  }
}
