///|
test "learned index query precision composes exact scratch and immutable scale" {
  let transform = @precision_ir.RotaryHadamardFp4Precision::new(
    head_count=2,
    head_width=128,
    rotary_width=64,
    maximum_rows=3,
    maximum_positions=1024,
    frequency=Base(theta=10000.0),
    simulation_block_width=32,
    maximum_workspace_bytes=3072L,
  )
  let plan = @precision_ir.LearnedIndexQueriesPrecision::new(
    hidden_width=128,
    query_rank=128,
    head_count=2,
    transform,
    maximum_workspace_bytes=4632L,
  )
  assert_eq(plan.geometry(), (128, 128, 2, 128, 3))
  assert_eq(plan.frame_bytes()[:].to_owned(), [1536L, 12L, 12L])
  assert_eq(plan.workspace_bytes(), 4632L)
  assert_eq(plan.weight_bytes()[:].to_owned(), [32768L, 2L, 512L])
  assert_eq(plan.head_scale(), 1.0 / 128.0.sqrt() * (1.0 / 2.0.sqrt()))
  assert_eq(plan.effect(), @precision_ir.NumericEffect::StepWrite)
  try
    @precision_ir.LearnedIndexQueriesPrecision::new(
      hidden_width=128,
      query_rank=128,
      head_count=2,
      transform,
      maximum_workspace_bytes=4631L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    error => raise error
  } noraise {
    _ => fail("aggregate budget was not checked")
  }
  try
    @precision_ir.LearnedIndexQueriesPrecision::new(
      hidden_width=128,
      query_rank=128,
      head_count=1,
      transform,
      maximum_workspace_bytes=4632L,
    )
  catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    error => raise error
  } noraise {
    _ => fail("mismatched head ownership accepted")
  }
}
