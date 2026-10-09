///|
test "grouped attention output bounds all intermediates and distinct numeric stages" {
  let plan = @precision_ir.GroupedAttentionOutputPrecision::new(
    head_count=64,
    head_width=512,
    rotary_width=64,
    group_count=8,
    output_rank=1024,
    hidden_width=4096,
    maximum_rows=3,
    maximum_positions=1048576,
    frequency=Base(theta=10000.0),
    maximum_workspace_bytes=1000000L,
  )
  assert_eq(plan.geometry(), (8, 4096, 1024, 3))
  assert_eq(plan.frame_bytes()[:], [196608L, 49152L, 24576L])
  assert_eq(plan.workspace_bytes(), 270336L)
  assert_true(plan.rotary().inverse())
  assert_eq(plan.weight_representations()[0].shape(), (8192, 4096))
  assert_eq(plan.weight_representations()[1].shape(), (4096, 8192))
  try
    @precision_ir.GroupedAttentionOutputPrecision::new(
      head_count=64,
      head_width=512,
      rotary_width=64,
      group_count=8,
      output_rank=1024,
      hidden_width=4096,
      maximum_rows=3,
      maximum_positions=1048576,
      frequency=Base(theta=10000.0),
      maximum_workspace_bytes=270335L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    error => raise error
  } noraise {
    _ => fail("one-byte-short output budget accepted")
  }
  try
    @precision_ir.RotaryTensorPrecision::new(
      head_count=2,
      head_width=128,
      rotary_width=3,
      maximum_rows=1,
      maximum_positions=64,
      frequency=Base(theta=10000.0),
      inverse=true,
      maximum_workspace_bytes=512L,
    )
  catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    error => raise error
  } noraise {
    _ => fail("odd rotary suffix accepted")
  }
}
