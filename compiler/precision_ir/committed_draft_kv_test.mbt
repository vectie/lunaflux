///|
test "committed draft read set retains all future draft rows and physical ring order" {
  let p = @precision_ir.CommittedDraftKvPrecision::new(
    query_heads=64,
    head_width=512,
    window_tokens=128,
    maximum_main_rows=256,
    maximum_draft_rows=5,
    maximum_positions=1048576,
    maximum_state_bytes=458776L,
  )
  assert_eq(p.state_bytes(), 458776L)
  assert_eq(p.buffer_bytes()[0], 131072L)
  assert_eq(p.buffer_bytes()[3], 327680L)
  assert_eq(p.visible_main_slots(1), 2)
  assert_eq(p.visible_main_slots(127), 128)
  assert_eq(p.visible_main_slots(128), 128)
  assert_eq(p.visible_main_slots(1048575), 128)
  try p.visible_main_slots(0) catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    e => raise e
  } noraise {
    _ => fail("initial prefill is not draft attention")
  }
  try
    @precision_ir.CommittedDraftKvPrecision::new(
      query_heads=64,
      head_width=512,
      window_tokens=128,
      maximum_main_rows=256,
      maximum_draft_rows=5,
      maximum_positions=1048576,
      maximum_state_bytes=458775L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    e => raise e
  } noraise {
    _ => fail("budget accepted")
  }
}
