///|
test "retained row budgets include metadata and an unreadable empty-cache sentinel" {
  for (capacity, state_bytes) in [(3, 124L), (0, 60L)] {
    let plan = @precision_ir.RetainedRowsPrecision::new(
      word_width=16,
      position_stride=4,
      maximum_input_rows=2,
      maximum_stored_rows=capacity,
      maximum_state_bytes=state_bytes,
      maximum_workspace_bytes=12L,
    )
    assert_eq(plan.geometry(), (16, 4, 2, capacity))
    assert_eq(
      plan.state_bytes().fold(init=0L, (total, bytes) => total + bytes),
      state_bytes,
    )
    assert_eq(plan.workspace_bytes(), 12L)
    assert_eq(plan.effect(), @precision_ir.NumericEffect::CacheCommit)
    try
      @precision_ir.RetainedRowsPrecision::new(
        word_width=16,
        position_stride=4,
        maximum_input_rows=2,
        maximum_stored_rows=capacity,
        maximum_state_bytes=state_bytes - 1L,
        maximum_workspace_bytes=12L,
      )
    catch {
      @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
      error => raise error
    } noraise {
      _ => fail("state budget omitted metadata/sentinel")
    }
  }
  for
    (stride, rows, capacity) in [
      (0, 2, 3),
      (4, 0, 3),
      (4, 2, -1),
      (4, 2, 262145),
    ] {
    try
      @precision_ir.RetainedRowsPrecision::new(
        word_width=16,
        position_stride=stride,
        maximum_input_rows=rows,
        maximum_stored_rows=capacity,
        maximum_state_bytes=104857600L,
        maximum_workspace_bytes=12L,
      )
    catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement => ()
      error => raise error
    } noraise {
      _ => fail("illegal retained geometry")
    }
  }
}
