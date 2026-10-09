///|
test "learned text plan has exact sizes and a one-stream text consumer" {
  let plan = @precision_ir.LearnedTextIoPrecision::new(
    vocabulary_size=17,
    hidden_width=8,
    stream_count=4,
    maximum_rows=5,
    maximum_output_rows=2,
    normalization_epsilon=1.0e-5,
    control_epsilon=1.0e-6,
    control_normalization_epsilon=1.0e-5,
    maximum_workspace_bytes=248L,
  )
  assert_eq(plan.text().geometry(), (17, 8, 1, 5, 2))
  assert_eq(plan.input_bytes(), 320L)
  assert_eq(plan.collapse_bytes(), 80L)
  assert_eq(plan.control_weight_bytes(), (512L, 16L, 4L))
  assert_eq(plan.workspace_bytes(), 248L)
  assert_eq(plan.control_epsilon(), 1.0e-6)
  assert_eq(plan.control_normalization_epsilon(), 1.0e-5)
}

///|
test "learned text plan rejects invalid semantics and includes collapse in budget" {
  for
    (streams, control, control_norm, budget, expected_budget) in [
      (1, 1.0e-6, 1.0e-5, 248L, false),
      (17, 1.0e-6, 1.0e-5, 248L, false),
      (4, 0.0, 1.0e-5, 248L, false),
      (4, 1.0e-6, 0.0, 248L, false),
      (4, 1.0e-6, 1.0e-5, 247L, true),
    ] {
    try
      @precision_ir.LearnedTextIoPrecision::new(
        vocabulary_size=17,
        hidden_width=8,
        stream_count=streams,
        maximum_rows=5,
        maximum_output_rows=2,
        normalization_epsilon=1.0e-5,
        control_epsilon=control,
        control_normalization_epsilon=control_norm,
        maximum_workspace_bytes=budget,
      )
    catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement =>
        assert_false(expected_budget)
      @precision_ir.PrecisionPlanError::MemoryBudgetExceeded =>
        assert_true(expected_budget)
      e => raise e
    } noraise {
      _ => fail("invalid learned text plan accepted")
    }
  }
}
