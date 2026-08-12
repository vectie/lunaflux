///|
test "checked startup arithmetic accepts boundaries without wrapping" {
  assert_eq(checked_add_nonnegative(2_147_483_646, 1), Some(2_147_483_647))
  assert_eq(checked_add_nonnegative(2_147_483_647, 1), None)
  assert_eq(checked_multiply_nonnegative(46_340, 46_340), Some(2_147_395_600))
  assert_eq(checked_multiply_nonnegative(46_341, 46_341), None)
  assert_eq(ceiling_divide_positive(2_147_483_647, 2), Some(1_073_741_824))
  assert_eq(ceiling_divide_positive(2_147_483_647, 2_147_483_647), Some(1))
  assert_eq(ceiling_divide_positive(0, 1), None)
  assert_eq(ceiling_divide_positive(1, 0), None)
}

///|
test "defensive canonical rechecks keep their exact bounded reason" {
  try require_at_least(7, 8, WorkerCompletionSlots) catch {
    error => assert_eq(error, Invalid(WorkerCompletionSlots))
  } noraise {
    _ => fail("expected completion-slot rejection")
  }
  assert_eq(checked_add_nonnegative(MAX_SAFE_INT, 1), None)
}
