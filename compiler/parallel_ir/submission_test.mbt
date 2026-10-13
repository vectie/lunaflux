///|
test "submission fences preserve every compute interval and collective cut" {
  for policy in [RetirementPolicy::EachCollective, WholeFrame] {
    let plan = OrderedSubmissionPlan::new(
      kernel_count=7,
      [1, 3, 6],
      retirement=policy,
    )
    assert_eq(plan.interval_count(), 4)
    assert_eq(plan.interval(0), (0, 1))
    assert_eq(plan.interval(1), (1, 3))
    assert_eq(plan.interval(2), (3, 6))
    assert_eq(plan.interval(3), (6, 7))
    assert_true(plan.retirement_policy() == policy)
  }
  for cuts in [[], [0], [7], [3, 3], [4, 2]] {
    try
      OrderedSubmissionPlan::new(kernel_count=7, cuts, retirement=WholeFrame)
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid ordered cut accepted")
    }
  }
}
