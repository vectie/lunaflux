///|
test "independent producers join before a collective and preserve interval coordinates" {
  let plan = @parallel_ir.OrderedSubmissionPlan::new(
    kernel_count=12,
    [7],
    retirement=WholeFrame,
    replay=CaptureIntervals,
  )
  let fork = @parallel_ir.ForkJoinSpan::new(first=2, split=4, join=7)
  plan.validate_forks([fork])
  assert_eq(fork.shift(8).range(), (10, 12, 15))
  assert_eq(fork.shift(-2).range(), (0, 2, 5))
}

///|
test "reject malformed overlapping and collective-straddling forks" {
  let plan = @parallel_ir.OrderedSubmissionPlan::new(
    kernel_count=12,
    [7],
    retirement=WholeFrame,
  )
  assert_true(
    (try? @parallel_ir.ForkJoinSpan::new(first=2, split=2, join=7)) is Err(_),
  )
  let crossing = @parallel_ir.ForkJoinSpan::new(first=2, split=4, join=8)
  assert_true((try? plan.validate_forks([crossing])) is Err(_))
  let left = @parallel_ir.ForkJoinSpan::new(first=0, split=1, join=3)
  let right = @parallel_ir.ForkJoinSpan::new(first=2, split=4, join=7)
  assert_true((try? plan.validate_forks([left, right])) is Err(_))
}
