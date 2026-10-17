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

///|
test "replay budgeting is explicit and overflow checked" {
  let plan = OrderedSubmissionPlan::new(
    kernel_count=7,
    [1, 3, 6],
    retirement=WholeFrame,
    replay=CaptureIntervals,
  )
  assert_eq(
    plan.replay_reserve_bytes(bytes_per_kernel=100L, bytes_per_interval=10L),
    740L,
  )
  try
    plan.replay_reserve_bytes(
      bytes_per_kernel=9223372036854775807L,
      bytes_per_interval=1L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("overflowed graph reserve")
  }
}

///|
test "collective-inclusive replay requires whole-frame retirement" {
  try
    OrderedSubmissionPlan::new(
      kernel_count=3,
      [1, 2],
      retirement=EachCollective,
      replay=CapturePaddedCollectives,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("per-site capture retirement accepted")
  }
  let p = OrderedSubmissionPlan::new(
    kernel_count=3,
    [1, 2],
    retirement=WholeFrame,
    replay=CapturePaddedCollectives,
  )
  assert_eq(p.interval_count(), 3)
  assert_eq(
    p.replay_reserve_bytes(bytes_per_kernel=100L, bytes_per_interval=10L),
    330L,
  )
}

///|
test "collective buckets distinguish decode from prefill with bounded storage" {
  let p = OrderedSubmissionPlan::new(
    kernel_count=3,
    [1, 2],
    retirement=WholeFrame,
    replay=CaptureBucketedCollectives,
  )
  assert_eq(p.collective_row_buckets(64), [1, 64])
  assert_eq(p.collective_row_buckets(1), [1])
  assert_eq(p.collective_bucket(1), 0)
  assert_eq(p.collective_bucket(2), 1)
  assert_eq(p.collective_bucket(64), 1)
  assert_eq(
    p.replay_reserve_bytes(bytes_per_kernel=100L, bytes_per_interval=10L),
    660L,
  )
  try
    OrderedSubmissionPlan::new(
      kernel_count=3,
      [1, 2],
      retirement=EachCollective,
      replay=CaptureBucketedCollectives,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("bucket graphs require a whole-frame retirement fence")
  }
}
