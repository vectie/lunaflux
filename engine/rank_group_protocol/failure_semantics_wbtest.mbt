///|
fn wb_failure_report(
  reason : RankGroupFailureReason,
  failed_rank : Int,
) -> RankGroupFailureReport {
  RankGroupFailureReport(1UL, 7UL, reason, failed_rank, 3)
}

///|
test "failure publication rejects impossible reason and failed-rank pairs" {
  for reason in [DeadlineExpired, HostFailure] {
    assert_true(wb_failure_report(reason, -1).has_canonical_failure_rank())
    assert_false(wb_failure_report(reason, 0).has_canonical_failure_rank())
  }
  for reason in [RankLost, CollectiveFailure, RankExecutionFailure] {
    assert_false(wb_failure_report(reason, -1).has_canonical_failure_rank())
    for rank in 0..<3 {
      assert_true(wb_failure_report(reason, rank).has_canonical_failure_rank())
    }
    assert_false(wb_failure_report(reason, 3).has_canonical_failure_rank())
  }
  let protocol = ProtocolViolation(PlanSequence)
  assert_true(wb_failure_report(protocol, -1).has_canonical_failure_rank())
  assert_true(wb_failure_report(protocol, 2).has_canonical_failure_rank())
  assert_false(wb_failure_report(protocol, -2).has_canonical_failure_rank())
  assert_false(wb_failure_report(protocol, 3).has_canonical_failure_rank())
}
