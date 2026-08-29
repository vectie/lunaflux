///|
test "tensor parallel failure publication is payload safe" {
  assert_eq(
    tensor_parallel_worker_failure_reason(ProtocolViolation(PlanSequence)),
    ExecutionFailed,
  )
  assert_eq(
    tensor_parallel_worker_failure_reason(RankExecutionFailure),
    ExecutionFailed,
  )
  let unavailable : Array[@rank_group_protocol.RankGroupFailureReason] = [
    DeadlineExpired,
    RankLost,
    CollectiveFailure,
    HostFailure,
  ]
  for reason in unavailable {
    assert_eq(
      tensor_parallel_worker_failure_reason(reason),
      ResourceUnavailable,
    )
  }
}
