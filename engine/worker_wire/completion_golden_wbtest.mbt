///|
test "completion kind wire tags are pinned independently in both directions" {
  assert_eq(completion_kind_to_wire(PrefillCompletion), 0)
  assert_eq(completion_kind_to_wire(FinalPrefillCompletion), 1)
  assert_eq(completion_kind_to_wire(DecodeCompletion), 2)
  assert_eq(completion_kind_to_wire(FailedCompletion), 3)
  assert_eq(completion_kind_from_wire(0), PrefillCompletion)
  assert_eq(completion_kind_from_wire(1), FinalPrefillCompletion)
  assert_eq(completion_kind_from_wire(2), DecodeCompletion)
  assert_eq(completion_kind_from_wire(3), FailedCompletion)
}

///|
test "worker failure wire tags are pinned independently in both directions" {
  assert_eq(worker_failure_to_wire(ExecutionFailed), 0)
  assert_eq(worker_failure_to_wire(ResourceUnavailable), 1)
  assert_eq(worker_failure_to_wire(CancelledBeforeExecution), 2)
  assert_eq(worker_failure_from_wire(0), ExecutionFailed)
  assert_eq(worker_failure_from_wire(1), ResourceUnavailable)
  assert_eq(worker_failure_from_wire(2), CancelledBeforeExecution)
}
