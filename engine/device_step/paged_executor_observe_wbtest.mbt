///|
fn expect_offline_selection_rejected(
  token_ids : ArrayView[Int],
  destination_length : Int,
) -> Unit raise {
  try
    validate_offline_logit_selection(3000, token_ids, destination_length)
  catch {
    InvalidExecutor(SamplingOutput) => ()
    _ => fail("unexpected selected-logit validation error")
  } noraise {
    _ => fail("invalid selected-logit observation was admitted")
  }
}

///|
test "offline selected-logit observer is fixed-capacity and exact" {
  validate_offline_logit_selection(3000, [0, 42, 1031, 2999], 4)
  expect_offline_selection_rejected([], 0)
  expect_offline_selection_rejected([0], 0)
  expect_offline_selection_rejected([-1], 1)
  expect_offline_selection_rejected([3000], 1)
  expect_offline_selection_rejected(FixedArray::make(33, 0), 33)
}

///|
test "offline selected-logit observer requires the exact completed owner" {
  let closed = dummy_paged_executor(GraphClosed)
  let closed_execution : ExecutedPagedGraphExecution = {
    executor: closed,
    epoch: 1UL,
    ordered_executor_index: 0,
  }
  try
    closed.observe_offline_selected_logits(
      closed_execution,
      0,
      [0],
      FixedArray::make(1, 0.0),
    )
  catch {
    InvalidExecutorLifecycle(expected=GraphExecuted, actual=GraphClosed) => ()
    _ => fail("unexpected closed observer error")
  } noraise {
    _ => fail("closed executor exposed selected logits")
  }
}
