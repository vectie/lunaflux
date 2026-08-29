///|
test "readiness requires the complete singular executor ownership state" {
  assert_true(fp8_resources_publishable_v3(WorkerReady, true, false, true))
  for
    state in [
      (WorkerFaulted, true, false, true),
      (WorkerClosingFailed, true, false, true),
      (WorkerClosed, true, false, true),
      (WorkerReady, false, false, true),
      (WorkerReady, true, true, true),
      (WorkerReady, true, false, false),
    ] {
    let (lifecycle, executor, numeric, context) = state
    assert_false(
      fp8_resources_publishable_v3(lifecycle, executor, numeric, context),
    )
  }
}
