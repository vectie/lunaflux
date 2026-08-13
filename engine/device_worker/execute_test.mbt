///|
test "faulted device worker is a stable nonready lifecycle" {
  assert_true(@device_worker.WorkerFaulted != WorkerReady)
  assert_true(@device_worker.WorkerFaulted != WorkerClosingFailed)
  assert_true(@device_worker.WorkerFaulted != WorkerClosed)
}
