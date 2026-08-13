///|
test "release allocation gate retains service progress symbols" {
  let _ = WorkerService::progress
  let _ = WorkerService::commit_received
  assert_eq(Bytes::make(17, b'Q').length(), 17)
}
