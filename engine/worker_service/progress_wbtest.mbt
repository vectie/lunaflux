///|
test "release allocation gate retains service progress symbols" {
  let _ = WorkerService::progress
  let _ = WorkerService::commit_received
  assert_eq(Bytes::make(17, b'Q').length(), 17)
  assert_eq(owned_preparation_allocation_positive_control().length(), 19)
}

///|
fn owned_preparation_allocation_positive_control() -> Bytes {
  Bytes::make(19, b'R')
}
