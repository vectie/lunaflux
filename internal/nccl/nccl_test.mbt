///|
test "runtime probe is bounded on hosts without NCCL" {
  let _ : (@cuda.Context) -> Unit = fn(_context) { () }
  let inventory = @nccl.probe()
  assert_true(inventory.version_code() >= 0)
}

///|
test "communicator identity import is exactly 128 bytes" {
  let id = @nccl.communicator_id_from_bytes(Bytes::make(128, b'\x5a'))
  assert_eq(id.to_bytes(), Bytes::make(128, b'\x5a'))
  try @nccl.communicator_id_from_bytes(b"short") catch {
    InvalidArgument(CommunicatorIdCreate) => ()
    _ => fail("unexpected communicator identity failure")
  } noraise {
    _ => fail("short communicator identity was admitted")
  }
}
