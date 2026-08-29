///|
extern "c" fn raw_test_lifecycle(cycles : Int) -> Int = "lunaflux_nccl_test_lifecycle"

///|
test "fake NCCL seam balances generation sequence failure and close" {
  @cuda.link_collective_interop()
  let _ : (@cuda.Context) -> Unit = fn(_context) { () }
  assert_eq(raw_test_lifecycle(512), 0)
}

///|
test "fake NCCL seam rejects adversarial cycle bounds" {
  @cuda.link_collective_interop()
  assert_eq(raw_test_lifecycle(0), -2)
  assert_eq(raw_test_lifecycle(1000001), -2)
}

///|
test "payload-safe status mapping is complete" {
  let cases : Array[(Int, NcclError)] = [
    (-1, Unavailable(RuntimeAdmission)),
    (-2, InvalidArgument(RuntimeAdmission)),
    (-3, Closed(RuntimeAdmission)),
    (-4, Busy(RuntimeAdmission)),
    (-5, RuntimeFailure(RuntimeAdmission)),
    (-6, VersionMismatch(RuntimeAdmission)),
    (-7, GenerationMismatch(RuntimeAdmission)),
    (-8, SequenceMismatch(RuntimeAdmission)),
    (-9, Failed(RuntimeAdmission)),
  ]
  for entry in cases {
    let (status, expected) = entry
    try check_status(status, RuntimeAdmission) catch {
      error => assert_eq(error, expected)
    } noraise {
      _ => fail("native failure status was accepted")
    }
  }
}
