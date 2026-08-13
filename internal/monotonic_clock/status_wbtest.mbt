///|
test "native statuses map without errno or clock payload" {
  assert_eq(finish_native_read(NATIVE_OK, 17UL), 17UL)
  try finish_native_read(NATIVE_UNAVAILABLE, 0UL) catch {
    NativeUnavailable => ()
    _ => fail("unexpected unavailable mapping")
  } noraise {
    _ => fail("expected native unavailable failure")
  }
  try finish_native_read(NATIVE_OUT_OF_RANGE, 0UL) catch {
    NativeOutOfRange => ()
    _ => fail("unexpected range mapping")
  } noraise {
    _ => fail("expected native range failure")
  }
  try finish_native_read(99, 0UL) catch {
    NativeUnavailable => ()
    _ => fail("unexpected unknown-status mapping")
  } noraise {
    _ => fail("expected fail-closed unknown status")
  }
}
