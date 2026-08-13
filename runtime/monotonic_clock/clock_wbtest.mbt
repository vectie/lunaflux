///|
test "native failures map to bounded public errors" {
  assert_eq(map_native_error(NativeUnavailable), Unavailable)
  assert_eq(map_native_error(NativeOutOfRange), OutOfRange)
  debug_inspect(Unavailable, content="Unavailable")
  debug_inspect(OutOfRange, content="OutOfRange")
}
