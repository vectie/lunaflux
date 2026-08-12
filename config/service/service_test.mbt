///|
test "service accepts inclusive numeric boundaries" {
  let low = @service.new("127.0.0.1", 1, 1024, 0, Normal)
  assert_eq(low.listen_port(), 1)
  assert_eq(low.max_request_bytes(), 1024)
  assert_eq(low.graceful_drain_milliseconds(), 0)
  let high = @service.new("::1", 65535, 16777216, 600000, Verbose)
  assert_eq(high.listen_port(), 65535)
  assert_eq(high.diagnostic_mode(), Verbose)
}

///|
test "service rejects invalid port and controlled host" {
  try @service.new("host\nname", 8080, 4096, 1000, Normal) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=ListenHost, issue=ContainsControlCharacter)",
      )
  } noraise {
    _ => fail("expected host rejection")
  }
  try @service.new("localhost", 65536, 4096, 1000, Normal) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=ListenPort, issue=AboveMaximum)",
      )
  } noraise {
    _ => fail("expected port rejection")
  }
}
