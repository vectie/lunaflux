///|
fn test_limits(
  io_timeout_millis? : Int = 1000,
) -> ProcessLimits raise ProcessError {
  ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis~,
    shutdown_timeout_millis=1000,
  )
}

///|
test "private socketpair child echoes exact fixed-buffer ranges" {
  let child = spawn(b"/bin/cat", test_limits())
  let source = FixedArray::make(16, b'\x7f')
  source[4] = b'l'
  source[5] = b'u'
  source[6] = b'n'
  source[7] = b'a'
  let destination = FixedArray::make(16, b'\x33')
  child.write_exact(source, offset=4, byte_count=4)
  child.read_exact(destination, offset=6, byte_count=4)
  assert_eq(destination[5], b'\x33')
  assert_eq(destination[6], b'l')
  assert_eq(destination[7], b'u')
  assert_eq(destination[8], b'n')
  assert_eq(destination[9], b'a')
  assert_eq(destination[10], b'\x33')
  child.shutdown_write()
  let exit = child.wait()
  assert_eq(exit.kind(), 0)
  assert_eq(exit.code(), 0)
  child.close()
  child.close()
  assert_true(child.is_closed())
}

///|
test "read timeout and invalid ranges preserve cleanup authority" {
  let child = spawn(b"/bin/cat", test_limits(io_timeout_millis=5))
  let buffer = FixedArray::make(8, b'\x00')
  try child.read_exact(buffer, offset=0, byte_count=1) catch {
    Timeout(Read) => ()
    _ => fail("unexpected child read timeout")
  } noraise {
    _ => fail("idle child read did not time out")
  }
  try child.write_exact(buffer, offset=7, byte_count=2) catch {
    InvalidRange(Write) => ()
    _ => fail("unexpected child range error")
  } noraise {
    _ => fail("invalid child range was accepted")
  }
  try child.wait() catch {
    Timeout(Wait) => ()
    _ => fail("unexpected live-child wait result")
  } noraise {
    _ => fail("live child exited before shutdown")
  }
  child.terminate()
  let exit = child.wait()
  assert_eq(exit.kind(), 1)
  assert_eq(exit.code(), 15)
  child.close()
}

///|
test "spawn failures are payload safe" {
  try spawn(b"/definitely/not/a/lunaflux/worker", test_limits()) catch {
    SpawnFailed => ()
    _ => fail("unexpected spawn failure")
  } noraise {
    _ => fail("missing worker executable was spawned")
  }
}

///|
test "repeated child lifecycle balances exact echo and reap" {
  let source = FixedArray::make(1, b'x')
  let destination = FixedArray::make(1, b'\x00')
  for _ in 0..<32 {
    let child = spawn(b"/bin/cat", test_limits())
    child.write_exact(source, offset=0, byte_count=1)
    child.read_exact(destination, offset=0, byte_count=1)
    assert_eq(destination[0], b'x')
    child.shutdown_write()
    let exit = child.wait()
    assert_eq(exit.kind(), 0)
    assert_eq(exit.code(), 0)
    child.close()
  }
}
