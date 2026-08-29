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
test "private socketpair child echoes one exact bounded frame" {
  let child = spawn(b"/bin/cat", test_limits())
  let source = FixedArray::make(16, b'\x7f')
  source[4] = b'l'
  source[5] = b'u'
  source[6] = b'n'
  source[7] = b'a'
  let destination = FixedArray::make(16, b'\x33')
  child.write_frame(source, offset=4, byte_count=4)
  assert_eq(child.read_frame(destination), 4)
  assert_eq(destination[0], b'l')
  assert_eq(destination[1], b'u')
  assert_eq(destination[2], b'n')
  assert_eq(destination[3], b'a')
  assert_eq(destination[4], b'\x33')
  child.shutdown_write()
  let exit = child.wait()
  assert_eq(exit.kind(), 0)
  assert_eq(exit.code(), 0)
  child.close()
  child.close()
  assert_true(child.is_closed())
}

///|
test "explicit frame timeout overrides steady limit and rejects before mutation" {
  let child = spawn(b"/bin/cat", test_limits(io_timeout_millis=5))
  let source = FixedArray::make(4, b'x')
  let destination = FixedArray::make(4, b'\x00')
  try
    child.write_frame_with_timeout(
      source,
      offset=0,
      byte_count=4,
      timeout_millis=0,
    )
  catch {
    InvalidRange(Write) => ()
    _ => fail("unexpected explicit-timeout validation error")
  } noraise {
    _ => fail("invalid explicit timeout was accepted")
  }
  assert_false(child.is_failed())
  child.write_frame_with_timeout(
    source,
    offset=0,
    byte_count=4,
    timeout_millis=1000,
  )
  assert_eq(child.read_frame_with_timeout(destination, timeout_millis=1000), 4)
  assert_eq(destination, source)
  child.shutdown_write()
  ignore(child.wait())
  child.close()
}

///|
test "read timeout poisons framing but preserves cleanup authority" {
  let child = spawn(b"/bin/cat", test_limits(io_timeout_millis=5))
  let buffer = FixedArray::make(8, b'\x00')
  try child.read_frame(buffer) catch {
    Timeout(Read) => ()
    _ => fail("unexpected child read timeout")
  } noraise {
    _ => fail("idle child read did not time out")
  }
  assert_true(child.is_failed())
  try child.write_frame(buffer, offset=0, byte_count=1) catch {
    InvalidState(Write) => ()
    _ => fail("unexpected poisoned child error")
  } noraise {
    _ => fail("poisoned child accepted another frame")
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
    child.write_frame(source, offset=0, byte_count=1)
    assert_eq(child.read_frame(destination), 1)
    assert_eq(destination[0], b'x')
    child.shutdown_write()
    let exit = child.wait()
    assert_eq(exit.kind(), 0)
    assert_eq(exit.code(), 0)
    child.close()
  }
}
