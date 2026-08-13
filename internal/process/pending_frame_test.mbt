///|
fn pending_test_limits(
  io_timeout_millis? : Int = 1000,
) -> @process.ProcessLimits raise @process.ProcessError {
  @process.ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis~,
    shutdown_timeout_millis=1000,
  )
}

///|
fn finish_pending_write(child : @process.ChildProcess) -> Unit raise {
  for _ in 0..<100000 {
    if child.progress_write_frame() {
      return
    }
  }
  fail("pending frame write did not complete")
}

///|
fn finish_pending_read(child : @process.ChildProcess) -> Int raise {
  for _ in 0..<100000 {
    match child.progress_read_frame() {
      Some(byte_count) => return byte_count
      None => ()
    }
  }
  fail("pending frame read did not complete")
}

///|
test "owner-resident nonblocking frames preserve one-flight state" {
  let child = @process.spawn(b"/bin/cat", pending_test_limits())
  let source = FixedArray::make(8, b'\x00')
  source[2] = b'l'
  source[3] = b'u'
  source[4] = b'n'
  source[5] = b'a'
  let destination = FixedArray::make(8, b'\x33')
  let staging = FixedArray::make(8, b'\x44')
  child.begin_write_frame(source, offset=2, byte_count=4, timeout_millis=1000)
  try
    child.begin_write_frame(source, offset=2, byte_count=4, timeout_millis=1000)
  catch {
    InvalidState(Write) => ()
    _ => fail("unexpected concurrent write error")
  } noraise {
    _ => fail("second frame write was admitted")
  }
  try child.begin_read_frame(destination, staging~, timeout_millis=1000) catch {
    InvalidState(Read) => ()
    _ => fail("unexpected write-then-read overlap error")
  } noraise {
    _ => fail("read overlapped pending write")
  }
  assert_false(child.progress_write_frame())
  finish_pending_write(child)
  try child.progress_write_frame() catch {
    InvalidState(Write) => ()
    _ => fail("unexpected completed write error")
  } noraise {
    _ => fail("completed write transfer was reusable")
  }
  child.begin_read_frame(destination, staging~, timeout_millis=1000)
  try
    child.begin_write_frame(source, offset=2, byte_count=4, timeout_millis=1000)
  catch {
    InvalidState(Write) => ()
    _ => fail("unexpected read-then-write overlap error")
  } noraise {
    _ => fail("write overlapped pending read")
  }
  assert_eq(destination, FixedArray::make(8, b'\x33'))
  assert_eq(finish_pending_read(child), 4)
  assert_eq(destination[0], b'l')
  assert_eq(destination[1], b'u')
  assert_eq(destination[2], b'n')
  assert_eq(destination[3], b'a')
  assert_eq(destination[4], b'\x33')
  try child.progress_read_frame() catch {
    InvalidState(Read) => ()
    _ => fail("unexpected completed read error")
  } noraise {
    _ => fail("completed read transfer was reusable")
  }
  child.shutdown_write()
  ignore(child.wait())
  child.close()
}

///|
test "transactional read keeps destination unchanged on channel failure" {
  let child = @process.spawn(b"/usr/bin/true", pending_test_limits())
  let destination = FixedArray::make(8, b'\x55')
  let staging = FixedArray::make(8, b'\x00')
  child.begin_read_frame(destination, staging~, timeout_millis=1000)
  let mut failed = false
  for _ in 0..<100000 {
    try child.progress_read_frame() catch {
      ChannelClosed(Read) => {
        failed = true
        break
      }
      _ => fail("unexpected transactional read failure")
    } noraise {
      Some(_) => fail("empty child published a frame")
      None => assert_eq(destination, FixedArray::make(8, b'\x55'))
    }
  }
  assert_true(failed)
  assert_eq(destination, FixedArray::make(8, b'\x55'))
  assert_true(child.is_failed())
  ignore(child.wait())
  child.close()
}

///|
test "transactional read rejects aliased or undersized staging before I/O" {
  let child = @process.spawn(b"/bin/cat", pending_test_limits())
  let destination = FixedArray::make(8, b'\x00')
  try child.begin_read_frame(destination, staging=destination) catch {
    InvalidRange(Read) => ()
    _ => fail("unexpected aliased staging error")
  } noraise {
    _ => fail("aliased staging was admitted")
  }
  try
    child.begin_read_frame(destination, staging=FixedArray::make(7, b'\x00'))
  catch {
    InvalidRange(Read) => ()
    _ => fail("unexpected undersized staging error")
  } noraise {
    _ => fail("undersized staging was admitted")
  }
  assert_false(child.is_failed())
  child.close()
}
