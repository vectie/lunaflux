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
fn finish_pending_write(pending : @process.PendingFrameWrite) -> Unit raise {
  for _ in 0..<100000 {
    if pending.progress() {
      return
    }
  }
  fail("pending frame write did not complete")
}

///|
fn finish_pending_read(pending : @process.PendingFrameRead) -> Int raise {
  for _ in 0..<100000 {
    match pending.progress() {
      Some(byte_count) => return byte_count
      None => ()
    }
  }
  fail("pending frame read did not complete")
}

///|
test "nonblocking frame capabilities preserve one-flight transactional state" {
  let child = @process.spawn(b"/bin/cat", pending_test_limits())
  let source = FixedArray::make(8, b'\x00')
  source[2] = b'l'
  source[3] = b'u'
  source[4] = b'n'
  source[5] = b'a'
  let destination = FixedArray::make(8, b'\x33')
  let staging = FixedArray::make(8, b'\x44')
  let write = child.begin_write_frame(
    source,
    offset=2,
    byte_count=4,
    timeout_millis=1000,
  )
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
  assert_false(write.progress())
  finish_pending_write(write)
  try write.progress() catch {
    InvalidState(Write) => ()
    _ => fail("unexpected completed write error")
  } noraise {
    _ => fail("completed write capability was reusable")
  }
  let read = child.begin_read_frame(destination, staging~, timeout_millis=1000)
  try
    child.begin_write_frame(source, offset=2, byte_count=4, timeout_millis=1000)
  catch {
    InvalidState(Write) => ()
    _ => fail("unexpected read-then-write overlap error")
  } noraise {
    _ => fail("write overlapped pending read")
  }
  assert_eq(destination, FixedArray::make(8, b'\x33'))
  assert_eq(finish_pending_read(read), 4)
  assert_eq(destination[0], b'l')
  assert_eq(destination[1], b'u')
  assert_eq(destination[2], b'n')
  assert_eq(destination[3], b'a')
  assert_eq(destination[4], b'\x33')
  try read.progress() catch {
    InvalidState(Read) => ()
    _ => fail("unexpected completed read error")
  } noraise {
    _ => fail("completed read capability was reusable")
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
  let pending = child.begin_read_frame(
    destination,
    staging~,
    timeout_millis=1000,
  )
  let mut failed = false
  for _ in 0..<100000 {
    try pending.progress() catch {
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
