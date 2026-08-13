///|
test "one-attempt frame progress retains one absolute deadline and source" {
  let limits = ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let child = spawn(b"/bin/cat", limits)
  let source = FixedArray::make(4, b'x')
  child.begin_write_frame(source, offset=0, byte_count=4, timeout_millis=1000)
  let deadline = child.transfer_deadline_millis
  assert_true(physical_equal(child.transfer_primary, source))
  assert_eq(child.transfer_phase, 0)
  assert_false(child.progress_write_frame())
  assert_eq(child.transfer_phase, 1)
  assert_eq(child.transfer_deadline_millis, deadline)
  for _ in 0..<100000 {
    if child.progress_write_frame() {
      break
    }
  }
  assert_eq(child.transfer_deadline_millis, 0L)
  assert_true(
    physical_equal(child.transfer_primary, child.empty_transfer_buffer),
  )
  assert_true(
    physical_equal(child.transfer_secondary, child.empty_transfer_buffer),
  )
  child.close()
}

///|
test "expired pending frame poisons the child without resetting its deadline" {
  let limits = ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let child = spawn(b"/bin/cat", limits)
  child.begin_read_frame(
    FixedArray::make(4, b'\x55'),
    staging=FixedArray::make(4, b'\x00'),
    timeout_millis=1,
  )
  let deadline = child.transfer_deadline_millis
  let mut now = raw_monotonic_now()
  while now < deadline {
    now = raw_monotonic_now()
  }
  try child.progress_read_frame() catch {
    Timeout(Read) => ()
    _ => fail("unexpected expired frame error")
  } noraise {
    _ => fail("expired frame remained pending")
  }
  assert_eq(child.transfer_deadline_millis, 0L)
  assert_true(
    physical_equal(child.transfer_primary, child.empty_transfer_buffer),
  )
  assert_true(
    physical_equal(child.transfer_secondary, child.empty_transfer_buffer),
  )
  assert_true(child.is_failed())
  try child.shutdown_write() catch {
    InvalidState(ShutdownWrite) => ()
    _ => fail("unexpected poisoned shutdown error")
  } noraise {
    _ => fail("poisoned child accepted shutdown_write")
  }
  child.close()
}

///|
test "legacy write and half-close cannot mutate or cut off an active frame" {
  let limits = ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let child = spawn(b"/bin/cat", limits)
  let source = FixedArray::make(8, b'x')
  child.begin_write_frame(source, offset=0, byte_count=4, timeout_millis=1000)
  assert_eq(read_u32_prefix(child.write_prefix), 4)
  try
    child.write_frame_with_timeout(
      source,
      offset=0,
      byte_count=8,
      timeout_millis=1000,
    )
  catch {
    InvalidState(Write) => ()
    _ => fail("unexpected legacy write overlap error")
  } noraise {
    _ => fail("legacy write overlapped pending frame")
  }
  assert_eq(read_u32_prefix(child.write_prefix), 4)
  try child.shutdown_write() catch {
    InvalidState(ShutdownWrite) => ()
    _ => fail("unexpected active-write shutdown error")
  } noraise {
    _ => fail("shutdown cut off pending frame")
  }
  assert_false(child.is_failed())
  let mut completed = false
  for _ in 0..<100000 {
    if child.progress_write_frame() {
      completed = true
      break
    }
  }
  assert_true(completed)
  child.shutdown_write()
  ignore(child.wait())
  child.close()
}

///|
fn malformed_read_fixture(
  raw : FixedArray[Byte],
  expected_range_error~ : Bool,
) -> Unit raise {
  let limits = ProcessLimits::new(
    max_frame_bytes=64,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let child = spawn(b"/bin/cat", limits)
  child.write_exact(raw, offset=0, byte_count=raw.length())
  child.shutdown_write()
  let destination = FixedArray::make(64, b'\x55')
  let staging = FixedArray::make(64, b'\x00')
  child.begin_read_frame(destination, staging~)
  let mut rejected = false
  for _ in 0..<100000 {
    try child.progress_read_frame() catch {
      InvalidRange(Read) => {
        if !expected_range_error {
          fail("partial frame mapped to range error")
        }
        rejected = true
        break
      }
      ChannelClosed(Read) => {
        if expected_range_error {
          fail("invalid declaration mapped to EOF")
        }
        rejected = true
        break
      }
      _ => fail("unexpected malformed frame error")
    } noraise {
      Some(_) => fail("malformed frame was published")
      None => assert_eq(destination, FixedArray::make(64, b'\x55'))
    }
  }
  assert_true(rejected)
  assert_true(child.is_failed())
  assert_eq(destination, FixedArray::make(64, b'\x55'))
  try child.progress_read_frame() catch {
    InvalidState(Read) => ()
    _ => fail("unexpected poisoned pending reuse error")
  } noraise {
    _ => fail("poisoned pending transfer was reusable")
  }
  try child.read_frame(destination) catch {
    InvalidState(Read) => ()
    _ => fail("unexpected poisoned channel reuse error")
  } noraise {
    _ => fail("poisoned channel accepted another read")
  }
  try child.shutdown_write() catch {
    InvalidState(ShutdownWrite) => ()
    _ => fail("unexpected poisoned shutdown error")
  } noraise {
    _ => fail("poisoned channel accepted shutdown")
  }
  ignore(child.wait())
  child.close()
}

///|
test "malformed frame EOF and declarations poison without publication" {
  malformed_read_fixture([4, 0, 0], expected_range_error=false)
  malformed_read_fixture([4, 0, 0, 0, 7, 9], expected_range_error=false)
  malformed_read_fixture([0, 0, 0, 0], expected_range_error=true)
  malformed_read_fixture([65, 0, 0, 0], expected_range_error=true)
}

///|
fn pending_frame_allocation_positive_control() -> FixedArray[Byte] {
  FixedArray::make(13, b'Z')
}

///|
test "pending-frame allocation gate positive control remains live" {
  let allocated = pending_frame_allocation_positive_control()
  assert_eq(allocated.length(), 13)
  assert_eq(allocated[12], b'Z')
}

///|
test "close clears an active transfer and retained buffer authority" {
  let limits = ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let child = spawn(b"/bin/cat", limits)
  let source = FixedArray::make(4, b'x')
  child.begin_write_frame(source, offset=0, byte_count=4)
  assert_true(physical_equal(child.transfer_primary, source))
  child.close()
  assert_eq(child.transfer_kind, 0)
  assert_true(
    physical_equal(child.transfer_primary, child.empty_transfer_buffer),
  )
  assert_true(
    physical_equal(child.transfer_secondary, child.empty_transfer_buffer),
  )
  try child.progress_write_frame() catch {
    InvalidState(Write) => ()
    _ => fail("unexpected closed transfer error")
  } noraise {
    _ => fail("closed transfer remained usable")
  }
}
