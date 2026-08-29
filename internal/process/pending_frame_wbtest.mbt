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
  let deadline = child.write_transfer_deadline_millis
  assert_true(physical_equal(child.write_transfer_source, source))
  assert_eq(child.write_transfer_phase, 0)
  assert_false(child.progress_write_frame())
  assert_eq(child.write_transfer_phase, 1)
  assert_eq(child.write_transfer_deadline_millis, deadline)
  for _ in 0..<100000 {
    if child.progress_write_frame() {
      break
    }
  }
  assert_eq(child.write_transfer_deadline_millis, 0L)
  assert_true(
    physical_equal(child.write_transfer_source, child.empty_transfer_buffer),
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
  let source = FixedArray::make(4, b'x')
  child.begin_write_frame(source, offset=0, byte_count=4, timeout_millis=1000)
  child.begin_read_frame(
    FixedArray::make(4, b'\x55'),
    staging=FixedArray::make(4, b'\x00'),
    timeout_millis=1,
  )
  let deadline = child.read_transfer_deadline_millis
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
  assert_eq(child.read_transfer_deadline_millis, 0L)
  assert_eq(child.write_transfer_deadline_millis, 0L)
  assert_false(child.write_transfer_active)
  assert_false(child.read_transfer_active)
  assert_true(
    physical_equal(child.write_transfer_source, child.empty_transfer_buffer),
  )
  assert_true(
    physical_equal(child.read_transfer_destination, child.empty_transfer_buffer),
  )
  assert_true(
    physical_equal(child.read_transfer_staging, child.empty_transfer_buffer),
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
test "close clears both active transfers and retained buffer authority" {
  let limits = ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let child = spawn(b"/bin/cat", limits)
  let source = FixedArray::make(4, b'x')
  let destination = FixedArray::make(4, b'y')
  let staging = FixedArray::make(4, b'z')
  child.begin_write_frame(source, offset=0, byte_count=4)
  child.begin_read_frame(destination, staging~)
  assert_true(physical_equal(child.write_transfer_source, source))
  assert_true(physical_equal(child.read_transfer_destination, destination))
  child.close()
  assert_false(child.write_transfer_active)
  assert_false(child.read_transfer_active)
  assert_true(
    physical_equal(child.write_transfer_source, child.empty_transfer_buffer),
  )
  assert_true(
    physical_equal(child.read_transfer_destination, child.empty_transfer_buffer),
  )
  assert_true(
    physical_equal(child.read_transfer_staging, child.empty_transfer_buffer),
  )
  try child.progress_write_frame() catch {
    InvalidState(Write) => ()
    _ => fail("unexpected closed transfer error")
  } noraise {
    _ => fail("closed transfer remained usable")
  }
}

///|
test "inherited owner admits one pending transfer per direction" {
  let limits = ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let channel = open_inherited(limits)
  let source = FixedArray::make(8, b'x')
  let destination = FixedArray::make(8, b'y')
  let staging = FixedArray::make(8, b'z')
  channel.begin_write_frame(source, offset=0, byte_count=8)
  channel.begin_read_frame_or_eof(destination, staging~)
  assert_true(channel.write_transfer_active)
  assert_true(channel.read_transfer_active)
  assert_true(channel.read_transfer_allows_eof)
  try channel.begin_write_frame(source, offset=0, byte_count=8) catch {
    InvalidState(Write) => ()
    _ => fail("unexpected inherited duplicate write error")
  } noraise {
    _ => fail("inherited duplicate write was admitted")
  }
  try channel.begin_read_frame(destination, staging~) catch {
    InvalidState(Read) => ()
    _ => fail("unexpected inherited duplicate read error")
  } noraise {
    _ => fail("inherited duplicate read was admitted")
  }
  try channel.progress_read_frame() catch {
    InvalidState(Read) => ()
    _ => fail("wrong inherited EOF progress mode was accepted")
  } noraise {
    _ => fail("inherited read mode was substitutable")
  }
  channel.clear_pending_transfers()
}

///|
test "inherited timeout revokes both pending directions" {
  let limits = ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let channel = open_inherited(limits)
  let source = FixedArray::make(8, b'x')
  let destination = FixedArray::make(8, b'y')
  let staging = FixedArray::make(8, b'z')
  channel.begin_write_frame(source, offset=0, byte_count=8)
  channel.begin_read_frame(destination, staging~, timeout_millis=1)
  let deadline = channel.read_transfer_deadline_millis
  let mut now = raw_monotonic_now()
  while now < deadline {
    now = raw_monotonic_now()
  }
  try channel.progress_read_frame() catch {
    Timeout(Read) => ()
    _ => fail("unexpected inherited timeout error")
  } noraise {
    _ => fail("expired inherited transfer remained pending")
  }
  assert_true(channel.is_failed())
  assert_false(channel.write_transfer_active)
  assert_false(channel.read_transfer_active)
  assert_true(
    physical_equal(channel.write_transfer_source, channel.empty_transfer_buffer),
  )
  assert_true(
    physical_equal(
      channel.read_transfer_destination,
      channel.empty_transfer_buffer,
    ),
  )
  assert_true(
    physical_equal(channel.read_transfer_staging, channel.empty_transfer_buffer),
  )

  let write_channel = open_inherited(limits)
  write_channel.begin_write_frame(
    source,
    offset=0,
    byte_count=8,
    timeout_millis=1,
  )
  let write_deadline = write_channel.write_transfer_deadline_millis
  let mut write_now = raw_monotonic_now()
  while write_now < write_deadline {
    write_now = raw_monotonic_now()
  }
  try write_channel.progress_write_frame() catch {
    Timeout(Write) => ()
    _ => fail("unexpected inherited write timeout error")
  } noraise {
    _ => fail("expired inherited write remained pending")
  }
  assert_true(write_channel.is_failed())
  assert_false(write_channel.write_transfer_active)
}

///|
test "child duplex alias rejection preserves active opposite direction" {
  let limits = ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let read_owner = spawn(b"/bin/cat", limits)
  let destination = FixedArray::make(8, b'd')
  let staging = FixedArray::make(8, b's')
  read_owner.begin_read_frame(destination, staging~)
  try read_owner.begin_write_frame(destination, offset=0, byte_count=8) catch {
    InvalidRange(Write) => ()
    _ => fail("unexpected child destination/write alias error")
  } noraise {
    _ => fail("child destination/write alias was admitted")
  }
  try read_owner.write_frame(staging, offset=0, byte_count=8) catch {
    InvalidRange(Write) => ()
    _ => fail("unexpected child staging/write alias error")
  } noraise {
    _ => fail("child staging/write alias was admitted")
  }
  assert_eq(read_owner.startup_write_frame_status(staging, 8, 1000), 1)
  assert_true(read_owner.read_transfer_active)
  assert_true(physical_equal(read_owner.read_transfer_destination, destination))
  assert_true(physical_equal(read_owner.read_transfer_staging, staging))
  assert_false(read_owner.is_failed())
  read_owner.close()

  let write_owner = spawn(b"/bin/cat", limits)
  let source = FixedArray::make(8, b'w')
  let other = FixedArray::make(8, b'o')
  write_owner.begin_write_frame(source, offset=0, byte_count=8)
  try write_owner.begin_read_frame(source, staging=other) catch {
    InvalidRange(Read) => ()
    _ => fail("unexpected child source/destination alias error")
  } noraise {
    _ => fail("child source/destination alias was admitted")
  }
  try write_owner.begin_read_frame(other, staging=source) catch {
    InvalidRange(Read) => ()
    _ => fail("unexpected child source/staging alias error")
  } noraise {
    _ => fail("child source/staging alias was admitted")
  }
  try write_owner.read_frame(source) catch {
    InvalidRange(Read) => ()
    _ => fail("unexpected child blocking read alias error")
  } noraise {
    _ => fail("child blocking read alias was admitted")
  }
  assert_eq(write_owner.startup_read_frame_status(source, 1000), -2)
  assert_true(write_owner.write_transfer_active)
  assert_true(physical_equal(write_owner.write_transfer_source, source))
  assert_false(write_owner.is_failed())
  write_owner.close()
}

///|
test "inherited duplex alias rejection preserves active opposite direction" {
  let limits = ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let read_owner = open_inherited(limits)
  let destination = FixedArray::make(8, b'd')
  let staging = FixedArray::make(8, b's')
  read_owner.begin_read_frame(destination, staging~)
  try read_owner.begin_write_frame(destination, offset=0, byte_count=8) catch {
    InvalidRange(Write) => ()
    _ => fail("unexpected inherited destination/write alias error")
  } noraise {
    _ => fail("inherited destination/write alias was admitted")
  }
  try read_owner.write_frame(staging, offset=0, byte_count=8) catch {
    InvalidRange(Write) => ()
    _ => fail("unexpected inherited staging/write alias error")
  } noraise {
    _ => fail("inherited staging/write alias was admitted")
  }
  assert_true(read_owner.read_transfer_active)
  assert_true(physical_equal(read_owner.read_transfer_destination, destination))
  assert_true(physical_equal(read_owner.read_transfer_staging, staging))
  assert_false(read_owner.is_failed())
  read_owner.clear_pending_transfers()

  let write_owner = open_inherited(limits)
  let source = FixedArray::make(8, b'w')
  let other = FixedArray::make(8, b'o')
  write_owner.begin_write_frame(source, offset=0, byte_count=8)
  try write_owner.begin_read_frame(source, staging=other) catch {
    InvalidRange(Read) => ()
    _ => fail("unexpected inherited source/destination alias error")
  } noraise {
    _ => fail("inherited source/destination alias was admitted")
  }
  try write_owner.begin_read_frame_or_eof(other, staging=source) catch {
    InvalidRange(Read) => ()
    _ => fail("unexpected inherited source/staging alias error")
  } noraise {
    _ => fail("inherited source/staging alias was admitted")
  }
  try write_owner.read_frame(source) catch {
    InvalidRange(Read) => ()
    _ => fail("unexpected inherited blocking read alias error")
  } noraise {
    _ => fail("inherited blocking read alias was admitted")
  }
  try write_owner.read_frame_or_eof(source) catch {
    InvalidRange(Read) => ()
    _ => fail("unexpected inherited EOF read alias error")
  } noraise {
    _ => fail("inherited EOF read alias was admitted")
  }
  assert_true(write_owner.write_transfer_active)
  assert_true(physical_equal(write_owner.write_transfer_source, source))
  assert_false(write_owner.is_failed())
  write_owner.clear_pending_transfers()
}
