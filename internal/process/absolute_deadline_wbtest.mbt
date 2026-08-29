///|
test "write and ordered read share one caller absolute frame deadline" {
  let limits = ProcessLimits::new(
    max_frame_bytes=4096,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  let child = spawn(b"/bin/cat", limits)
  let source = FixedArray::make(4, b'x')
  let destination = FixedArray::make(4, b'\x55')
  let staging = FixedArray::make(4, b'\x00')
  let deadline = raw_monotonic_now() + 1000L
  child.begin_write_frame_until(
    source,
    offset=0,
    byte_count=4,
    deadline_millis=deadline,
  )
  child.begin_read_frame_until(destination, staging~, deadline_millis=deadline)
  assert_eq(child.write_transfer_deadline_millis, deadline)
  assert_eq(child.read_transfer_deadline_millis, deadline)
  child.close()
}

///|
test "expired absolute read deadline poisons before consuming a response" {
  let child = spawn(
    b"/bin/cat",
    ProcessLimits::new(
      max_frame_bytes=4096,
      io_timeout_millis=1000,
      shutdown_timeout_millis=1000,
    ),
  )
  let destination = FixedArray::make(4, b'\x55')
  let staging = FixedArray::make(4, b'\x00')
  try
    child.begin_read_frame_until(
      destination,
      staging~,
      deadline_millis=raw_monotonic_now(),
    )
  catch {
    Timeout(Read) => ()
    _ => fail("unexpected expired absolute read error")
  } noraise {
    _ => fail("expired absolute read was admitted")
  }
  assert_true(child.is_failed())
  assert_eq(destination, FixedArray::make(4, b'\x55'))
  child.close()
}

///|
test "absolute frame deadline remains bounded by process timeout policy" {
  let child = spawn(
    b"/bin/cat",
    ProcessLimits::new(
      max_frame_bytes=4096,
      io_timeout_millis=1000,
      shutdown_timeout_millis=1000,
    ),
  )
  try
    child.begin_write_frame_until(
      FixedArray::make(4, b'x'),
      offset=0,
      byte_count=4,
      deadline_millis=raw_monotonic_now() + 700000L,
    )
  catch {
    InvalidRange(Write) => ()
    _ => fail("unexpected distant absolute deadline error")
  } noraise {
    _ => fail("distant absolute deadline was admitted")
  }
  assert_false(child.is_failed())
  child.close()
}
