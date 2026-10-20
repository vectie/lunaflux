///|
test "pending header transition can already have a readable payload" {
  with_inherited_frame(1, () => {
    let channel = open_inherited(
      ProcessLimits::new(
        max_frame_bytes=64,
        io_timeout_millis=1000,
        shutdown_timeout_millis=1000,
      ),
    )
    let input = FixedArray::make(64, b'\x00')
    let staging = FixedArray::make(64, b'\x00')
    channel.begin_wait_frame_or_eof(input, staging~)
    assert_eq(channel.read_transfer_deadline_millis, 0L)
    assert_true(channel.wait_readable(maximum_wait_millis=1))
    assert_eq(channel.read_transfer_deadline_millis, 0L)
    // None means no whole frame yet, not necessarily EAGAIN. The header and
    // payload are already in the same pipe fixture; waiting on a timer here
    // would impose a delay even though the second read can complete now.
    assert_eq(channel.progress_read_frame_or_eof(), None)
    assert_eq(channel.read_transfer_phase, 1)
    let deadline = channel.read_transfer_deadline_millis
    assert_true(deadline > 0L)
    assert_true(channel.wait_readable(maximum_wait_millis=1))
    assert_eq(channel.read_transfer_deadline_millis, deadline)
    assert_eq(input[0], b'\x00')
    assert_eq(channel.progress_read_frame_or_eof(), Some(2))
    assert_eq(input[0], b'\x07')
    assert_eq(input[1], b'\x09')
    channel.begin_wait_frame_or_eof(input, staging~)
    assert_true(channel.wait_readable(maximum_wait_millis=1))
    assert_eq(channel.progress_read_frame_or_eof(), Some(0))
  })
}
