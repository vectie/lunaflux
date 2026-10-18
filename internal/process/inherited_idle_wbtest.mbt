///|
test "idle request wait starts finite deadline only at first byte" {
  let channel = open_inherited(
    ProcessLimits::new(
      max_frame_bytes=64,
      io_timeout_millis=1000,
      shutdown_timeout_millis=1000,
    ),
  )
  let destination = FixedArray::make(64, b'\x00')
  let staging = FixedArray::make(64, b'\x00')
  channel.begin_wait_frame_or_eof(destination, staging~)
  assert_eq(channel.read_transfer_deadline_millis, 0L)
  assert_true(channel.read_transfer_allows_eof)
  channel.start_waiting_frame_deadline()
  let started = channel.read_transfer_deadline_millis
  assert_true(started > 0L)
  channel.start_waiting_frame_deadline()
  assert_eq(channel.read_transfer_deadline_millis, started)
  channel.clear_pending_transfers()
  channel.begin_read_frame_or_eof(destination, staging~)
  assert_true(channel.read_transfer_deadline_millis > 0L)
  channel.clear_pending_transfers()
}
