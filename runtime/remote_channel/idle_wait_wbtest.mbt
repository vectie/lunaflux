///|
async test "idle receive suspends, preserves framing and starts a fresh deadline" {
  let (client, peer) = sockets()
  defer client.close()
  let channel = RemoteChannel::new(
    peer,
    max_frame_bytes=16,
    timeout_millis=10UL,
  )
  defer channel.close()
  channel.begin_receive(0UL, idle=true)
  let output = FixedArray::make(16, b'x')
  @async.with_task_group(group => {
    group.spawn_bg(() => {
      @async.sleep(5)
      client.write(b"\x03\x00\x00\x00abc")
    })
    assert_true(channel.wait_idle_receive())
  })
  assert_eq(channel.receive_cursor, 1)
  assert_eq(output, FixedArray::make(16, b'x'))
  assert_false(channel.wait_idle_receive())
  assert_eq(channel.poll_receive(output, 100000UL), None)
  assert_eq(channel.receive_deadline, 100010UL)
  assert_eq(channel.poll_receive(output, 100001UL), Some(3))
  assert_eq(output[0], b'a')
  assert_eq(output[1], b'b')
  assert_eq(output[2], b'c')
}

///|
async test "first idle prefix byte cannot extend the absolute transfer deadline" {
  let (client, peer) = sockets()
  defer client.close()
  let channel = RemoteChannel::new(
    peer,
    max_frame_bytes=16,
    timeout_millis=10UL,
  )
  defer channel.close()
  channel.begin_receive(0UL, idle=true)
  client.write(b"\x03")
  assert_true(channel.wait_idle_receive())
  let output = FixedArray::make(16, b'x')
  assert_eq(channel.poll_receive(output, 100000UL), None)
  assert_eq(channel.poll_receive(output, 100009UL), None)
  try channel.poll_receive(output, 100010UL) catch {
    TimedOut => ()
    error => raise error
  } noraise {
    _ => fail("partial prefix must time out")
  }
  assert_false(channel.is_open())
  assert_eq(output, FixedArray::make(16, b'x'))
}

///|
async test "idle wait cancellation retires the pending read and closes transport" {
  let (client, peer) = sockets()
  defer client.close()
  let channel = RemoteChannel::new(
    peer,
    max_frame_bytes=16,
    timeout_millis=10UL,
  )
  defer channel.close()
  channel.begin_receive(0UL, idle=true)
  assert_eq(@async.with_timeout_opt(5, () => channel.wait_idle_receive()), None)
  assert_false(channel.is_open())
  assert_false(channel.reading)
}

///|
async test "idle wait respects orderly EOF policy" {
  for allowed in [false, true] {
    let (client, peer) = sockets()
    let channel = RemoteChannel::new(
      peer,
      max_frame_bytes=16,
      timeout_millis=10UL,
    )
    defer channel.close()
    channel.begin_receive(0UL, idle=true)
    client.close()
    try channel.wait_idle_receive(allow_idle_eof=allowed) catch {
      Disconnected => assert_false(allowed)
      error => raise error
    } noraise {
      waited => assert_true(allowed && waited)
    }
    assert_false(channel.is_open())
    assert_false(channel.reading)
  }
}
