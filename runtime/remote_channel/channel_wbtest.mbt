///|
async fn sockets() -> (@socket.Tcp, @socket.Tcp) {
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
  )
  defer listener.close()
  let client = @socket.Tcp::connect(listener.addr)
  errdefer client.close()
  let (peer, _) = listener.accept()
  (client, peer)
}

///|
async test "fragmented header and body publish atomically and preserve destination tail" {
  let (client, peer) = sockets()
  defer client.close()
  let channel = RemoteChannel::new(
    peer,
    max_frame_bytes=16,
    timeout_millis=1000UL,
  )
  defer channel.close()
  let output = FixedArray::make(16, b'x')
  channel.begin_receive(0UL)
  assert_eq(channel.poll_receive(output, 0UL), None)
  for fragment in [b"\x05", b"\x00", b"\x00", b"\x00", b"he", b"ll"] {
    client.write(fragment)
    @async.sleep(1)
    assert_eq(channel.poll_receive(output, 0UL), None)
    assert_eq(output, FixedArray::make(16, b'x'))
  }
  client.write(b"o")
  @async.sleep(1)
  assert_eq(channel.poll_receive(output, 0UL), Some(5))
  assert_eq(Bytes::makei(5, i => output[i]), b"hello")
  assert_eq(output[5], b'x')
  channel.begin_receive(1UL)
  client.write(b"\x03\x00\x00\x00bye")
  @async.sleep(1)
  assert_eq(channel.poll_receive(output, 1UL), None)
  assert_eq(channel.poll_receive(output, 1UL), Some(3))
  assert_eq(Bytes::makei(3, i => output[i]), b"bye")
}

///|
async test "send snapshots source and both lanes support independent full duplex progress" {
  let (client, peer) = sockets()
  let left = RemoteChannel::new(
    client,
    max_frame_bytes=16,
    timeout_millis=1000UL,
  )
  defer left.close()
  let right = RemoteChannel::new(
    peer,
    max_frame_bytes=16,
    timeout_millis=1000UL,
  )
  defer right.close()
  let source = FixedArray::make(7, b'a')
  let output_left = FixedArray::make(16, b'x')
  let output_right = FixedArray::make(16, b'x')
  left.begin_send(source, 7, 0UL)
  source[0] = b'b'
  right.begin_send(source, 7, 0UL)
  left.begin_receive(0UL)
  right.begin_receive(0UL)
  assert_true(left.poll_send(0UL))
  assert_true(right.poll_send(0UL))
  @async.sleep(1)
  assert_eq(left.poll_receive(output_left, 0UL), None)
  assert_eq(right.poll_receive(output_right, 0UL), None)
  assert_eq(left.poll_receive(output_left, 0UL), Some(7))
  assert_eq(right.poll_receive(output_right, 0UL), Some(7))
  assert_eq(output_right[0], b'a')
  assert_eq(output_left[0], b'b')
}

///|
async test "malformed or oversize lengths close the stream before publishing" {
  for prefix in [b"\x00\x00\x00\x00", b"\xff\xff\xff\xff", b"\x11\x00\x00\x00"] {
    let (client, peer) = sockets()
    defer client.close()
    let channel = RemoteChannel::new(
      peer,
      max_frame_bytes=16,
      timeout_millis=1000UL,
    )
    defer channel.close()
    let output = FixedArray::make(16, b'x')
    channel.begin_receive(0UL)
    client.write(prefix)
    @async.sleep(1)
    try channel.poll_receive(output, 0UL) catch {
      RemoteChannelError::InvalidFrame => ()
      error => fail("unexpected: \{Repr(error)}")
    } noraise {
      _ => fail("invalid prefix accepted")
    }
    assert_false(channel.is_open())
    assert_eq(output, FixedArray::make(16, b'x'))
  }
}

///|
async test "absolute deadline and monotonic regression close idle readers" {
  for regressed in [false, true] {
    let (client, peer) = sockets()
    defer client.close()
    let channel = RemoteChannel::new(
      peer,
      max_frame_bytes=16,
      timeout_millis=10UL,
    )
    defer channel.close()
    let output = FixedArray::make(16, b'x')
    channel.begin_receive(100UL)
    for _ in 0..<32 {
      assert_eq(channel.poll_receive(output, 109UL), None)
    }
    try
      channel.poll_receive(output, if regressed { 108UL } else { 110UL })
    catch {
      RemoteChannelError::ClockRegressed => assert_true(regressed)
      RemoteChannelError::TimedOut => assert_false(regressed)
      error => fail("unexpected: \{Repr(error)}")
    } noraise {
      _ => fail("deadline or clock regression ignored")
    }
    assert_false(channel.is_open())
    try channel.begin_receive(111UL) catch {
      InvalidState => ()
      _ => fail("unexpected")
    } noraise {
      _ => fail("closed stream revived")
    }
  }
}

///|
async test "EOF during a body discards partial payload" {
  let (client, peer) = sockets()
  let channel = RemoteChannel::new(
    peer,
    max_frame_bytes=16,
    timeout_millis=1000UL,
  )
  defer channel.close()
  let output = FixedArray::make(16, b'x')
  channel.begin_receive(0UL)
  client.write(b"\x05\x00\x00\x00hi")
  client.close()
  @async.sleep(1)
  assert_eq(channel.poll_receive(output, 0UL), None)
  assert_eq(channel.poll_receive(output, 0UL), None)
  try channel.poll_receive(output, 0UL) catch {
    RemoteChannelError::Disconnected => ()
    error => fail("unexpected: \{Repr(error)}")
  } noraise {
    _ => fail("truncated frame accepted")
  }
  assert_false(channel.is_open())
  assert_eq(output, FixedArray::make(16, b'x'))
}

///|
async test "optional boundary EOF never accepts a partial next frame" {
  for fragment in [b"", b"\x05", b"\x05\x00\x00\x00hi"] {
    let (client, peer) = sockets()
    let channel = RemoteChannel::new(
      peer,
      max_frame_bytes=16,
      timeout_millis=1000UL,
    )
    defer channel.close()
    let output = FixedArray::make(16, b'x')
    channel.begin_receive(0UL, idle=true)
    if !fragment.is_empty() {
      client.write(fragment)
    }
    client.close()
    @async.sleep(1)
    let mut failed = false
    for _ in 0..<8 {
      ignore(channel.poll_receive(output, 0UL, allow_idle_eof=true)) catch {
        RemoteChannelError::Disconnected => {
          failed = true
          break
        }
        error => raise error
      }
      if !channel.is_open() {
        break
      }
      @async.sleep(1)
    }
    assert_false(channel.is_open())
    assert_eq(failed, !fragment.is_empty())
    assert_eq(output, FixedArray::make(16, b'x'))
  }
}

///|
async test "idle service lifetime is unbounded but a started frame has an absolute deadline" {
  let (client, peer) = sockets()
  defer client.close()
  let channel = RemoteChannel::new(
    peer,
    max_frame_bytes=16,
    timeout_millis=10UL,
  )
  defer channel.close()
  let output = FixedArray::make(16, b'x')
  channel.begin_receive(0UL, idle=true)
  assert_eq(channel.poll_receive(output, 100000UL), None)
  assert_true(channel.is_open())
  client.write(b"\x05")
  @async.sleep(1)
  assert_eq(channel.poll_receive(output, 100000UL), None)
  assert_eq(channel.poll_receive(output, 100009UL), None)
  try channel.poll_receive(output, 100010UL) catch {
    TimedOut => ()
    error => raise error
  } noraise {
    _ => fail("started frame extended its deadline")
  }
  assert_false(channel.is_open())
  assert_eq(output, FixedArray::make(16, b'x'))
}
