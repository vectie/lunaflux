///|
async test "spawned serving reads buffered terminal before owner progress and drain" {
  let inference = spawned_serving_wb_limits()
  let framed = @framed_wire.FramedWireLimits::new(
    inference,
    max_frame_bytes=4096,
  )
  let decoder = @framed_wire.EventFrameBuffer::new(framed)
  let usage = @inference.Usage::initial(
    input_tokens=1,
    cached_input_tokens=0,
    inference,
  ).advance(output_tokens=2, inference)
  let encoded = decoder.encode(
    Completed(
      @inference.CompletedEvent::new(
        finish_reason=TokenLimit,
        final_usage=usage,
      ),
    ),
  )
  let frame = FixedArray::make(encoded.length(), b'\x00')
  encoded.copy_to(frame, destination_offset=0)
  let wire = Bytes::makei(frame.length(), index => frame[index])

  let listener = @socket.TcpServer(
    @socket.Addr::parse("127.0.0.1:0"),
    reuse_addr=true,
  )
  defer listener.close()
  let client = @socket.Tcp::connect(listener.addr)
  defer client.close()
  let (server_connection, _) = listener.accept()
  defer server_connection.close()
  let mut written = 0
  while written < wire.length() {
    let count = server_connection.write_once(
      wire,
      offset=written,
      len=wire.length() - written,
    )
    if count <= 0 {
      abort("buffered terminal fixture write stalled")
    }
    written += count
  }

  // Any eager owner progress closes this synthetic draining owner and makes the
  // read fail at health. A buffered terminal must be consumed first.
  let owner = empty_owner_for_state_test(Draining)
  let storage = FixedArray::make(decoder.capacity(), b'\x00')
  let event = read_spawned_serving_event(owner, client, storage, decoder, 100)
  match event {
    Completed(completed) => {
      assert_eq(completed.finish_reason(), TokenLimit)
      assert_eq(completed.final_usage(), usage)
    }
    _ => abort("buffered terminal decoded as the wrong event")
  }
  assert_eq(owner.phase(), Draining)
  assert_eq(owner.progress(), RuntimeClosedProgress)
  assert_eq(owner.phase(), Closed)
  assert_true(owner.cleanup_complete())
}
