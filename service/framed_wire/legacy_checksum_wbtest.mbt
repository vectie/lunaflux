///|
// Legacy checksums only exist in tests, so structural mutation tests also
// cover old peers. No production encoder or decoder scans frames for them.
fn frame_checksum(bytes : FixedArray[Byte], length : Int) -> UInt {
  let mut hash : UInt = 2166136261
  for index in 0..<length {
    if index < 16 || index >= 20 {
      hash = (hash ^ bytes[index].to_uint()) * 16777619
    }
  }
  hash
}

///|
test "service request ignores legacy checksum words and skips integrity phase" {
  let (source, length) = luna_request_source()
  assert_eq(read_u32(source, 16), 0U)
  for checksum in [0U, 1U, 0xffffffffU, frame_checksum(source, length)] {
    write_u32(source, 16, checksum)
    let workspace = LunaFramedRequestWorkspace::new(
      fixture_limits(),
      LunaFramedRequestStepBudget::new(65536),
    )
    let work = workspace.begin()
    assert_eq(work.offer(source, source_offset=0, length~), 16)
    ignore(work.progress())
    assert_eq(
      work.offer(source, source_offset=16, length=length - 16),
      length - 16,
    )
    assert_eq(workspace.phase, LUNA_FRAMED_REQUEST_HEADER)
    // Reading one header group consumes one unit; no frame-sized checksum
    // phase precedes it, even when the sender supplied a legacy checksum.
    workspace.progress_one()
    assert_eq(workspace.phase_index, 1)
    ignore(work.progress())
    assert_eq(work.state(), LunaFramedRequestReady)
    work.take_view().release()
    ignore(RequestFrameBuffer::new(fixture_limits()).load(source, length))
  }
}

///|
test "service event accepts legacy checksum words without changing semantics" {
  let inference = fixture_inference_limits()
  let event = @inference.StreamEvent::Token(
    @inference.TokenEvent::new(
      token_id=7,
      position=0,
      decoded_delta=Some(
        @inference.DecodedDelta::from_utf8(b"hello", inference),
      ),
      inference,
    ),
  )
  let frame = EventFrameBuffer::new(fixture_limits()).encode(event)
  let source = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(source, destination_offset=0)
  assert_eq(read_u32(source, 16), 0U)
  for checksum in [0U, 1U, 0xffffffffU, frame_checksum(source, source.length())] {
    write_u32(source, 16, checksum)
    match
      EventFrameBuffer::new(fixture_limits())
      .load(source, source.length())
      .event() {
      Token(value) => {
        assert_eq(value.token_id(), 7)
        assert_eq(value.position(), 0)
        assert_eq(value.decoded_delta().unwrap().text(), "hello")
      }
      _ => fail("token event changed kind")
    }
  }
}
