///|
test "online capacity preflight accepts the exact largest payload envelope" {
  let limits = FramedWireLimits::new(
    fixture_inference_limits(),
    max_frame_bytes=EVENT_HEADER_BYTES + 64,
  )
  let writer = CanonicalEventWriter::new(limits)
  writer.preflight_online_capacity(64)
  writer.preflight_online_capacity(16)
  assert_true(writer.has_credit())
}

///|
test "online capacity rejection preserves pinned credit and frame" {
  let limits = FramedWireLimits::new(
    fixture_inference_limits(),
    max_frame_bytes=EVENT_HEADER_BYTES + 63,
  )
  let writer = CanonicalEventWriter::new(limits)
  let source = FixedArray::make(1, b'x')
  writer.prepare_token_bytes(
    token_id=1,
    position=0,
    source,
    source_offset=0,
    length=1,
  )
  let pinned = writer_bytes(writer)
  try writer.preflight_online_capacity(64) catch {
    InvalidFrame(frame=EventFrame, rule=Length) => ()
    _ => fail("unexpected online capacity error")
  } noraise {
    _ => fail("one-byte-short online capacity passed")
  }
  assert_false(writer.has_credit())
  assert_fixed_bytes_equal(writer_bytes(writer), pinned)
}

///|
test "online capacity preflight rejects negative delta bounds" {
  let writer = CanonicalEventWriter::new(fixture_limits())
  try writer.preflight_online_capacity(-1) catch {
    InvalidFrame(frame=EventFrame, rule=Length) => ()
    _ => fail("unexpected negative online capacity error")
  } noraise {
    _ => fail("negative online delta capacity passed")
  }
  assert_true(writer.has_credit())
}

///|
test "scalar token writer rejects without consuming credit" {
  let writer = CanonicalEventWriter::new(fixture_limits())
  let source = FixedArray::make(1, b'x')
  assert_false(
    writer.prepare_token_bytes_status(
      token_id=fixture_inference_limits().max_token_id() + 1,
      position=0,
      source,
      source_offset=0,
      length=1,
    ),
  )
  assert_true(writer.has_credit())
  assert_true(
    writer.prepare_token_bytes_status(
      token_id=1,
      position=0,
      source,
      source_offset=0,
      length=1,
    ),
  )
  assert_false(writer.has_credit())
}
