///|
fn request_reader_fixture() -> (FixedArray[Byte], Int) raise {
  let frame = RequestFrameBuffer::new(fixture_limits()).encode(
    fixture_request(),
  )
  let source = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(source, destination_offset=0)
  (source, source.length())
}

///|
fn feed_request_range(
  reader : IncrementalRequestReader,
  source : FixedArray[Byte],
  start : Int,
  end : Int,
) -> Unit raise {
  let mut cursor = start
  while cursor < end {
    let capacity = reader.next_capacity()
    let remaining = end - cursor
    let amount = if remaining < capacity { remaining } else { capacity }
    reader.append(source, source_offset=cursor, length=amount)
    cursor += amount
  }
}

///|
fn assert_reader_poisoned(reader : IncrementalRequestReader) -> Unit raise {
  try reader.next_capacity() catch {
    RequestReaderPoisoned => ()
    _ => fail("unexpected poisoned-reader error")
  } noraise {
    _ => fail("poisoned reader remained usable")
  }
}

///|
fn assert_prefix_rejected(
  source : FixedArray[Byte],
  rule : FramedWireRule,
) -> Unit raise {
  let reader = IncrementalRequestReader::new(fixture_limits())
  try reader.append(source, source_offset=0, length=16) catch {
    InvalidFrame(frame=RequestFrame, rule=actual) => assert_eq(actual, rule)
    _ => fail("unexpected request-prefix error")
  } noraise {
    _ => fail("invalid request prefix was accepted")
  }
  assert_reader_poisoned(reader)
}

///|
test "incremental request accepts bytewise and every split point" {
  let (source, length) = request_reader_fixture()
  let bytewise = IncrementalRequestReader::new(fixture_limits())
  assert_eq(bytewise.next_capacity(), 16)
  for index in 0..<length {
    bytewise.append(source, source_offset=index, length=1)
    if index < 15 {
      assert_eq(bytewise.next_capacity(), 15 - index)
    }
  }
  assert_eq(bytewise.next_capacity(), 0)
  assert_eq(bytewise.take().request().request_id().value(), 42UL)
  for split in 0..<=length {
    let reader = IncrementalRequestReader::new(fixture_limits())
    feed_request_range(reader, source, 0, split)
    reader.append(source, source_offset=split, length=0)
    feed_request_range(reader, source, split, length)
    assert_eq(reader.next_capacity(), 0)
    assert_eq(reader.take().request().request_id().value(), 42UL)
  }
}

///|
test "empty append is inert and truncated input remains appendable" {
  let (source, length) = request_reader_fixture()
  let reader = IncrementalRequestReader::new(fixture_limits())
  reader.append(source, source_offset=0, length=0)
  assert_eq(reader.next_capacity(), 16)
  feed_request_range(reader, source, 0, length - 1)
  try reader.take() catch {
    RequestReaderIncomplete => ()
    _ => fail("unexpected truncated-frame error")
  } noraise {
    _ => fail("truncated frame was taken")
  }
  feed_request_range(reader, source, length - 1, length)
  assert_eq(reader.take().request().request_id().value(), 42UL)
}

///|
test "request prefix rejects magic version kind and hostile lengths" {
  let (source, _) = request_reader_fixture()
  source[0] = b'X'
  assert_prefix_rejected(source, Magic)
  let (source, _) = request_reader_fixture()
  write_u32(source, 4, 2)
  assert_prefix_rejected(source, Version)
  let (source, _) = request_reader_fixture()
  write_u32(source, 8, 2)
  assert_prefix_rejected(source, Kind)
  let (source, _) = request_reader_fixture()
  write_u32(source, 12, 0)
  assert_prefix_rejected(source, Length)
  let (source, _) = request_reader_fixture()
  write_u32(source, 12, 0x7fffffff)
  assert_prefix_rejected(source, Length)
  let (source, _) = request_reader_fixture()
  write_u32(source, 12, 0x80000000)
  assert_prefix_rejected(source, Length)
  let (source, _) = request_reader_fixture()
  write_u32(source, 12, 0xffffffff)
  assert_prefix_rejected(source, Length)
  let (source, _) = request_reader_fixture()
  write_u32(source, 12, 255)
  assert_prefix_rejected(source, Length)
  let (source, _) = request_reader_fixture()
  write_u32(source, 12, 0x80000100)
  assert_prefix_rejected(source, Length)
  let (source, _) = request_reader_fixture()
  write_u32(source, 12, 4097)
  assert_prefix_rejected(source, Length)
}

///|
test "invalid caller ranges preserve source and poison before copying" {
  for pair in [(-1, 1), (0, -1), (1, 4)] {
    let (source_offset, length) = pair
    let source = FixedArray::make(4, b'k')
    let reader = IncrementalRequestReader::new(fixture_limits())
    try reader.append(source, source_offset~, length~) catch {
      InvalidFrame(frame=RequestFrame, rule=Length) => ()
      _ => fail("unexpected caller-range error")
    } noraise {
      _ => fail("invalid caller range was accepted")
    }
    for byte in source {
      assert_eq(byte, b'k')
    }
    assert_eq(reader.length, 0)
    assert_eq(reader.frame_buffer.scratch[0], b'\x00')
    assert_eq(reader.frame_buffer.scratch[15], b'\x00')
    assert_reader_poisoned(reader)
  }
}

///|
test "accepted bytes are snapshotted from later caller mutation" {
  let (source, length) = request_reader_fixture()
  let reader = IncrementalRequestReader::new(fixture_limits())
  reader.append(source, source_offset=0, length=16)
  for index in 0..<16 {
    source[index] = b'X'
  }
  let (canonical, _) = request_reader_fixture()
  reader.append(canonical, source_offset=16, length=length - 16)
  for index in 16..<canonical.length() {
    canonical[index] = b'X'
  }
  assert_eq(reader.take().request().request_id().value(), 42UL)
}

///|
test "final frame error poisons and trailing or pipelined bytes reject" {
  let (source, length) = request_reader_fixture()
  source[16] = source[16] ^ b'\x01'
  let reader = IncrementalRequestReader::new(fixture_limits())
  reader.append(source, source_offset=0, length=16)
  try reader.append(source, source_offset=16, length=length - 16) catch {
    InvalidFrame(frame=RequestFrame, ..) => ()
    _ => fail("unexpected final-frame error")
  } noraise {
    _ => fail("corrupt final frame was accepted")
  }
  assert_reader_poisoned(reader)
  let (source, length) = request_reader_fixture()
  let trailing = IncrementalRequestReader::new(fixture_limits())
  feed_request_range(trailing, source, 0, length)
  try trailing.append([b'x'], source_offset=0, length=1) catch {
    InvalidFrame(frame=RequestFrame, rule=Length) => ()
    _ => fail("unexpected trailing-byte error")
  } noraise {
    _ => fail("trailing byte was accepted")
  }
  assert_reader_poisoned(trailing)
  let pipelined = IncrementalRequestReader::new(fixture_limits())
  pipelined.append(source, source_offset=0, length=16)
  let capacity = pipelined.next_capacity()
  try
    pipelined.append(
      FixedArray::make(capacity + 1, b'x'),
      source_offset=0,
      length=capacity + 1,
    )
  catch {
    InvalidFrame(frame=RequestFrame, rule=Length) => ()
    _ => fail("unexpected pipelined-frame error")
  } noraise {
    _ => fail("pipelined byte was accepted")
  }
  assert_reader_poisoned(pipelined)
}

///|
test "validated request can be taken exactly once" {
  let (source, length) = request_reader_fixture()
  let reader = IncrementalRequestReader::new(fixture_limits())
  feed_request_range(reader, source, 0, length)
  ignore(reader.take())
  try reader.take() catch {
    StaleFrame => ()
    _ => fail("unexpected request replay error")
  } noraise {
    _ => fail("request frame was taken twice")
  }
}
