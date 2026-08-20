///|
fn receiver_limits() -> @framed_wire.FramedWireLimits raise {
  @framed_wire.FramedWireLimits::new(fixture_inference(), max_frame_bytes=4096)
}

///|
fn IncrementalRequestReceiver::append_with_clock(
  self : IncrementalRequestReceiver,
  source : FixedArray[Byte],
  source_offset : Int,
  length : Int,
  read_clock : () -> UInt64 raise RequestAdmissionError,
) -> Unit raise RequestAdmissionError {
  if !self.preflight_append(source, source_offset, length) {
    return
  }
  if !self.has_receipt {
    let receipt = capture_receipt_with_clock(read_clock)
    self.receipt_at_millis = receipt.at_millis
    self.has_receipt = true
  }
  self.append_validated(source, source_offset, length)
}

///|
fn feed_received_range(
  receiver : IncrementalRequestReceiver,
  source : FixedArray[Byte],
  start : Int,
  end : Int,
  reads : Ref[Int],
  at_millis : UInt64,
) -> Unit raise RequestAdmissionError {
  let mut cursor = start
  while cursor < end {
    let capacity = receiver.next_capacity()
    let remaining = end - cursor
    let amount = if remaining < capacity { remaining } else { capacity }
    receiver.append_with_clock(source, cursor, amount, () => {
      reads.val += 1
      at_millis
    })
    cursor += amount
  }
}

///|
test "first nonempty byte captures exactly once before reader mutation" {
  let identity = fixture_identity("0")
  let (source, length, _) = fixture_received_source(fixture_request(identity))
  let receiver = IncrementalRequestReceiver::new(
    receiver_limits(),
    @monotonic_clock.MonotonicClock::system(),
  )
  let reads = Ref(0)
  let capacity_before = Ref(-1)
  let receipt_before = Ref(true)
  receiver.append_with_clock(source, 0, 0, () => {
    reads.val += 1
    100UL
  })
  assert_eq(reads.val, 0)
  receiver.append_with_clock(source, 0, 1, () => {
    capacity_before.val = receiver.reader.next_capacity() catch { _ => -1 }
    receipt_before.val = receiver.has_receipt
    reads.val += 1
    100UL
  })
  assert_eq(capacity_before.val, 16)
  assert_false(receipt_before.val)
  assert_eq(reads.val, 1)
  assert_eq(receiver.next_capacity(), 15)
  feed_received_range(receiver, source, 1, length, reads, 999UL)
  assert_eq(reads.val, 1)
  let received = receiver.take_received()
  assert_eq(received.receipt.at_millis, 100UL)
  assert_eq(received.deadline.at_millis(), 200UL)
  assert_eq(received.request.request_id().value(), 7UL)
}

///|
test "public incremental append uses the retained clock" {
  let identity = fixture_identity("0")
  let (source, _, _) = fixture_received_source(fixture_request(identity))
  let receiver = IncrementalRequestReceiver::new(
    receiver_limits(),
    @monotonic_clock.MonotonicClock::system(),
  )
  receiver.append(source, source_offset=0, length=1)
  assert_eq(receiver.next_capacity(), 15)
}

///|
test "clock failure copies nothing and first-byte receipt can retry" {
  let identity = fixture_identity("0")
  let (source, length, _) = fixture_received_source(fixture_request(identity))
  let receiver = IncrementalRequestReceiver::new(
    receiver_limits(),
    @monotonic_clock.MonotonicClock::system(),
  )
  let failed_reads = Ref(0)
  try
    receiver.append_with_clock(source, 0, 1, () => {
      failed_reads.val += 1
      raise ClockUnavailable
    })
  catch {
    ClockUnavailable => ()
    _ => fail("unexpected receipt clock error")
  } noraise {
    _ => fail("failed clock accepted a byte")
  }
  assert_eq(failed_reads.val, 1)
  assert_false(receiver.has_receipt)
  assert_eq(receiver.next_capacity(), 16)
  let reads = Ref(0)
  feed_received_range(receiver, source, 0, length, reads, 20UL)
  assert_eq(reads.val, 1)
  assert_eq(receiver.take_received().receipt.at_millis, 20UL)
}

///|
test "invalid source range reads no clock and malformed prefix poisons" {
  let receiver = IncrementalRequestReceiver::new(
    receiver_limits(),
    @monotonic_clock.MonotonicClock::system(),
  )
  let reads = Ref(0)
  try
    receiver.append_with_clock([b'x'], 0, 2, () => {
      reads.val += 1
      1UL
    })
  catch {
    Invalid(FrameContract) => ()
    _ => fail("unexpected source-range error")
  } noraise {
    _ => fail("invalid source range was accepted")
  }
  assert_eq(reads.val, 0)
  assert_eq(receiver.next_capacity(), 16)
  try
    receiver.append_with_clock(FixedArray::make(4, b'\x00'), 0, 4, () => {
      reads.val += 1
      1UL
    })
  catch {
    Invalid(FrameContract) => ()
    _ => fail("unexpected malformed-prefix error")
  } noraise {
    _ => fail("malformed prefix was accepted")
  }
  assert_eq(reads.val, 1)
  try receiver.next_capacity() catch {
    Invalid(FrameContract) => ()
    _ => fail("unexpected poisoned receiver error")
  } noraise {
    _ => fail("poisoned receiver remained usable")
  }
}

///|
test "incremental receiver takes one immutable received request" {
  let identity = fixture_identity("0")
  let (source, length, _) = fixture_received_source(fixture_request(identity))
  let receiver = IncrementalRequestReceiver::new(
    receiver_limits(),
    @monotonic_clock.MonotonicClock::system(),
  )
  let reads = Ref(0)
  feed_received_range(receiver, source, 0, length - 1, reads, 30UL)
  try receiver.take_received() catch {
    Invalid(FrameContract) => ()
    _ => fail("unexpected incomplete-receipt error")
  } noraise {
    _ => fail("incomplete received request was taken")
  }
  assert_eq(receiver.next_capacity(), 1)
  feed_received_range(receiver, source, length - 1, length, reads, 99UL)
  assert_eq(reads.val, 1)
  let received = receiver.take_received()
  assert_eq(received.request.request_id().value(), 7UL)
  try receiver.take_received() catch {
    Invalid(FrameContract) => ()
    _ => fail("unexpected received replay error")
  } noraise {
    _ => fail("incremental received request was taken twice")
  }
}

///|
test "binding rejects a stale validated frame before request publication" {
  let identity = fixture_identity("0")
  let request = fixture_request(identity)
  let limits = receiver_limits()
  let buffer = @framed_wire.RequestFrameBuffer::new(limits)
  let stale = buffer.encode(request)
  ignore(buffer.encode(request))
  try bind_received({ at_millis: 10UL }, stale, fixture_inference()) catch {
    Invalid(FrameContract) => ()
    _ => fail("unexpected stale-frame binding error")
  } noraise {
    _ => fail("stale frame was receipt-bound")
  }
}
