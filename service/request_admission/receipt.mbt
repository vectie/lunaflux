///|
fn capture_receipt_with_clock(
  read_clock : () -> UInt64 raise RequestAdmissionError,
) -> RequestReceipt raise RequestAdmissionError {
  { at_millis: read_clock() }
}

///|
fn bind_received(
  receipt : RequestReceipt,
  frame : @framed_wire.ValidatedRequestFrame,
  inference : @inference.InferenceLimits,
) -> ReceivedRequest raise RequestAdmissionError {
  let request = frame.request() catch { _ => raise Invalid(FrameContract) }
  let deadline = @inference.AdmissionDeadline::from_receipt(
    receipt.at_millis,
    request.deadline(),
  ) catch {
    _ => raise Invalid(Deadline)
  }
  { receipt, request, deadline, inference, consumed: false }
}

///|
pub fn IncrementalRequestReceiver::new(
  limits : @framed_wire.FramedWireLimits,
  clock : @monotonic_clock.MonotonicClock,
) -> IncrementalRequestReceiver {
  {
    reader: @framed_wire.IncrementalRequestReader::new(limits),
    clock,
    inference: limits.inference_limits(),
    receipt_at_millis: 0UL,
    has_receipt: false,
    terminal: false,
  }
}

///|
fn IncrementalRequestReceiver::preflight_append(
  self : IncrementalRequestReceiver,
  source : FixedArray[Byte],
  source_offset : Int,
  length : Int,
) -> Bool raise RequestAdmissionError {
  if self.terminal {
    raise Invalid(FrameContract)
  }
  if source_offset < 0 || length < 0 || source_offset > source.length() - length {
    raise Invalid(FrameContract)
  }
  if length == 0 {
    return false
  }
  let capacity = self.reader.next_capacity() catch {
    _ => {
      self.terminal = true
      raise Invalid(FrameContract)
    }
  }
  if length > capacity {
    self.terminal = true
    raise Invalid(FrameContract)
  }
  true
}

///|
fn IncrementalRequestReceiver::append_validated(
  self : IncrementalRequestReceiver,
  source : FixedArray[Byte],
  source_offset : Int,
  length : Int,
) -> Unit raise RequestAdmissionError {
  self.reader.append(source, source_offset~, length~) catch {
    _ => {
      self.terminal = true
      raise Invalid(FrameContract)
    }
  }
}

///|
/// Appends one byte range. The first nonempty append validates source/capacity,
/// samples the retained monotonic clock exactly once, and only then permits the
/// framed reader to copy bytes.
pub fn IncrementalRequestReceiver::append(
  self : IncrementalRequestReceiver,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> Unit raise RequestAdmissionError {
  if !self.preflight_append(source, source_offset, length) {
    return
  }
  if !self.has_receipt {
    let at_millis = self.clock.now_millis() catch {
      _ => raise ClockUnavailable
    }
    self.receipt_at_millis = at_millis
    self.has_receipt = true
  }
  self.append_validated(source, source_offset, length)
}

///|
pub fn IncrementalRequestReceiver::next_capacity(
  self : IncrementalRequestReceiver,
) -> Int raise RequestAdmissionError {
  if self.terminal {
    raise Invalid(FrameContract)
  }
  self.reader.next_capacity() catch {
    _ => {
      self.terminal = true
      raise Invalid(FrameContract)
    }
  }
}

///|
/// Takes one fully validated and receipt-bound request. Incomplete input can
/// still be appended; poison or a successful take terminally closes the owner.
pub fn IncrementalRequestReceiver::take_received(
  self : IncrementalRequestReceiver,
) -> ReceivedRequest raise RequestAdmissionError {
  if self.terminal {
    raise Invalid(FrameContract)
  }
  let frame = self.reader.take() catch {
    RequestReaderIncomplete => raise Invalid(FrameContract)
    _ => {
      self.terminal = true
      raise Invalid(FrameContract)
    }
  }
  if !self.has_receipt {
    abort("completed request frame has no trusted receipt")
  }
  self.terminal = true
  bind_received({ at_millis: self.receipt_at_millis }, frame, self.inference)
}

///|
/// Captures one monotonic receipt before the authoritative whole-frame decoder
/// can copy bytes, then binds the immutable request and absolute deadline.
pub fn receive(
  buffer : @framed_wire.RequestFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
  clock : @monotonic_clock.MonotonicClock,
) -> ReceivedRequest raise RequestAdmissionError {
  let receipt = capture_receipt_with_clock(() => {
    clock.now_millis() catch {
      _ => raise ClockUnavailable
    }
  })
  let frame = buffer.load(source, length) catch {
    _ => raise Invalid(FrameContract)
  }
  bind_received(receipt, frame, buffer.inference_limits())
}
