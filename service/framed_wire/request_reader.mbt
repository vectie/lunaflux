///|
const REQUEST_READER_RECEIVING : Int = 0

///|
const REQUEST_READER_COMPLETE : Int = 1

///|
const REQUEST_READER_POISONED : Int = 2

///|
const REQUEST_READER_TAKEN : Int = 3

///|
fn request_reader_error(rule : FramedWireRule) -> FramedWireError {
  InvalidFrame(frame=RequestFrame, rule~)
}

///|
fn IncrementalRequestReader::poison(
  self : IncrementalRequestReader,
  error : FramedWireError,
) -> Unit raise FramedWireError {
  self.frame = None
  self.state = REQUEST_READER_POISONED
  raise error
}

///|
fn IncrementalRequestReader::virtual_byte(
  self : IncrementalRequestReader,
  source : FixedArray[Byte],
  source_offset : Int,
  index : Int,
) -> Byte {
  if index < self.length {
    self.frame_buffer.scratch[index]
  } else {
    source[source_offset + index - self.length]
  }
}

///|
fn IncrementalRequestReader::virtual_u32(
  self : IncrementalRequestReader,
  source : FixedArray[Byte],
  source_offset : Int,
  offset : Int,
) -> UInt {
  self.virtual_byte(source, source_offset, offset).to_uint() |
  (self.virtual_byte(source, source_offset, offset + 1).to_uint() << 8) |
  (self.virtual_byte(source, source_offset, offset + 2).to_uint() << 16) |
  (self.virtual_byte(source, source_offset, offset + 3).to_uint() << 24)
}

///|
fn IncrementalRequestReader::preflight_prefix(
  self : IncrementalRequestReader,
  source : FixedArray[Byte],
  source_offset : Int,
  next_length : Int,
) -> Int raise FramedWireError {
  if next_length >= 4 &&
    self.virtual_u32(source, source_offset, 0) != request_magic {
    self.poison(request_reader_error(Magic))
  }
  if next_length >= 8 &&
    self.virtual_u32(source, source_offset, 4) != request_version {
    self.poison(request_reader_error(Version))
  }
  if next_length >= 12 && self.virtual_u32(source, source_offset, 8) != 1 {
    self.poison(request_reader_error(Kind))
  }
  if next_length < 16 {
    return 0
  }
  let declared = self.virtual_u32(source, source_offset, 12)
  if (declared & 0x80000000) != 0 {
    self.poison(request_reader_error(Length))
  }
  let declared_length = declared.reinterpret_as_int()
  if declared_length < REQUEST_HEADER_BYTES ||
    declared_length > self.frame_buffer.capacity() ||
    next_length > declared_length {
    self.poison(request_reader_error(Length))
  }
  declared_length
}

///|
pub fn IncrementalRequestReader::new(
  limits : FramedWireLimits,
) -> IncrementalRequestReader {
  {
    frame_buffer: RequestFrameBuffer::new(limits),
    frame: None,
    length: 0,
    declared_length: 0,
    state: REQUEST_READER_RECEIVING,
  }
}

///|
/// Returns the exact remaining header or declared-frame capacity. Zero means
/// the canonical frame is complete and ready to take.
pub fn IncrementalRequestReader::next_capacity(
  self : IncrementalRequestReader,
) -> Int raise FramedWireError {
  if self.state == REQUEST_READER_POISONED {
    raise RequestReaderPoisoned
  }
  if self.state == REQUEST_READER_TAKEN {
    raise StaleFrame
  }
  if self.state == REQUEST_READER_COMPLETE {
    return 0
  }
  if self.declared_length == 0 {
    16 - self.length
  } else {
    self.declared_length - self.length
  }
}

///|
/// Transactionally appends one caller-owned byte range. The request prefix is
/// authenticated before bytes beyond it are accepted. Any prefix, trailing,
/// or final canonical-frame failure permanently poisons this owner.
pub fn IncrementalRequestReader::append(
  self : IncrementalRequestReader,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> Unit raise FramedWireError {
  if self.state == REQUEST_READER_POISONED {
    raise RequestReaderPoisoned
  }
  if self.state == REQUEST_READER_TAKEN {
    raise StaleFrame
  }
  if length == 0 {
    return
  }
  if self.state == REQUEST_READER_COMPLETE {
    self.poison(request_reader_error(Length))
  }
  if source_offset < 0 ||
    length < 0 ||
    source_offset > source.length() - length ||
    length > self.next_capacity() {
    self.poison(request_reader_error(Length))
  }
  let next_length = self.length + length
  let declared = self.preflight_prefix(source, source_offset, next_length)
  for index in 0..<length {
    self.frame_buffer.scratch[self.length + index] = source[source_offset +
      index]
  }
  self.length = next_length
  if declared != 0 {
    self.declared_length = declared
  }
  if self.declared_length != 0 && self.length == self.declared_length {
    let frame = self.frame_buffer.load(self.frame_buffer.scratch, self.length) catch {
      error => return self.poison(error)
    }
    self.frame = Some(frame)
    self.state = REQUEST_READER_COMPLETE
  }
}

///|
/// Takes the exact validated request frame once. Incomplete input remains
/// retryable; poisoned or already-taken owners reject deterministically.
pub fn IncrementalRequestReader::take(
  self : IncrementalRequestReader,
) -> ValidatedRequestFrame raise FramedWireError {
  if self.state == REQUEST_READER_POISONED {
    raise RequestReaderPoisoned
  }
  if self.state == REQUEST_READER_TAKEN {
    raise StaleFrame
  }
  if self.state != REQUEST_READER_COMPLETE {
    raise RequestReaderIncomplete
  }
  let frame = self.frame.unwrap()
  self.frame = None
  self.state = REQUEST_READER_TAKEN
  frame
}
