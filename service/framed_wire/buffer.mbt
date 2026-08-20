///|
pub fn RequestFrameBuffer::new(limits : FramedWireLimits) -> RequestFrameBuffer {
  let scratch = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  let request_workspace = new_luna_framed_request_workspace(
    limits,
    try! LunaFramedRequestStepBudget::new(
      LUNA_FRAMED_REQUEST_MAX_STEP_WORK_UNITS,
    ),
    scratch,
  )
  {
    limits,
    storage: FixedArray::make(limits.max_frame_bytes(), b'\x00'),
    scratch,
    request_workspace,
    request: None,
    epoch: 0UL,
    length: 0,
  }
}

///|
/// Returns the exact inference envelope that authenticated this buffer. The
/// scalar contract value carries no frame bytes or mutable buffer authority.
pub fn RequestFrameBuffer::inference_limits(
  self : RequestFrameBuffer,
) -> @inference.InferenceLimits {
  self.limits.inference_limits()
}

///|
pub fn RequestFrameBuffer::capacity(self : RequestFrameBuffer) -> Int {
  self.storage.length()
}

///|
fn RequestFrameBuffer::publish(
  self : RequestFrameBuffer,
  request : @inference.GenerateRequest,
  length : Int,
) -> ValidatedRequestFrame raise FramedWireError {
  if self.epoch == 0xffffffffffffffffUL {
    raise StaleFrame
  }
  copy_bytes(self.scratch, self.storage, length)
  self.request = Some(request)
  self.length = length
  self.epoch += 1UL
  { owner_ref: self, epoch: self.epoch }
}

///|
fn ValidatedRequestFrame::owner(
  self : ValidatedRequestFrame,
) -> RequestFrameBuffer raise FramedWireError {
  if self.epoch == 0UL ||
    self.owner_ref.epoch != self.epoch ||
    self.owner_ref.length == 0 {
    raise StaleFrame
  }
  self.owner_ref
}

///|
pub fn ValidatedRequestFrame::length(
  self : ValidatedRequestFrame,
) -> Int raise FramedWireError {
  self.owner().length
}

///|
pub fn ValidatedRequestFrame::request(
  self : ValidatedRequestFrame,
) -> @inference.GenerateRequest raise FramedWireError {
  self.owner().request.unwrap()
}

///|
pub fn ValidatedRequestFrame::copy_to(
  self : ValidatedRequestFrame,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise FramedWireError {
  let owner = self.owner()
  if destination_offset < 0 ||
    destination_offset > destination.length() - owner.length {
    raise DestinationTooSmall
  }
  for index in 0..<owner.length {
    destination[destination_offset + index] = owner.storage[index]
  }
}

///|
pub fn EventFrameBuffer::new(limits : FramedWireLimits) -> EventFrameBuffer {
  {
    limits,
    storage: FixedArray::make(limits.max_frame_bytes(), b'\x00'),
    scratch: FixedArray::make(limits.max_frame_bytes(), b'\x00'),
    event: None,
    epoch: 0UL,
    length: 0,
  }
}

///|
pub fn EventFrameBuffer::capacity(self : EventFrameBuffer) -> Int {
  self.storage.length()
}

///|
fn EventFrameBuffer::publish(
  self : EventFrameBuffer,
  event : @inference.StreamEvent,
  length : Int,
) -> ValidatedEventFrame raise FramedWireError {
  if self.epoch == 0xffffffffffffffffUL {
    raise StaleFrame
  }
  copy_bytes(self.scratch, self.storage, length)
  self.event = Some(event)
  self.length = length
  self.epoch += 1UL
  { owner_ref: self, epoch: self.epoch }
}

///|
fn ValidatedEventFrame::owner(
  self : ValidatedEventFrame,
) -> EventFrameBuffer raise FramedWireError {
  if self.epoch == 0UL ||
    self.owner_ref.epoch != self.epoch ||
    self.owner_ref.length == 0 {
    raise StaleFrame
  }
  self.owner_ref
}

///|
pub fn ValidatedEventFrame::length(
  self : ValidatedEventFrame,
) -> Int raise FramedWireError {
  self.owner().length
}

///|
pub fn ValidatedEventFrame::event(
  self : ValidatedEventFrame,
) -> @inference.StreamEvent raise FramedWireError {
  self.owner().event.unwrap()
}

///|
pub fn ValidatedEventFrame::copy_to(
  self : ValidatedEventFrame,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise FramedWireError {
  let owner = self.owner()
  if destination_offset < 0 ||
    destination_offset > destination.length() - owner.length {
    raise DestinationTooSmall
  }
  for index in 0..<owner.length {
    destination[destination_offset + index] = owner.storage[index]
  }
}
