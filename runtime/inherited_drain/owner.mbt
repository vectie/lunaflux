///|
fn inherited_drain_error(rule : InheritedDrainRule) -> InheritedDrainError {
  InheritedDrainFailed(rule)
}

///|
pub fn prepare_inherited_drain_v1() -> InheritedDrainPreparation raise InheritedDrainError {
  // Allocate every managed backing before moving the inherited descriptor.
  let input = FixedArray::make(INHERITED_DRAIN_FRAME_BYTES, b'\x00')
  let output = FixedArray::make(INHERITED_DRAIN_FRAME_BYTES, b'\x00')
  let native = @native.open_fixed()
  match @native.open_status(native) {
    0 => { native, input, output, consumed: false }
    1 => raise inherited_drain_error(DescriptorUnavailable)
    _ => raise inherited_drain_error(InvalidDescriptor)
  }
}

///|
pub fn InheritedDrainPreparation::activate(
  self : InheritedDrainPreparation,
) -> InheritedDrainChannel raise InheritedDrainError {
  if self.consumed || @native.is_closed(self.native) {
    raise inherited_drain_error(InvalidState)
  }
  self.consumed = true
  {
    native: self.native,
    input: self.input,
    output: self.output,
    input_cursor: 0,
    output_cursor: 0,
    phase: 0,
    request_latched: false,
    response_started: false,
  }
}

///|
pub fn InheritedDrainPreparation::close(
  self : InheritedDrainPreparation,
) -> Unit raise InheritedDrainError {
  if self.consumed {
    raise inherited_drain_error(InvalidState)
  }
  self.consumed = true
  if @native.close(self.native) != 0 {
    raise inherited_drain_error(ChannelLost)
  }
}

///|
fn InheritedDrainChannel::lose(
  self : InheritedDrainChannel,
) -> InheritedDrainProgress {
  ignore(@native.close(self.native))
  self.phase = 3
  InheritedDrainChannelLost
}

///|
fn InheritedDrainChannel::progress_read(
  self : InheritedDrainChannel,
) -> InheritedDrainProgress {
  let remaining = INHERITED_DRAIN_FRAME_BYTES - self.input_cursor
  let count = @native.try_read(
    self.native,
    self.input,
    self.input_cursor,
    remaining,
  )
  if count > 0 && count <= remaining {
    self.input_cursor += count
  } else if count == 0 {
    return InheritedDrainPending
  } else {
    return self.lose()
  }
  if self.input_cursor != INHERITED_DRAIN_FRAME_BYTES {
    return InheritedDrainPending
  }
  // A queued ninth byte is trailing/replay input, never another command.
  if @native.peek(self.native) != 0 || !drain_command_matches(self.input) {
    return self.lose()
  }
  self.phase = 1
  self.request_latched = true
  InheritedDrainRequested
}

///|
fn InheritedDrainChannel::progress_write(
  self : InheritedDrainChannel,
) -> InheritedDrainProgress {
  let remaining = INHERITED_DRAIN_FRAME_BYTES - self.output_cursor
  let count = @native.try_write(
    self.native,
    self.output,
    self.output_cursor,
    remaining,
  )
  if count > 0 && count <= remaining {
    self.output_cursor += count
  } else if count == 0 {
    return InheritedDrainPending
  } else {
    return self.lose()
  }
  if self.output_cursor == INHERITED_DRAIN_FRAME_BYTES {
    self.phase = 2
    InheritedDrainResponseConfirmed
  } else {
    InheritedDrainPending
  }
}

///|
pub fn InheritedDrainChannel::begin_response(
  self : InheritedDrainChannel,
  response : InheritedDrainResponse,
) -> Unit raise InheritedDrainError {
  if self.phase != 1 || self.response_started || @native.is_closed(self.native) {
    raise inherited_drain_error(InvalidState)
  }
  fill_drain_response(self.output, response)
  self.response_started = true
}

///|
pub fn InheritedDrainChannel::progress(
  self : InheritedDrainChannel,
) -> InheritedDrainProgress {
  match self.phase {
    0 => self.progress_read()
    1 =>
      if self.response_started {
        self.progress_write()
      } else {
        InheritedDrainRequested
      }
    2 =>
      if @native.peek(self.native) == 0 {
        InheritedDrainResponseConfirmed
      } else {
        self.lose()
      }
    3 => InheritedDrainChannelLost
    _ => InheritedDrainClosed
  }
}

///|
pub fn InheritedDrainChannel::close(
  self : InheritedDrainChannel,
) -> Unit raise InheritedDrainError {
  if self.phase == 4 {
    return
  }
  let status = @native.close(self.native)
  self.phase = 4
  if status != 0 {
    raise inherited_drain_error(ChannelLost)
  }
}

///|
pub fn InheritedDrainChannel::is_live(self : InheritedDrainChannel) -> Bool {
  self.phase < 3 && !@native.is_closed(self.native)
}

///|
pub fn InheritedDrainChannel::request_received(
  self : InheritedDrainChannel,
) -> Bool {
  self.request_latched
}

///|
pub fn InheritedDrainChannel::response_confirmed(
  self : InheritedDrainChannel,
) -> Bool {
  self.phase == 2
}

///|
pub fn InheritedDrainChannel::is_closed(self : InheritedDrainChannel) -> Bool {
  self.phase == 4 && @native.is_closed(self.native)
}
