///|
pub(all) suberror RemoteMailboxError {
  Closed
  Busy
  Range
  State
}

///|
/// One bounded frame in each direction. Shared only by cooperative tasks on
/// one event loop. No locks, native handles, callbacks or per-operation arrays.
pub struct RemoteMailbox {
  priv input : FixedArray[Byte]
  priv output : FixedArray[Byte]
  priv mut input_length : Int
  priv mut output_length : Int
  priv mut output_held : Bool
  priv mut read_active : Bool
  priv mut write_active : Bool
  priv mut failed : Bool
}

///|
pub fn RemoteMailbox::new(
  capacity : Int,
) -> RemoteMailbox raise RemoteMailboxError {
  if capacity < 1 || capacity > 67108864 {
    raise Range
  }
  {
    input: FixedArray::make(capacity, b'\x00'),
    output: FixedArray::make(capacity, b'\x00'),
    input_length: 0,
    output_length: 0,
    output_held: false,
    read_active: false,
    write_active: false,
    failed: false,
  }
}

///|
pub fn RemoteMailbox::capacity(self : RemoteMailbox) -> Int {
  self.input.length()
}

///|
pub fn RemoteMailbox::is_failed(self : RemoteMailbox) -> Bool {
  self.failed
}

///|
pub fn RemoteMailbox::fail(self : RemoteMailbox) -> Unit {
  self.failed = true
  self.input_length = 0
  self.output_length = 0
}

///|
fn RemoteMailbox::require_open(
  self : RemoteMailbox,
) -> Unit raise RemoteMailboxError {
  if self.failed {
    raise Closed
  }
}

///|
pub fn RemoteMailbox::can_deliver(self : RemoteMailbox) -> Bool {
  !self.failed && self.input_length == 0
}

///|
/// Network pump publishes only a complete authenticated frame.
pub fn RemoteMailbox::deliver(
  self : RemoteMailbox,
  bytes : FixedArray[Byte],
  length : Int,
) -> Unit raise RemoteMailboxError {
  self.require_open()
  if self.input_length != 0 {
    raise Busy
  }
  if length < 1 || length > bytes.length() || length > self.input.length() {
    raise Range
  }
  for i in 0..<length {
    self.input[i] = bytes[i]
  }
  self.input_length = length
}

///|
pub fn RemoteMailbox::begin_read(
  self : RemoteMailbox,
) -> Unit raise RemoteMailboxError {
  self.require_open()
  if self.read_active {
    raise Busy
  }
  self.read_active = true
}

///|
pub fn RemoteMailbox::poll_read(
  self : RemoteMailbox,
  destination : FixedArray[Byte],
) -> Int? raise RemoteMailboxError {
  self.require_open()
  if !self.read_active {
    raise State
  }
  let length = self.input_length
  if length == 0 {
    return None
  }
  if length > destination.length() {
    raise Range
  }
  for i in 0..<length {
    destination[i] = self.input[i]
  }
  self.input_length = 0
  self.read_active = false
  Some(length)
}

///|
pub fn RemoteMailbox::begin_write(
  self : RemoteMailbox,
  bytes : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Unit raise RemoteMailboxError {
  self.require_open()
  if self.write_active {
    raise Busy
  }
  if length < 1 ||
    offset < 0 ||
    offset > bytes.length() ||
    length > bytes.length() - offset ||
    length > self.output.length() {
    raise Range
  }
  for i in 0..<length {
    self.output[i] = bytes[offset + i]
  }
  self.output_length = length
  self.output_held = false
  self.write_active = true
}

///|
/// Pump borrows a snapshot until sent(); repeated claims are rejected.
pub fn RemoteMailbox::outgoing(
  self : RemoteMailbox,
  destination : FixedArray[Byte],
) -> Int? raise RemoteMailboxError {
  self.require_open()
  if self.output_held {
    raise Busy
  }
  let length = self.output_length
  if length == 0 {
    return None
  }
  if length > destination.length() {
    raise Range
  }
  for i in 0..<length {
    destination[i] = self.output[i]
  }
  self.output_held = true
  Some(length)
}

///|
pub fn RemoteMailbox::sent(
  self : RemoteMailbox,
) -> Unit raise RemoteMailboxError {
  self.require_open()
  if !self.output_held || self.output_length == 0 {
    raise State
  }
  self.output_length = 0
  self.output_held = false
}

///|
pub fn RemoteMailbox::poll_write(
  self : RemoteMailbox,
) -> Bool raise RemoteMailboxError {
  self.require_open()
  if !self.write_active {
    raise State
  }
  if self.output_length != 0 {
    return false
  }
  self.write_active = false
  true
}
