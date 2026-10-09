///|
pub(all) suberror RemoteChannelError {
  InvalidConfiguration
  InvalidState
  InvalidFrame
  TimedOut
  ClockRegressed
  Disconnected
}

///|
/// Owns a connected socket and fixed independent receive/transmit lanes.
/// A caller transfers exclusive socket ownership at construction and explicitly
/// closes it. This transport provides framing, not peer authentication.
pub struct RemoteChannel {
  priv connection : @socket.Tcp
  priv input : FixedArray[Byte]
  priv output : Bytes
  priv mutable_output : FixedArray[Byte]
  priv timeout_millis : UInt64
  priv mut open : Bool
  priv mut receiving : Bool
  priv mut sending : Bool
  priv mut reading : Bool
  priv mut writing : Bool
  priv mut receive_cursor : Int
  priv mut receive_length : Int
  priv mut send_cursor : Int
  priv mut send_length : Int
  priv mut receive_deadline : UInt64
  priv mut send_deadline : UInt64
  priv mut last_tick : UInt64
}

///|
pub fn RemoteChannel::new(
  connection : @socket.Tcp,
  max_frame_bytes~ : Int,
  timeout_millis~ : UInt64,
) -> RemoteChannel raise RemoteChannelError {
  if max_frame_bytes < 1 ||
    max_frame_bytes > 67108864 ||
    timeout_millis == 0UL ||
    timeout_millis > 300000UL {
    // A rejected configuration does not consume the supplied socket.
    raise InvalidConfiguration
  }
  let output = Bytes::make(max_frame_bytes + 4, b'\x00')
  {
    connection,
    input: FixedArray::make(max_frame_bytes + 4, b'\x00'),
    output,
    mutable_output: @buffer_alias.retain_bytes_as_fixed_array(output),
    timeout_millis,
    open: true,
    receiving: false,
    sending: false,
    reading: false,
    writing: false,
    receive_cursor: 0,
    receive_length: 4,
    send_cursor: 0,
    send_length: 0,
    receive_deadline: 0UL,
    send_deadline: 0UL,
    last_tick: 0UL,
  }
}

///|
pub fn RemoteChannel::is_open(self : RemoteChannel) -> Bool {
  self.open
}

///|
pub fn RemoteChannel::close(self : RemoteChannel) -> Unit {
  if !self.open {
    return
  }
  self.open = false
  self.receiving = false
  self.sending = false
  self.connection.close()
}

///|
fn RemoteChannel::fail(
  self : RemoteChannel,
  error : RemoteChannelError,
) -> Unit raise RemoteChannelError {
  self.close()
  raise error
}

///|
fn RemoteChannel::observe(
  self : RemoteChannel,
  now : UInt64,
) -> Unit raise RemoteChannelError {
  if !self.open {
    raise InvalidState
  }
  if now < self.last_tick {
    self.fail(ClockRegressed)
  }
  self.last_tick = now
  if (self.receiving && now >= self.receive_deadline) ||
    (self.sending && now >= self.send_deadline) {
    self.fail(TimedOut)
  }
}

///|
fn RemoteChannel::deadline(
  self : RemoteChannel,
  now : UInt64,
) -> UInt64 raise RemoteChannelError {
  self.observe(now)
  if now > 0xffffffffffffffffUL - self.timeout_millis {
    self.fail(TimedOut)
  }
  now + self.timeout_millis
}

///|
/// Snapshots one frame before any async write, including its bounded length.
pub fn RemoteChannel::begin_send(
  self : RemoteChannel,
  source : FixedArray[Byte],
  length : Int,
  local_now_millis : UInt64,
) -> Unit raise RemoteChannelError {
  if !self.open || self.sending {
    raise InvalidState
  }
  if length <= 0 ||
    length > source.length() ||
    length > self.output.length() - 4 {
    raise InvalidFrame
  }
  let deadline = self.deadline(local_now_millis)
  for index in 0..<4 {
    self.mutable_output[index] = (length.reinterpret_as_uint() >> (index * 8)).to_byte()
  }
  for index in 0..<length {
    self.mutable_output[index + 4] = source[index]
  }
  self.send_cursor = 0
  self.send_length = length + 4
  self.send_deadline = deadline
  self.sending = true
}

///|
pub fn RemoteChannel::begin_receive(
  self : RemoteChannel,
  local_now_millis : UInt64,
) -> Unit raise RemoteChannelError {
  if !self.open || self.receiving {
    raise InvalidState
  }
  self.receive_deadline = self.deadline(local_now_millis)
  self.receive_cursor = 0
  self.receive_length = 4
  self.receiving = true
}

///|
/// Performs at most one zero-wait socket write. Partial progress never extends
/// the absolute operation deadline. The control loop supplies its local clock.
pub async fn RemoteChannel::poll_send(
  self : RemoteChannel,
  local_now_millis : UInt64,
) -> Bool {
  self.observe(local_now_millis)
  if !self.sending || self.writing {
    raise InvalidState
  }
  self.writing = true
  defer {
    self.writing = false
  }
  let outcome = @async.with_timeout_opt(0, () => {
    self.connection.write_once(
      self.output,
      offset=self.send_cursor,
      len=self.send_length - self.send_cursor,
    )
  }) catch {
    error => {
      self.close()
      if @async.is_being_cancelled() {
        raise error
      }
      raise Disconnected
    }
  }
  guard outcome is Some(count) else { return false }
  if count <= 0 || count > self.send_length - self.send_cursor {
    self.fail(Disconnected)
  }
  self.send_cursor += count
  if self.send_cursor != self.send_length {
    return false
  }
  self.sending = false
  true
}

///|
/// Performs at most one zero-wait read into private fixed storage. Publishes a
/// frame to `destination` only after every byte arrives; no partial data escapes.
pub async fn RemoteChannel::poll_receive(
  self : RemoteChannel,
  destination : FixedArray[Byte],
  local_now_millis : UInt64,
) -> Int? {
  self.observe(local_now_millis)
  if !self.receiving || self.reading {
    raise InvalidState
  }
  self.reading = true
  defer {
    self.reading = false
  }
  let outcome = @async.with_timeout_opt(0, () => {
    self.connection.read(
      self.input,
      offset=self.receive_cursor,
      max_len=self.receive_length - self.receive_cursor,
    )
  }) catch {
    error => {
      self.close()
      if @async.is_being_cancelled() {
        raise error
      }
      raise Disconnected
    }
  }
  guard outcome is Some(count) else { return None }
  if count <= 0 || count > self.receive_length - self.receive_cursor {
    self.fail(Disconnected)
  }
  self.receive_cursor += count
  if self.receive_cursor != self.receive_length {
    return None
  }
  if self.receive_length == 4 {
    let mut length = 0U
    for index in 0..<4 {
      length = length | (self.input[index].to_uint() << (index * 8))
    }
    if length == 0U ||
      length > (self.input.length() - 4).reinterpret_as_uint() ||
      length > destination.length().reinterpret_as_uint() {
      self.fail(InvalidFrame)
    }
    self.receive_length = length.reinterpret_as_int() + 4
    return None
  }
  let length = self.receive_length - 4
  // A caller can provide another destination between polls; check again before
  // publishing any bytes, even if the original destination was large enough.
  if length > destination.length() {
    self.fail(InvalidFrame)
  }
  for index in 0..<length {
    destination[index] = self.input[index + 4]
  }
  self.receiving = false
  Some(length)
}
