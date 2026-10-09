///|
pub(all) suberror RemoteTlsError {
  Configuration
  Authentication
  Closed
  Busy
  Frame
}

///|
/// An operator-provisioned, per-rank secret. No formatting/serialization API.
pub struct RemoteCredential {
  priv bytes : Bytes
}

///|
pub fn RemoteCredential::from_bytes(
  bytes : Bytes,
) -> RemoteCredential raise RemoteTlsError {
  if bytes.length() != 32 {
    raise Configuration
  }
  { bytes: Bytes::makei(32, i => bytes[i]), }
}

///|
/// Exclusive TLS and TCP ownership. Async operations run outside scheduling.
pub struct AuthenticatedRemoteChannel {
  priv socket : @socket.Tcp
  priv tls : @tls.Tls
  priv input : FixedArray[Byte]
  priv output : Bytes
  priv output_alias : FixedArray[Byte]
  priv timeout_millis : Int
  priv binding : Bytes
  priv mut open : Bool
  priv mut reading : Bool
  priv mut writing : Bool
}

///|
fn validate_limits(
  binding : Bytes,
  max_frame_bytes : Int,
  timeout_millis : Int,
) -> Unit raise RemoteTlsError {
  if binding.length() < 1 ||
    binding.length() > 65536 ||
    max_frame_bytes < 1 ||
    max_frame_bytes > 67108864 ||
    timeout_millis < 1 ||
    timeout_millis > 300000 {
    raise Configuration
  }
}

///|
async fn read_exact(
  tls : @tls.Tls,
  bytes : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Unit {
  let mut cursor = offset
  while cursor < offset + length {
    let count = tls.read(bytes, offset=cursor, max_len=offset + length - cursor)
    if count <= 0 {
      raise Closed
    }
    cursor += count
  }
}

///|
fn hello(credential : RemoteCredential, binding : Bytes) -> Bytes {
  Bytes::makei(40 + binding.length(), i => {
    if i < 8 {
      b"LFTLS001"[i]
    } else if i < 40 {
      credential.bytes[i - 8]
    } else {
      binding[i - 40]
    }
  })
}

///|
fn matches(actual : FixedArray[Byte], expected : Bytes) -> Bool {
  if actual.length() != expected.length() {
    return false
  }
  let mut difference = 0U
  for i in 0..<actual.length() {
    difference = difference | (actual[i].to_uint() ^ expected[i].to_uint())
  }
  difference == 0U
}

///|
fn channel(
  socket : @socket.Tcp,
  tls : @tls.Tls,
  binding : Bytes,
  max_frame_bytes : Int,
  timeout_millis : Int,
) -> AuthenticatedRemoteChannel {
  let output = Bytes::make(max_frame_bytes + 4, b'\x00')
  {
    socket,
    tls,
    input: FixedArray::make(max_frame_bytes + 4, b'\x00'),
    output,
    output_alias: @buffer_alias.retain_bytes_as_fixed_array(output),
    timeout_millis,
    binding,
    open: true,
    reading: false,
    writing: false,
  }
}

///|
/// Consumes the socket, including on failure. Verifies CA AND hostname before
/// sending the secret. One deadline covers TLS plus the complete startup join.
pub async fn connect(
  socket : @socket.Tcp,
  credential : RemoteCredential,
  expected_binding : Bytes,
  server_name~ : String,
  ca_pem_file~ : String,
  max_frame_bytes~ : Int,
  timeout_millis~ : Int,
) -> AuthenticatedRemoteChannel {
  errdefer socket.close()
  validate_limits(expected_binding, max_frame_bytes, timeout_millis)
  if server_name.is_empty() || ca_pem_file.is_empty() {
    raise Configuration
  }
  @async.with_timeout(timeout_millis, () => {
    let tls = @tls.Tls::client(
      socket,
      host=server_name,
      trust=CustomPemFile(ca_pem_file),
    )
    errdefer tls.close()
    tls.write(hello(credential, expected_binding))
    let response = FixedArray::make(8, b'\x00')
    read_exact(tls, response, 0, 8)
    if !matches(response, b"LFREADY1") {
      raise Authentication
    }
    channel(socket, tls, expected_binding, max_frame_bytes, timeout_millis)
  })
}

///|
/// Consumes the socket, including on failure. Expected binding and credential
/// come from independent local deployment admission, never the incoming hello.
pub async fn accept(
  socket : @socket.Tcp,
  credential : RemoteCredential,
  expected_binding : Bytes,
  certificate_pem_file~ : String,
  private_key_pem_file~ : String,
  max_frame_bytes~ : Int,
  timeout_millis~ : Int,
) -> AuthenticatedRemoteChannel {
  errdefer socket.close()
  validate_limits(expected_binding, max_frame_bytes, timeout_millis)
  if certificate_pem_file.is_empty() || private_key_pem_file.is_empty() {
    raise Configuration
  }
  @async.with_timeout(timeout_millis, () => {
    let tls = @tls.Tls::server(
      socket,
      certificate_file=certificate_pem_file,
      certificate_type=PEM,
      private_key_file=private_key_pem_file,
      private_key_type=PEM,
    )
    errdefer tls.close()
    let expected = hello(credential, expected_binding)
    let received = FixedArray::make(expected.length(), b'\x00')
    read_exact(tls, received, 0, received.length())
    let admitted = matches(received, expected)
    for i in 0..<received.length() {
      received[i] = b'\x00'
    }
    if !admitted {
      raise Authentication
    }
    tls.write(b"LFREADY1")
    channel(socket, tls, expected_binding, max_frame_bytes, timeout_millis)
  })
}

///|
pub fn AuthenticatedRemoteChannel::close(
  self : AuthenticatedRemoteChannel,
) -> Unit {
  if self.open {
    self.open = false
    self.tls.close()
    self.socket.close()
  }
}

///|
pub fn AuthenticatedRemoteChannel::is_open(
  self : AuthenticatedRemoteChannel,
) -> Bool {
  self.open
}

///|
pub fn AuthenticatedRemoteChannel::verify_binding(
  self : AuthenticatedRemoteChannel,
  expected : Bytes,
) -> Unit raise RemoteTlsError {
  if !self.open {
    raise Closed
  }
  if self.binding != expected {
    self.close()
    raise Authentication
  }
}

///|
pub fn AuthenticatedRemoteChannel::capacity(
  self : AuthenticatedRemoteChannel,
) -> Int {
  self.input.length() - 4
}

///|
/// Snapshots the bounded source before yielding. Cancellation poisons the
/// entire stream; retrying a TLS write after partial progress is forbidden.
pub async fn AuthenticatedRemoteChannel::send(
  self : AuthenticatedRemoteChannel,
  source : FixedArray[Byte],
  length : Int,
) -> Unit {
  if !self.open {
    raise Closed
  }
  if self.writing {
    raise Busy
  }
  errdefer self.close()
  if length <= 0 ||
    length > source.length() ||
    length > self.output.length() - 4 {
    raise Frame
  }
  self.writing = true
  defer {
    self.writing = false
  }
  for i in 0..<4 {
    self.output_alias[i] = (length.reinterpret_as_uint() >> (8 * i)).to_byte()
  }
  for i in 0..<length {
    self.output_alias[i + 4] = source[i]
  }
  @async.with_timeout(self.timeout_millis, () => {
    let mut cursor = 0
    while cursor < length + 4 {
      let count = self.tls.write_once(
        self.output,
        offset=cursor,
        len=length + 4 - cursor,
      )
      if count <= 0 {
        raise Closed
      }
      cursor += count
    }
  })
}

///|
/// One absolute deadline for header and body. Partial bytes stay private.
pub async fn AuthenticatedRemoteChannel::receive(
  self : AuthenticatedRemoteChannel,
  destination : FixedArray[Byte],
) -> Int {
  if !self.open {
    raise Closed
  }
  if self.reading {
    raise Busy
  }
  self.reading = true
  defer {
    self.reading = false
  }
  errdefer self.close()
  @async.with_timeout(self.timeout_millis, () => {
    read_exact(self.tls, self.input, 0, 4)
    let mut length = 0U
    for i in 0..<4 {
      length = length | (self.input[i].to_uint() << (8 * i))
    }
    if length == 0U ||
      length > (self.input.length() - 4).reinterpret_as_uint() ||
      length > destination.length().reinterpret_as_uint() {
      raise Frame
    }
    let count = length.reinterpret_as_int()
    read_exact(self.tls, self.input, 4, count)
    for i in 0..<count {
      destination[i] = self.input[i + 4]
    }
    count
  })
}
