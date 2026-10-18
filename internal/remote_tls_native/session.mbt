///|
pub(all) suberror ProviderError {
  InvalidIdentity
  Closed
  Transport
}

///|
pub struct ServerIdentity {
  priv mut native : NativeIdentity?
}

///|
/// Synchronous startup operation. Sessions retain the prepared context;
/// accepting a connection never opens certificate or key files.
pub fn ServerIdentity::load(
  certificate : String,
  key : String,
) -> ServerIdentity raise ProviderError {
  let native = load_identity(@utf8.encode(certificate), @utf8.encode(key))
  guard identity_valid(native) else { raise InvalidIdentity }
  { native: Some(native), }
}

///|
pub fn ServerIdentity::close(self : ServerIdentity) -> Unit {
  if self.native is Some(native) {
    self.native = None
    release_identity(native)
  }
}

///|
/// Single-threaded async owner with serialized ciphertext publication.
/// Read and write may overlap while a network operation yields.
pub struct ServerSession {
  priv mut native : NativeSession?
  priv socket : @socket.Tcp
  priv input : FixedArray[Byte]
  priv output : Bytes
  priv output_alias : FixedArray[Byte]
  priv output_lock : @async.Mutex
}

///|
fn ServerSession::handle(
  self : ServerSession,
) -> NativeSession raise ProviderError {
  match self.native {
    Some(native) => native
    None => raise Closed
  }
}

///|
pub fn ServerSession::close(self : ServerSession) -> Unit {
  if self.native is Some(native) {
    self.native = None
    self.socket.close()
    release_session(native)
  }
}

///|
async fn ServerSession::flush(self : ServerSession) -> Unit {
  self.output_lock.acquire()
  defer self.output_lock.release()
  while true {
    let count = drain(self.handle(), self.output_alias, self.output.length())
    guard count >= 0 else { raise Transport }
    if count == 0 {
      return
    }
    let mut cursor = 0
    while cursor < count {
      let written = self.socket.write_once(
        self.output,
        offset=cursor,
        len=count - cursor,
      )
      guard written > 0 else { raise Transport }
      cursor += written
    }
  }
}

///|
async fn ServerSession::supply(self : ServerSession) -> Unit {
  let count = self.socket.read(self.input)
  guard count > 0 else { raise Transport }
  guard feed(self.handle(), self.input, count) == count else { raise Transport }
}

///|
/// Consumes the socket, including failed or cancelled handshakes.
pub async fn ServerIdentity::accept(
  self : ServerIdentity,
  socket : @socket.Tcp,
) -> ServerSession {
  errdefer socket.close()
  guard self.native is Some(identity) else { raise Closed }
  let native = create_session(identity)
  guard session_valid(native) else { raise Transport }
  let output = Bytes::make(32768, b'\x00')
  let session : ServerSession = {
    native: Some(native),
    socket,
    input: FixedArray::make(32768, b'\x00'),
    output,
    output_alias: @buffer_alias.retain_bytes_as_fixed_array(output),
    output_lock: @async.Mutex::Mutex(),
  }
  errdefer session.close()
  while true {
    let status = handshake(session.handle())
    session.flush()
    match status {
      1 => break session
      -2 => session.supply()
      -3 => ()
      _ => raise Transport
    }
  } nobreak {
    raise Transport
  }
}

///|
pub async fn ServerSession::read(
  self : ServerSession,
  destination : FixedArray[Byte],
  offset~ : Int,
  max_len~ : Int,
) -> Int {
  guard offset >= 0 && max_len > 0 && offset <= destination.length() - max_len else {
    raise Transport
  }
  errdefer self.close()
  while true {
    let count = decrypt(self.handle(), destination, offset, max_len)
    self.flush()
    match count {
      -2 => self.supply()
      -3 => ()
      -1 => raise Transport
      count => break count
    }
  } nobreak {
    raise Transport
  }
}

///|
pub async fn ServerSession::write_once(
  self : ServerSession,
  source : Bytes,
  offset~ : Int,
  len~ : Int,
) -> Int {
  guard offset >= 0 && len > 0 && offset <= source.length() - len else {
    raise Transport
  }
  errdefer self.close()
  // A bounded TLS record keeps the memory BIO and ciphertext arena bounded.
  let count = encrypt(self.handle(), source, offset, Int::min(len, 16384))
  guard count > 0 else { raise Transport }
  self.flush()
  count
}
