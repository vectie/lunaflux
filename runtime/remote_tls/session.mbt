///|
/// Prepared at startup and reusable across authenticated rank connections.
pub struct RemoteServerIdentity {
  priv identity : @native_tls.ServerIdentity
}

///|
pub fn RemoteServerIdentity::load(
  certificate_pem_file~ : String,
  private_key_pem_file~ : String,
) -> RemoteServerIdentity raise RemoteTlsError {
  let identity = @native_tls.ServerIdentity::load(
    certificate_pem_file, private_key_pem_file,
  ) catch {
    _ => raise Configuration
  }
  { identity, }
}

///|
pub fn RemoteServerIdentity::close(self : RemoteServerIdentity) -> Unit {
  self.identity.close()
}

///|
priv enum RemoteTlsSession {
  Client(@tls.Tls)
  Server(@native_tls.ServerSession)
}

///|
fn RemoteTlsSession::close(self : RemoteTlsSession) -> Unit {
  match self {
    Client(tls) => tls.close()
    Server(tls) => tls.close()
  }
}

///|
async fn RemoteTlsSession::read(
  self : RemoteTlsSession,
  bytes : FixedArray[Byte],
  offset~ : Int,
  max_len~ : Int,
) -> Int {
  match self {
    Client(tls) => tls.read(bytes, offset~, max_len~)
    Server(tls) => tls.read(bytes, offset~, max_len~)
  }
}

///|
async fn RemoteTlsSession::write_once(
  self : RemoteTlsSession,
  bytes : Bytes,
  offset~ : Int,
  len~ : Int,
) -> Int {
  match self {
    Client(tls) => tls.write_once(bytes, offset~, len~)
    Server(tls) => tls.write_once(bytes, offset~, len~)
  }
}

///|
async fn RemoteTlsSession::write(
  self : RemoteTlsSession,
  bytes : Bytes,
) -> Unit {
  let mut offset = 0
  while offset < bytes.length() {
    let count = self.write_once(bytes, offset~, len=bytes.length() - offset)
    guard count > 0 else { raise Closed }
    offset += count
  }
}
