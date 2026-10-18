///|
#external
priv type NativeIdentity

///|
#external
priv type NativeSession

///|
#borrow(certificate, key)
extern "C" fn load_identity(certificate : Bytes, key : Bytes) -> NativeIdentity = "lunaflux_tls_identity_load"

///|
extern "C" fn identity_valid(identity : NativeIdentity) -> Bool = "lunaflux_tls_identity_valid"

///|
extern "C" fn release_identity(identity : NativeIdentity) -> Unit = "lunaflux_tls_identity_release"

///|
extern "C" fn create_session(identity : NativeIdentity) -> NativeSession = "lunaflux_tls_session_create"

///|
extern "C" fn session_valid(session : NativeSession) -> Bool = "lunaflux_tls_session_valid"

///|
extern "C" fn release_session(session : NativeSession) -> Unit = "lunaflux_tls_session_release"

///|
extern "C" fn handshake(session : NativeSession) -> Int = "lunaflux_tls_handshake"

///|
#borrow(bytes)
extern "C" fn feed(
  session : NativeSession,
  bytes : FixedArray[Byte],
  count : Int,
) -> Int = "lunaflux_tls_feed"

///|
#borrow(bytes)
extern "C" fn drain(
  session : NativeSession,
  bytes : FixedArray[Byte],
  capacity : Int,
) -> Int = "lunaflux_tls_drain"

///|
#borrow(bytes)
extern "C" fn decrypt(
  session : NativeSession,
  bytes : FixedArray[Byte],
  offset : Int,
  count : Int,
) -> Int = "lunaflux_tls_read"

///|
#borrow(bytes)
extern "C" fn encrypt(
  session : NativeSession,
  bytes : Bytes,
  offset : Int,
  count : Int,
) -> Int = "lunaflux_tls_write"
