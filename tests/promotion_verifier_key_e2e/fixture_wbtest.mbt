///|
priv type NativePromotionPeer

///|
extern "c" fn fixture_install(mode : Int) -> NativePromotionPeer = "lunaflux_promotion_verifier_fixture_install"

///|
#borrow(peer, source)
extern "c" fn fixture_send(
  peer : NativePromotionPeer,
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int = "lunaflux_promotion_verifier_fixture_send"

///|
#borrow(peer)
extern "c" fn fixture_close_write(peer : NativePromotionPeer) -> Int = "lunaflux_promotion_verifier_fixture_close_write"

///|
#borrow(peer)
extern "c" fn fixture_close(peer : NativePromotionPeer) -> Int = "lunaflux_promotion_verifier_fixture_close"

///|
extern "c" fn fixture_clear_fixed() -> Int = "lunaflux_promotion_verifier_fixture_clear_fixed"

///|
extern "c" fn fixture_fixed_closed() -> Bool = "lunaflux_promotion_verifier_fixture_fixed_closed"

///|
extern "c" fn fixture_fixed_cloexec() -> Bool = "lunaflux_promotion_verifier_fixture_fixed_cloexec"

///|
fn verifier_frame() -> FixedArray[Byte] {
  let key = b"0123456789abcdef0123456789abcdef"
  let verifier = b"release-key-1"
  let authority = b"release-authority-1"
  let payload_length = 64 +
    key.length() +
    verifier.length() +
    authority.length()
  let frame = FixedArray::make(16 + payload_length, b'\x00')
  let magic : FixedArray[Byte] = [
    b'L', b'F', b'P', b'1', b'K', b'E', b'Y', b'\n',
  ]
  for index in 0..<magic.length() {
    frame[index] = magic[index]
  }
  frame[8] = key.length().to_byte()
  frame[9] = verifier.length().to_byte()
  frame[10] = authority.length().to_byte()
  frame[12] = (payload_length & 0xff).to_byte()
  frame[13] = ((payload_length >> 8) & 0xff).to_byte()
  for index in 0..<32 {
    frame[16 + index] = (index + 1).to_byte()
    frame[48 + index] = (index + 33).to_byte()
  }
  let mut cursor = 80
  for byte in key {
    frame[cursor] = byte
    cursor += 1
  }
  for byte in verifier {
    frame[cursor] = byte
    cursor += 1
  }
  for byte in authority {
    frame[cursor] = byte
    cursor += 1
  }
  frame
}

///|
fn progress_until_terminal(
  owner : @manifest.LunaStartupVerifierKeyOwner,
) -> @manifest.LunaStartupVerifierKeyProgress {
  for _ in 0..<1000 {
    let progress = owner.progress()
    if progress != LunaStartupVerifierKeyPending {
      return progress
    }
  }
  LunaStartupVerifierKeyPending
}
