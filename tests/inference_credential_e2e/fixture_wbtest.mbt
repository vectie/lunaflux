///|
priv type NativeCredentialPeer

///|
extern "c" fn fixture_install(mode : Int) -> NativeCredentialPeer = "lunaflux_inference_credential_fixture_install"

///|
#borrow(peer, source)
extern "c" fn fixture_send(
  peer : NativeCredentialPeer,
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int = "lunaflux_inference_credential_fixture_send"

///|
#borrow(peer)
extern "c" fn fixture_close_write(peer : NativeCredentialPeer) -> Int = "lunaflux_inference_credential_fixture_close_write"

///|
#borrow(peer)
extern "c" fn fixture_close(peer : NativeCredentialPeer) -> Int = "lunaflux_inference_credential_fixture_close"

///|
extern "c" fn fixture_clear_fixed() -> Int = "lunaflux_inference_credential_fixture_clear_fixed"

///|
extern "c" fn fixture_fixed_closed() -> Bool = "lunaflux_inference_credential_fixture_fixed_closed"

///|
fn credential_frame(secret : FixedArray[Byte]) -> FixedArray[Byte] {
  let frame = FixedArray::make(12 + secret.length(), b'\x00')
  frame[0] = b'L'
  frame[1] = b'F'
  frame[2] = b'C'
  frame[3] = b'1'
  frame[4] = b'K'
  frame[5] = b'E'
  frame[6] = b'Y'
  frame[7] = b'\n'
  frame[8] = (secret.length() & 0xff).to_byte()
  frame[9] = ((secret.length() >> 8) & 0xff).to_byte()
  for index in 0..<secret.length() {
    frame[12 + index] = secret[index]
  }
  frame
}

///|
fn progress_until_terminal(
  owner : @credential.InheritedInferenceCredentialOwner,
) -> @credential.InheritedInferenceCredentialProgress {
  for _ in 0..<1000 {
    let progress = owner.progress()
    if progress != InheritedInferenceCredentialPending {
      return progress
    }
  }
  InheritedInferenceCredentialPending
}
