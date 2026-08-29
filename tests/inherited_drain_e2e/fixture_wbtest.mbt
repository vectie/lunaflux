///|
priv type NativeDrainPeer

///|
extern "c" fn fixture_install(mode : Int) -> NativeDrainPeer = "lunaflux_inherited_drain_fixture_install"

///|
#borrow(peer, source)
extern "c" fn fixture_send(
  peer : NativeDrainPeer,
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int = "lunaflux_inherited_drain_fixture_send"

///|
#borrow(peer, destination)
extern "c" fn fixture_read(
  peer : NativeDrainPeer,
  destination : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int = "lunaflux_inherited_drain_fixture_read"

///|
#borrow(peer)
extern "c" fn fixture_close_write(peer : NativeDrainPeer) -> Int = "lunaflux_inherited_drain_fixture_close_write"

///|
#borrow(peer)
extern "c" fn fixture_close(peer : NativeDrainPeer) -> Int = "lunaflux_inherited_drain_fixture_close"

///|
extern "c" fn fixture_clear_fixed() -> Int = "lunaflux_inherited_drain_fixture_clear_fixed"

///|
extern "c" fn fixture_fixed_closed() -> Bool = "lunaflux_inherited_drain_fixture_fixed_closed"

///|
fn drain_command() -> FixedArray[Byte] {
  [b'L', b'F', b'D', b'1', b'D', b'R', b'N', b'\n']
}

///|
fn drain_expected_response(
  response : @drain.InheritedDrainResponse,
) -> FixedArray[Byte] {
  match response {
    InheritedDrainAccepted => [b'L', b'F', b'D', b'1', b'A', b'C', b'K', b'\n']
    InheritedDrainAlreadyDraining =>
      [b'L', b'F', b'D', b'1', b'I', b'D', b'M', b'\n']
    InheritedDrainAlreadyClosed =>
      [b'L', b'F', b'D', b'1', b'C', b'L', b'S', b'\n']
  }
}

///|
fn read_exact_response(peer : NativeDrainPeer) -> FixedArray[Byte] raise {
  let output = FixedArray::make(8, b'\x00')
  let mut cursor = 0
  let mut attempts = 0
  while cursor < output.length() && attempts < 1000 {
    let count = fixture_read(peer, output, cursor, output.length() - cursor)
    if count > 0 {
      cursor += count
    } else if count < 0 {
      abort("inherited drain response channel closed early")
    }
    attempts += 1
  }
  assert_eq(cursor, output.length())
  output
}

///|
fn assert_response(
  actual : FixedArray[Byte],
  response : @drain.InheritedDrainResponse,
) -> Unit raise {
  let expected = drain_expected_response(response)
  for index in 0..<expected.length() {
    assert_eq(actual[index], expected[index])
  }
}
