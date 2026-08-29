///|
const INHERITED_DRAIN_FRAME_BYTES : Int = 8

///|
fn drain_command_matches(source : FixedArray[Byte]) -> Bool {
  source.length() == INHERITED_DRAIN_FRAME_BYTES &&
  source[0] == b'L' &&
  source[1] == b'F' &&
  source[2] == b'D' &&
  source[3] == b'1' &&
  source[4] == b'D' &&
  source[5] == b'R' &&
  source[6] == b'N' &&
  source[7] == b'\n'
}

///|
fn fill_drain_response(
  destination : FixedArray[Byte],
  response : InheritedDrainResponse,
) -> Unit {
  destination[0] = b'L'
  destination[1] = b'F'
  destination[2] = b'D'
  destination[3] = b'1'
  match response {
    InheritedDrainAccepted => {
      destination[4] = b'A'
      destination[5] = b'C'
      destination[6] = b'K'
    }
    InheritedDrainAlreadyDraining => {
      destination[4] = b'I'
      destination[5] = b'D'
      destination[6] = b'M'
    }
    InheritedDrainAlreadyClosed => {
      destination[4] = b'C'
      destination[5] = b'L'
      destination[6] = b'S'
    }
  }
  destination[7] = b'\n'
}
