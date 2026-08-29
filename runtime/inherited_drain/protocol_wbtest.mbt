///|
test "fixed inherited drain v1 vocabulary is exact" {
  let command = FixedArray::make(INHERITED_DRAIN_FRAME_BYTES, b'\x00')
  for index, byte in [b'L', b'F', b'D', b'1', b'D', b'R', b'N', b'\n'] {
    command[index] = byte
  }
  assert_true(drain_command_matches(command))
  command[7] = b'X'
  assert_false(drain_command_matches(command))

  let response = FixedArray::make(INHERITED_DRAIN_FRAME_BYTES, b'\x00')
  fill_drain_response(response, InheritedDrainAccepted)
  assert_eq(response[4], b'A')
  assert_eq(response[5], b'C')
  assert_eq(response[6], b'K')
  fill_drain_response(response, InheritedDrainAlreadyDraining)
  assert_eq(response[4], b'I')
  assert_eq(response[5], b'D')
  assert_eq(response[6], b'M')
  fill_drain_response(response, InheritedDrainAlreadyClosed)
  assert_eq(response[4], b'C')
  assert_eq(response[5], b'L')
  assert_eq(response[6], b'S')
}
