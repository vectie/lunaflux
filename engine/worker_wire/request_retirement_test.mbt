///|
test "request retirement preserves exact identities and rejects malformed frames" {
  let buffer = FixedArray::make(40, b'\x00')
  let command : RequestRetirement = {
    predecessor: 0UL,
    request_id: 9UL,
    generation: 3UL,
    acknowledged: false,
  }
  command.encode(buffer)
  assert_eq(decode_request_retirement(buffer, 40), command)
  let ack = { ..command, acknowledged: true, }
  ack.encode(buffer)
  assert_eq(decode_request_retirement(buffer, 40), ack)
  buffer[36] = b'\x01'
  try decode_request_retirement(buffer, 40) catch {
    _ => ()
  } noraise {
    _ => fail("reserved bytes accepted")
  }
  try decode_request_retirement(buffer, 39) catch {
    _ => ()
  } noraise {
    _ => fail("short retirement accepted")
  }
}
