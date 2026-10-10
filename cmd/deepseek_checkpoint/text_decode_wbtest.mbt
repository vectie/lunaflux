///|
test "offline output token list retains order and rejects malformed geometry" {
  assert_eq(parse_decode_ids("0,19923,1", 3), [0, 19923, 1])
  assert_eq(parse_decode_ids("", 1), [])
  for
    (source, maximum) in [("0,1", 1), ("-1", 1), ("0,", 2), ("x", 1), ("0", 0)] {
    try parse_decode_ids(source, maximum) catch {
      _ => ()
    } noraise {
      _ => fail("invalid decoded-token list accepted")
    }
  }
}
