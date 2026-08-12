///|
test "reference arguments are bounded before artifact loading" {
  assert_eq(parse_positive("4"), 4)
  assert_eq(parse_token_ids("1,42,99"), [1, 42, 99])
  for text in ["0", "4097", "invalid"] {
    try parse_positive(text) catch {
      InvalidReferenceArgument(MaximumNewTokens) => ()
      _ => fail("unexpected maximum-new-token error")
    } noraise {
      _ => fail("expected invalid maximum new tokens")
    }
  }
  for text in ["", "1,", "-1", "x"] {
    try parse_token_ids(text) catch {
      InvalidReferenceArgument(TokenIds) => ()
      _ => fail("unexpected token ID error")
    } noraise {
      _ => fail("expected invalid token IDs")
    }
  }
}
