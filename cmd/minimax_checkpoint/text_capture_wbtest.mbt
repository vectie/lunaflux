///|
test "layer capture host budget is checked before any device preparation" {
  validate_text_capture_budget(30720L, 1566720L)
  for
    (bytes, budget) in [
      (30720L, 1566719L),
      (0L, 1566720L),
      (30720L, 0L),
      (30720L, 2147483648L),
    ] {
    try validate_text_capture_budget(bytes, budget) catch {
      _ => ()
    } noraise {
      _ => fail("invalid capture budget admitted")
    }
  }
}
