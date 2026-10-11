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

///|
test "matched observations budget literal inputs and outputs independently" {
  validate_text_capture_budget(30720L, 3102720L, matched_inputs=true)
  try
    validate_text_capture_budget(30720L, 3102719L, matched_inputs=true)
  catch {
    _ => ()
  } noraise {
    _ => fail("original input storage was omitted from the host budget")
  }
}
