///|
test "timestamp formatting matches exact Double .1f ties and decimal edges" {
  for
    (value, expected) in [
      (0.0, "<0.0 seconds>"),
      (0.15, "<0.1 seconds>"),
      (0.25, "<0.2 seconds>"),
      (0.35, "<0.3 seconds>"),
      (0.75, "<0.8 seconds>"),
      (1.25, "<1.2 seconds>"),
      (1.35, "<1.4 seconds>"),
      (9.99, "<10.0 seconds>"),
      (0.000001, "<0.0 seconds>"),
    ] {
    assert_eq(timestamp(value), expected)
  }
  try timestamp(-1.0) catch {
    _ => ()
  } noraise {
    _ => fail("negative timestamp")
  }
}
