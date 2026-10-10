///|
test "shared request sequence frame counts are exact and bounded before device startup" {
  assert_eq(sequence_frame_counts(b"1, 24\n", 2, 256), [1, 24])
  for bytes in [b"1", b"1,2,3", b"1,0", b"1,-1", b"1,257", b"1,x", b""] {
    try sequence_frame_counts(bytes, 2, 256) catch {
      _ => ()
    } noraise {
      _ => fail("invalid sequence accepted")
    }
  }
  try sequence_frame_counts(b"1", 0, 256) catch {
    _ => ()
  } noraise {
    _ => fail("zero requests accepted")
  }
}
