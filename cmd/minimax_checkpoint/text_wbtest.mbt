///|
test "text token IDs use signed I32 little endian with exact row length" {
  assert_eq(text_input_ids(b"\x01\x00\x00\x00\xff\xff\xff\xff", 2), [1, -1])
  for (bytes, rows) in [(b"", 0), (b"\x01\x00", 1), (b"\x01\x00\x00\x00", 2)] {
    try text_input_ids(bytes, rows) catch {
      _ => ()
    } noraise {
      _ => fail("invalid token geometry accepted")
    }
  }
}

///|
test "text hidden diagnostics preserve finite BF16 extrema and detect NaN or infinity" {
  assert_eq(text_hidden_statistics(b"\x00\x00\x80\x3f\x00\xc0"), (0, 2.0F))
  assert_eq(text_hidden_statistics(b"\x80\x7f\xc0\x7f\x80\xff"), (3, 0.0F))
  for bytes in [b"", b"\x00"] {
    try text_hidden_statistics(bytes) catch {
      _ => ()
    } noraise {
      _ => fail("empty or unaligned hidden output accepted")
    }
  }
}
