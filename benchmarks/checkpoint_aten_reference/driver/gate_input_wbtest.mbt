///|
test "gate attribution separates cancellation from propagated operand changes" {
  let first = b"\x80\x3f\x80\x3f\x80\x3f"
  let second = b"\x81\x3f\x80\x3f\x80\x3f"
  let weight = b"\xc8\x42\xc8\xc2\x80\x3f"
  let result = dot_sensitivity(first, second, weight)
  assert_eq(result.first, 1.0)
  assert_eq(result.second, 1.78125)
  assert_eq(result.signed_delta, 0.78125)
  assert_eq(result.absolute_delta, 0.78125)
  assert_eq(result.absolute_first, 201.0)
  assert_eq(result.absolute_second, 201.78125)
  assert_eq(result.largest_delta_index, 0)
  assert_eq(result.largest_delta, 0.78125)
  assert_eq(result.largest_input_delta_index, 0)
  assert_eq(result.largest_input_delta, 0.0078125)
  let same = dot_sensitivity(first, first, weight)
  assert_eq(same.signed_delta, 0.0)
  assert_eq(same.input_delta_squared, 0.0)
  assert_eq(diagnostic_ratio(0.0, 0.0), 0.0)
  assert_true(diagnostic_ratio(1.0, 0.0).is_inf())
  for
    (left, right, weights) in [
      (b"", b"", b""),
      (b"\x80", b"\x80", b"\x80"),
      (first, second, b"\x80\x3f"),
      (b"\x80\x7f", b"\x80\x3f", b"\x80\x3f"),
    ] {
    try dot_sensitivity(left, right, weights) catch {
      _ => ()
    } noraise {
      _ => fail("invalid dot operands accepted")
    }
  }
}
