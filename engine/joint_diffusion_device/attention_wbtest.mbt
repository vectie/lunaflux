///|
test "attention block projection geometry supports unequal hidden and inner widths" {
  let (hidden, matrix) = attention_projection_geometry(39744, 5376, 56, 128)
  assert_eq(hidden, 39744L * 5376L * 2L)
  assert_eq(matrix, 77070336L)
  let scratch = staged_attention_scratch_bytes(
    rows=39744,
    head_count=56,
    head_dimension=128,
  )
  assert_eq(scratch, 2279079936L)
  assert_eq(scratch / 4L * 6L - scratch, 1139539968L)
  for
    (rows, width, heads, dim) in [
      (1, 0, 1, 1),
      (1, 4, 2147483647, 2),
      (2147483647, 2, 1, 1),
      (1, 2147483647, 2, 1),
    ] {
    try attention_projection_geometry(rows, width, heads, dim) catch {
      @device.DeviceError::InvalidArgument(AotKernelLaunch) => ()
      _ => fail("wrong projection error")
    } noraise {
      _ => fail("overflowing projection accepted")
    }
  }
}
