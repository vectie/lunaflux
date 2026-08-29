///|
test "v1 size preflight rejects hostile counts without arithmetic overflow" {
  assert_true(i8_admission_v1_constants_are_consistent())
  assert_eq(i8_admission_v1_size_within(2, 4, 1721), Some(1721))
  assert_eq(i8_admission_v1_size_within(2, 4, 1720), None)
  assert_eq(i8_admission_v1_size_within(65536, 299590, 67108864), None)
  assert_eq(i8_admission_v1_size_within(2147483647, 2147483647, 67108864), None)
}

///|
test "writer checks every append and exact final length" {
  let exact = I8CanonicalWriter::new(8, 8)
  exact.append_i64(7L)
  assert_eq(exact.finish().length(), 8)
  let short = I8CanonicalWriter::new(8, 8)
  short.append_byte(1)
  try short.finish() catch {
    I8InertKernelCapabilityRejected(CanonicalBytes) => ()
    _ => fail("unexpected short canonical writer error")
  } noraise {
    _ => fail("short canonical writer was finalized")
  }
  let bounded = I8CanonicalWriter::new(1, 1)
  bounded.append_byte(1)
  try bounded.append_byte(2) catch {
    I8InertKernelCapabilityRejected(CanonicalBytes) => ()
    _ => fail("unexpected writer overflow error")
  } noraise {
    _ => fail("canonical writer exceeded its exact envelope")
  }
}
