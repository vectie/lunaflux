///|
test "I8 numeric capability has distinct canonical target identity" {
  let first = admit_symmetric_i8_weight_only_per_output_channel_v1(
    numeric_device_fixture(8, 9, true),
  )
  let later = admit_symmetric_i8_weight_only_per_output_channel_v1(
    numeric_device_fixture(9, 0, true),
  )
  let blackwell = admit_symmetric_i8_weight_only_per_output_channel_v1(
    numeric_device_fixture(12, 0, true, cublas_lt=true),
  )
  let library_changed = admit_symmetric_i8_weight_only_per_output_channel_v1(
    numeric_device_fixture(8, 9, true, cublas_lt=true),
  )
  let fp8 = admit_finite_fp8_e4m3_w8a8(numeric_device_fixture(8, 9, true))
  assert_eq(first.version(), DeviceNumericCapabilityVersion::v1())
  assert_eq(
    first.feature(),
    SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1,
  )
  assert_eq(first.compute_major(), 8)
  assert_eq(first.compute_minor(), 9)
  assert_eq(
    first.digest().as_hex(),
    "78f14c2730d5a6911eabdaf2b7b206f226c8be61f7c1c550a68a68cfe74c3c5b",
  )
  assert_eq(
    later.digest().as_hex(),
    "ef45e431392fbd587e0a15b3882a8a3b4e1bf9ad3c1944b084d456973a66fce6",
  )
  assert_false(first.digest() == later.digest())
  assert_eq(blackwell.compute_major(), 12)
  assert_eq(blackwell.compute_minor(), 0)
  assert_true(blackwell.supports_bf16())
  assert_true(blackwell.supports_cublas_lt())
  assert_false(first.digest() == blackwell.digest())
  assert_false(later.digest() == blackwell.digest())
  assert_false(first.digest() == library_changed.digest())
  assert_false(first.digest() == fp8.digest())
  assert_false(first.canonical_bytes() == fp8.canonical_bytes())
}

///|
test "I8 software target allowlist is exact and requires BF16 boundaries" {
  let cases = [
    (numeric_device_fixture(8, 0, true), UnsupportedArchitecture),
    (numeric_device_fixture(8, 8, true), UnsupportedArchitecture),
    (numeric_device_fixture(8, 10, true), UnsupportedArchitecture),
    (numeric_device_fixture(9, 1, true), UnsupportedArchitecture),
    (numeric_device_fixture(10, 0, true), UnsupportedArchitecture),
    (numeric_device_fixture(12, 1, true), UnsupportedArchitecture),
    (numeric_device_fixture(8, 9, false), MissingBf16BoundarySupport),
    (numeric_device_fixture(9, 0, false), MissingBf16BoundarySupport),
    (numeric_device_fixture(12, 0, false), MissingBf16BoundarySupport),
    (numeric_device_fixture(0, 0, true), InvalidObservedCapability),
  ]
  for item in cases {
    let (capability, expected) = item
    try admit_symmetric_i8_weight_only_per_output_channel_v1(capability) catch {
      InvalidDeviceNumericCapability(rule) => assert_eq(rule, expected)
    } noraise {
      _ => fail("unsupported I8 software target published a capability")
    }
  }
}
