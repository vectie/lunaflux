///|
fn numeric_device_fixture(
  major : Int,
  minor : Int,
  bf16 : Bool,
  cublas_lt? : Bool = false,
) -> DeviceCapability {
  {
    compute_major: major,
    compute_minor: minor,
    total_memory_bytes: 16L * 1024L * 1024L * 1024L,
    supports_bf16: bf16,
    supports_cublas_lt: cublas_lt,
  }
}

///|
test "FP8 numeric capability is derived from exact observed target facts" {
  let first = admit_finite_fp8_e4m3_w8a8(numeric_device_fixture(8, 9, true))
  let later = admit_finite_fp8_e4m3_w8a8(numeric_device_fixture(9, 0, true))
  let blackwell = admit_finite_fp8_e4m3_w8a8(
    numeric_device_fixture(12, 0, true, cublas_lt=true),
  )
  let library_changed = admit_finite_fp8_e4m3_w8a8(
    numeric_device_fixture(8, 9, true, cublas_lt=true),
  )
  assert_eq(first.version(), DeviceNumericCapabilityVersion::v1())
  assert_eq(first.feature(), FiniteFp8E4M3W8A8F32AccumulateBf16OutputV1)
  assert_eq(first.compute_major(), 8)
  assert_eq(first.compute_minor(), 9)
  assert_eq(
    first.digest().as_hex(),
    "4a1c907be641757f5e071d78a90dedc6ae725f2b40c0828ac4f78b11e7d4edf4",
  )
  assert_eq(
    later.digest().as_hex(),
    "d5a9f84f2a7c4de400db8fea269452275b05b617366e79574e977b912a81d931",
  )
  assert_false(first.digest() == later.digest())
  assert_eq(blackwell.compute_major(), 12)
  assert_eq(blackwell.compute_minor(), 0)
  assert_true(blackwell.supports_bf16())
  assert_true(blackwell.supports_cublas_lt())
  assert_false(first.digest() == blackwell.digest())
  assert_false(later.digest() == blackwell.digest())
  assert_false(first.digest() == library_changed.digest())
}

///|
test "unsupported and incomplete FP8 targets fail before capability publication" {
  let cases = [
    (numeric_device_fixture(8, 0, true), UnsupportedArchitecture),
    (numeric_device_fixture(8, 8, true), UnsupportedArchitecture),
    (numeric_device_fixture(9, 1, true), UnsupportedArchitecture),
    (numeric_device_fixture(10, 0, true), UnsupportedArchitecture),
    (numeric_device_fixture(12, 1, true), UnsupportedArchitecture),
    (numeric_device_fixture(9, 0, false), MissingBf16BoundarySupport),
    (numeric_device_fixture(12, 0, false), MissingBf16BoundarySupport),
    (numeric_device_fixture(0, 0, true), InvalidObservedCapability),
  ]
  for item in cases {
    let (capability, expected) = item
    try admit_finite_fp8_e4m3_w8a8(capability) catch {
      InvalidDeviceNumericCapability(rule) => assert_eq(rule, expected)
    } noraise {
      _ => fail("unsupported FP8 target published a numeric capability")
    }
  }
}
