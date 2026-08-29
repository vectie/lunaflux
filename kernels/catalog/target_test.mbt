///|
fn observed_numeric_capability(
  major : Int,
  minor : Int,
  cublas_lt : Bool,
) -> @device.DeviceNumericCapability {
  let observed = try! @device.DeviceCapability::from_observed_facts(
    compute_major=major,
    compute_minor=minor,
    total_memory_bytes=1L,
    supports_bf16=true,
    supports_cublas_lt=cublas_lt,
  )
  try! @device.admit_finite_fp8_e4m3_w8a8(observed)
}

///|
fn numeric_target(
  major : Int,
  minor : Int,
  bf16 : Bool,
  cublas_lt : Bool,
) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=major,
    compute_minor=minor,
    supports_bf16=bf16,
    supports_cublas_lt=cublas_lt,
  )
}

///|
test "device target exactly matches admitted numeric capability evidence" {
  let numeric = observed_numeric_capability(8, 9, true)
  assert_true(
    numeric_target(8, 9, true, true).matches_numeric_capability(numeric),
  )
}

///|
test "device target rejects every mismatched numeric capability fact" {
  let numeric = observed_numeric_capability(8, 9, true)
  let hostile = [
    numeric_target(9, 9, true, true),
    numeric_target(8, 8, true, true),
    numeric_target(8, 9, false, true),
    numeric_target(8, 9, true, false),
  ]
  for target in hostile {
    assert_false(target.matches_numeric_capability(numeric))
  }
}
