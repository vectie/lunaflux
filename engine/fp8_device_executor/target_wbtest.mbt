///|
fn fp8_executor_target(major : Int, minor : Int) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=major,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
test "FP8 executor target gate admits exact sm120 only" {
  assert_true(fp8_executor_target_supported(fp8_executor_target(8, 9)))
  assert_true(fp8_executor_target_supported(fp8_executor_target(9, 0)))
  assert_true(fp8_executor_target_supported(fp8_executor_target(12, 0)))
  assert_false(fp8_executor_target_supported(fp8_executor_target(12, 1)))
  assert_false(fp8_executor_target_supported(fp8_executor_target(10, 0)))
}
