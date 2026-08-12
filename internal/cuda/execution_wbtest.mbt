///|
extern "c" fn raw_test_execution_boundary(cycles : Int) -> Int = "lunaflux_cuda_test_execution_boundary"

///|
extern "c" fn raw_test_transfer_boundary(cycles : Int) -> Int = "lunaflux_cuda_test_transfer_boundary"

///|
extern "c" fn raw_test_region_validation(cycles : Int) -> Int = "lunaflux_cuda_test_region_validation"

///|
test "native region validation checks physical addresses and close guards" {
  assert_eq(raw_test_region_validation(256), 0)
}

///|
test "native transfers retain balanced retryable ownership" {
  assert_eq(raw_test_transfer_boundary(256), 0)
}

///|
test "native GEMM retains balanced retryable ownership" {
  assert_eq(raw_test_execution_boundary(256), 0)
}

///|
test "native execution probe rejects unbounded repetition" {
  assert_eq(raw_test_transfer_boundary(0), -2)
  assert_eq(raw_test_transfer_boundary(10001), -2)
  assert_eq(raw_test_execution_boundary(0), -2)
  assert_eq(raw_test_execution_boundary(10001), -2)
  assert_eq(raw_test_region_validation(0), -2)
  assert_eq(raw_test_region_validation(10001), -2)
}
