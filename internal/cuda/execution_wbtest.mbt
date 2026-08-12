///|
extern "c" fn raw_test_execution_boundary(cycles : Int) -> Int = "lunaflux_cuda_test_execution_boundary"

///|
extern "c" fn raw_test_transfer_boundary(cycles : Int) -> Int = "lunaflux_cuda_test_transfer_boundary"

///|
#borrow(source)
extern "c" fn raw_test_fixed_transfer_boundary(
  source : FixedArray[Byte],
  cycles : Int,
) -> Int = "lunaflux_cuda_test_fixed_transfer_boundary"

///|
#borrow(destination)
extern "c" fn raw_test_fixed_readback_boundary(
  destination : FixedArray[Byte],
  cycles : Int,
) -> Int = "lunaflux_cuda_test_fixed_readback_boundary"

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
test "fixed host transfers borrow exact storage and enforce lifecycle guards" {
  let source = FixedArray::makei(32, index => index.to_byte())
  assert_eq(raw_test_fixed_transfer_boundary(source, 256), 0)
  assert_eq(source[5], 5)
  assert_eq(source[12], 12)
}

///|
test "fixed host readback writes exact storage and enforces lifecycle guards" {
  let destination = FixedArray::make(32, b'\xa5')
  assert_eq(raw_test_fixed_readback_boundary(destination, 256), 0)
  assert_eq(destination[4], 0xa5)
  assert_eq(destination[5], 10)
  assert_eq(destination[10], 15)
  assert_eq(destination[11], 0xa5)
}

///|
test "native GEMM retains balanced retryable ownership" {
  assert_eq(raw_test_execution_boundary(256), 0)
}

///|
test "native execution probe rejects unbounded repetition" {
  assert_eq(raw_test_transfer_boundary(0), -2)
  assert_eq(raw_test_transfer_boundary(10001), -2)
  let source = FixedArray::make(32, b'\x00')
  assert_eq(raw_test_fixed_transfer_boundary(source, 0), -2)
  assert_eq(raw_test_fixed_transfer_boundary(source, 10001), -2)
  assert_eq(raw_test_fixed_readback_boundary(source, 0), -2)
  assert_eq(raw_test_fixed_readback_boundary(source, 10001), -2)
  assert_eq(raw_test_execution_boundary(0), -2)
  assert_eq(raw_test_execution_boundary(10001), -2)
  assert_eq(raw_test_region_validation(0), -2)
  assert_eq(raw_test_region_validation(10001), -2)
}
