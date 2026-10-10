///|
extern "c" fn test_device_copy(cycles : Int) -> Int = "lunaflux_cuda_test_device_copy"

///|
test "device-only copy snapshots restore bytes and retain asynchronous ownership" {
  assert_eq(test_device_copy(128), 0)
}
