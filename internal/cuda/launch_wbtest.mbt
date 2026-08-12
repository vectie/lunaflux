///|
extern "c" fn raw_test_aot_launch_boundary(cycles : Int) -> Int = "lunaflux_cuda_test_aot_launch_boundary"

///|
test "AOT launch checks parameters and balances synchronous lifetimes" {
  assert_eq(raw_test_aot_launch_boundary(256), 0)
}

///|
test "AOT launch probe bounds adversarial repetition" {
  assert_eq(raw_test_aot_launch_boundary(0), -2)
  assert_eq(raw_test_aot_launch_boundary(10001), -2)
}
