///|
extern "c" fn raw_test_inherited_wait(mode : Int) -> Int = "lunaflux_process_test_inherited_wait"

///|
test "inherited wait distinguishes timeout readiness EOF and EINTR" {
  assert_eq(raw_test_inherited_wait(0), 2)
  assert_eq(raw_test_inherited_wait(1), 0)
  assert_eq(raw_test_inherited_wait(2), 0)
  assert_eq(raw_test_inherited_wait(3), 0)
  assert_eq(raw_test_inherited_wait(4), 2)
}
