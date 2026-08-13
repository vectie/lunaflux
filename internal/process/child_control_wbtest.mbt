///|
extern "c" fn raw_test_expect_clean_eof(mode : Int) -> Int = "lunaflux_process_test_expect_clean_eof"

///|
test "inherited clean EOF rejects bytes partial prefixes and timeout" {
  assert_eq(raw_test_expect_clean_eof(0), 0)
  assert_eq(raw_test_expect_clean_eof(1), 4)
  assert_eq(raw_test_expect_clean_eof(2), 4)
  assert_eq(raw_test_expect_clean_eof(3), 2)
}
