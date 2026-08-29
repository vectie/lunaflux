///|
fn prefix_radix_allocation_positive_control() -> FixedArray[Byte] {
  FixedArray::make(13, b'R')
}

///|
test "prefix radix allocation positive control is linked" {
  assert_eq(prefix_radix_allocation_positive_control().length(), 13)
}
