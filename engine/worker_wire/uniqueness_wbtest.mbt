///|
test "fixed uniqueness scratch sorts arbitrary scalar order" {
  let requests = FixedArray::make(32, 0UL)
  let slots = FixedArray::make(32, -1)
  for shift in 0..<32 {
    for index in 0..<32 {
      let value = (index * 17 + shift) % 32
      requests[index] = Int64::from_int(value + 1).reinterpret_as_uint64()
      slots[index] = 31 - value
    }
    validate_unique_scalar_identities(requests, slots, 32)
    for index in 0..<32 {
      assert_eq(
        requests[index],
        Int64::from_int(index + 1).reinterpret_as_uint64(),
      )
      assert_eq(slots[index], index)
    }
  }
}

///|
test "fixed uniqueness scratch identifies each duplicate dimension" {
  let requests = FixedArray::make(4, 0UL)
  let slots = FixedArray::make(4, -1)
  requests[0] = 4UL
  requests[1] = 2UL
  requests[2] = 4UL
  slots[0] = 0
  slots[1] = 1
  slots[2] = 2
  try validate_unique_scalar_identities(requests, slots, 3) catch {
    InvalidFrame(RequestIdentity) => ()
    _ => fail("expected request duplicate")
  } noraise {
    _ => fail("expected request duplicate")
  }
  requests[0] = 4UL
  requests[1] = 2UL
  requests[2] = 3UL
  slots[0] = 1
  slots[1] = 0
  slots[2] = 1
  try validate_unique_scalar_identities(requests, slots, 3) catch {
    InvalidFrame(CompletionSlot) => ()
    _ => fail("expected slot duplicate")
  } noraise {
    _ => fail("expected slot duplicate")
  }
}
