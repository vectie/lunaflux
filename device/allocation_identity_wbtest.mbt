///|
test "same-shaped allocation owners never join by value equality" {
  // The exact pure predicate used by Allocation::same_storage. Ref values
  // stand in for opaque native owners without requiring a physical GPU.
  let first = Ref(4096L)
  let second = Ref(4096L)
  let borrowed = first
  assert_true(same_storage_owner(first, borrowed))
  assert_false(same_storage_owner(first, second))
  assert_false(same_storage_owner(second, first))
  first.val = 0L
  second.val = 0L
  assert_false(same_storage_owner(first, second))
}
