///|
test "long context KV observation admits exact live pages or full retirement" {
  assert_true(spawned_long_context_kv_valid(1, 1, 31, 32, 1))
  assert_true(spawned_long_context_kv_valid(1, 2, 30, 32, 2))
  assert_true(spawned_long_context_kv_valid(0, 0, 32, 32, 2))
  assert_false(spawned_long_context_kv_valid(1, 1, 31, 32, 2))
  assert_false(spawned_long_context_kv_valid(0, 1, 31, 32, 1))
  assert_false(spawned_long_context_kv_valid(0, 0, 31, 32, 1))
}

///|
test "long context frame copy retains caller-independent bytes" {
  let source : FixedArray[Byte] = [b'a', b'b', b'c']
  let copied = copy_long_context_frame(source)
  source[0] = b'z'
  assert_eq(copied, [b'a', b'b', b'c'])
}
