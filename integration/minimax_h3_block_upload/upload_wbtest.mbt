///|
test "bounded block copies preserve reordered slices and stop on first failure" {
  let arena : FixedArray[Byte] = [0, 1, 2, 3, 4, 5, 6, 7]
  let received : Array[Byte] = []
  let offsets : Array[Int64] = []
  assert_true(
    copy_chunks(arena, 4L, 100L, 4L, 3, (bytes, offset) => {
      for byte in bytes {
        received.push(byte)
      }
      offsets.push(offset)
      true
    }),
  )
  assert_eq(received, [4, 5, 6, 7])
  assert_eq(offsets, [100L, 103L])
  let mut calls = 0
  assert_false(
    copy_chunks(arena, 0L, 0L, 8L, 2, (_, _) => {
      calls += 1
      false
    }),
  )
  assert_eq(calls, 1)
  assert_false(
    copy_chunks(arena, 7L, 0L, 2L, 2, (_, _) => {
      calls += 1
      true
    }),
  )
  assert_eq(calls, 1)
}
