///|
test "construction stops at first hostile device boundary" {
  let reached = FixedArray::make(4, false)
  let stages : Array[() -> Unit raise Fp8DeviceExecutorError] = [
    () => reached[0] = true,
    () => {
      reached[1] = true
      raise InvalidFp8DeviceExecutor(ResourceState)
    },
    () => reached[2] = true,
    () => reached[3] = true,
  ]
  assert_eq(
    run_fp8_construction_stages(ReadOnlyArray::from_array(stages)),
    Some(InvalidFp8DeviceExecutor(ResourceState)),
  )
  assert_true(reached[0])
  assert_true(reached[1])
  assert_false(reached[2])
  assert_false(reached[3])
}

///|
test "construction observes every ordered success boundary exactly once" {
  let counts = FixedArray::make(3, 0)
  let stages : Array[() -> Unit raise Fp8DeviceExecutorError] = [
    () => counts[0] += 1,
    () => counts[1] += 1,
    () => counts[2] += 1,
  ]
  assert_eq(
    run_fp8_construction_stages(ReadOnlyArray::from_array(stages)),
    None,
  )
  assert_eq(counts[0], 1)
  assert_eq(counts[1], 1)
  assert_eq(counts[2], 1)
}
