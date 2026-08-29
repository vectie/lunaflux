///|
test "partial startup cleanup is reverse ordered and retry preserves remaining authority" {
  let open = FixedArray::make(11, true)
  let attempts : Array[Int] = []
  let fail_stream_once = Ref(true)
  let close = (index : Int) => {
    if open[index] {
      attempts.push(index)
      if index == 4 && fail_stream_once.val {
        fail_stream_once.val = false
        raise Invalid(ResourceState)
      }
      open[index] = false
    }
  }
  try
    run_cleanup_chain(
      () => close(0),
      () => close(1),
      () => close(2),
      () => close(3),
      () => close(4),
      () => close(5),
      () => close(6),
      () => close(7),
      () => close(8),
      () => close(9),
      () => close(10),
    )
  catch {
    Invalid(ResourceState) => ()
    _ => fail("unexpected cleanup failure")
  } noraise {
    _ => fail("expected injected cleanup failure")
  }
  assert_eq(attempts.length(), 5)
  for index in 0..<4 {
    assert_false(open[index])
  }
  for index in 4..<11 {
    assert_true(open[index])
  }
  run_cleanup_chain(
    () => close(0),
    () => close(1),
    () => close(2),
    () => close(3),
    () => close(4),
    () => close(5),
    () => close(6),
    () => close(7),
    () => close(8),
    () => close(9),
    () => close(10),
  )
  assert_eq(attempts.length(), 12)
  assert_eq(attempts[0], 0)
  assert_eq(attempts[1], 1)
  assert_eq(attempts[2], 2)
  assert_eq(attempts[3], 3)
  assert_eq(attempts[4], 4)
  for index in 4..<11 {
    assert_eq(attempts[index + 1], index)
  }
  for index in 0..<11 {
    assert_false(open[index])
  }
}
