///|
fn assert_sorted_tokens(
  logits : ArrayView[Double],
  initial : ArrayView[Int],
  expected : ArrayView[Int],
) -> Unit raise {
  let tokens = FixedArray::from_array(initial)
  sort_candidates(tokens, logits, initial.length())
  for index in 0..<expected.length() {
    assert_eq(tokens[index], expected[index])
  }
}

///|
test "heap sort directly orders odd even reverse and tied candidate sets" {
  assert_sorted_tokens([3.0], [0], [0])
  assert_sorted_tokens([1.0, 2.0], [0, 1], [1, 0])
  assert_sorted_tokens([5.0, 4.0, 3.0, 2.0, 1.0], [0, 1, 2, 3, 4], [
    0, 1, 2, 3, 4,
  ])
  assert_sorted_tokens([5.0, 4.0, 3.0, 2.0, 1.0], [4, 3, 2, 1, 0], [
    0, 1, 2, 3, 4,
  ])
  assert_sorted_tokens([2.0, 7.0, 7.0, -1.0, 2.0, 7.0], [5, 4, 3, 2, 1, 0], [
    1, 2, 5, 0, 4, 3,
  ])
  assert_sorted_tokens(
    [3.0, -4.0, 1.0, 8.0, 0.0, 8.0, 2.0, -4.0],
    [7, 6, 5, 4, 3, 2, 1, 0],
    [3, 5, 0, 6, 2, 4, 1, 7],
  )
}
