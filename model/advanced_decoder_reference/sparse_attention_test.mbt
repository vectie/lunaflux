///|
test "selected sparse attention uses denominator-only sink and preserves duplicate multiplicity" {
  let limits = ReferenceLimits::new(
    max_values=128,
    max_width=8,
    max_iterations=8,
  )
  let output = selected_sparse_attention(
    [1.0F, 0.0F],
    [1.0F, 2.0F, 0.0F, 4.0F],
    [0.0F],
    [0, 0, 1, -1],
    head_count=1,
    head_dimension=2,
    limits~,
  )
  let scale = 1.0F / 2.0F.sqrt()
  let first = @math.expf(scale)
  let second = @math.expf(0.0F)
  let denominator = 1.0F + first + first + second
  assert_true((output[0] - 2.0F * first / denominator).abs() < 0.00001F)
  assert_true(
    (output[1] - (4.0F * first + 4.0F * second) / denominator).abs() < 0.00001F,
  )
}

///|
test "selected sparse attention sink-only row produces a zero vector" {
  let output = selected_sparse_attention(
    [2.0F, -1.0F],
    [3.0F, 4.0F],
    [5.0F],
    [-1],
    head_count=1,
    head_dimension=2,
    limits=ReferenceLimits::new(max_values=32, max_width=4, max_iterations=4),
  )
  assert_eq(output, [0.0F, 0.0F])
}

///|
test "selected sparse attention rejects invalid geometry and indices" {
  let limits = ReferenceLimits::new(
    max_values=32,
    max_width=4,
    max_iterations=4,
  )
  try
    selected_sparse_attention(
      [1.0F, 2.0F],
      [1.0F, 2.0F],
      [0.0F],
      [1],
      head_count=1,
      head_dimension=2,
      limits~,
    )
  catch {
    KeyIndexOutOfRange(1) => ()
    _ => fail("unexpected invalid-index error")
  } noraise {
    _ => fail("out-of-range selected index admitted")
  }
  try
    selected_sparse_attention(
      [1.0F],
      [1.0F],
      [0.0F, 1.0F],
      [0],
      head_count=1,
      head_dimension=1,
      limits~,
    )
  catch {
    InvalidDimension => ()
    _ => fail("unexpected invalid-shape error")
  } noraise {
    _ => fail("wrong sink shape admitted")
  }
}
