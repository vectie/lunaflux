///|
test "ragged tile capacity includes per-row tails without multiplying token capacity" {
  let layout = AttentionMetadataLayout::new(maximum_rows=8, maximum_tokens=129).unwrap()
  assert_eq(layout.bucket(16), Some((0, 4, 15)))
  assert_eq(layout.bucket(32), Some((1, 124, 11)))
  assert_eq(layout.bucket(64), Some((2, 212, 9)))
  assert_eq(layout.bucket(128), Some((3, 284, 8)))
  assert_eq(layout.cell_count(), 348)
  assert_eq(layout.bucket(24), None)
  assert_eq(
    AttentionMetadataLayout::new(maximum_rows=8, maximum_tokens=7),
    None,
  )
}
