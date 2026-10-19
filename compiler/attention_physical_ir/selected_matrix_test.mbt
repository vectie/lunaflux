///|
test "matrix sparse tile covers selected tails without sharing query indices" {
  let p = @attention_physical_ir.SelectedMatrixFold::new(
    qk_dimension=256,
    value_dimension=256,
    columns=8,
    reduction=16,
  ).unwrap()
  assert_eq(p.geometry(), (8, 16))
  for (keys, tiles) in [(0, 0), (1, 1), (8, 1), (9, 2), (2051, 257)] {
    assert_eq(p.score_tiles(keys), tiles)
  }
  assert_true(
    @attention_physical_ir.SelectedMatrixFold::new(
      qk_dimension=255,
      value_dimension=256,
      columns=8,
      reduction=16,
    )
    is None,
  )
}
