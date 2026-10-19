///|
test "cooperative score copy is a bijection with bounded reader windows" {
  let schedule = @attention_physical_ir.SelectedSparseSchedule::new(
    queries=64,
    heads=32,
    selected_width=2051,
    history=8192,
    workers=256,
  ).unwrap()
  let tile = schedule.score_tile(32).unwrap()
  assert_eq(tile.geometry(), (256, 32))
  assert_eq(tile.storage_cells(), 8224L)
  let seen = FixedArray::make(8192, false)
  for key in 0..<256 {
    for component in 0..<32 {
      let cell = tile.cell(key, component).unwrap().to_int()
      assert_false(seen[cell])
      seen[cell] = true
    }
  }
  assert_true(seen.iter().all(x => x))
  assert_true(tile.cell(256, 0) is None)
  assert_true(tile.cell(0, 32) is None)
  assert_true(schedule.score_tile(129) is None)
}
