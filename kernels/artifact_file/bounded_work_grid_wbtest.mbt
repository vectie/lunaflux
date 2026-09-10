///|
test "optional bounded row work grid decodes without changing legacy entries" {
  let legacy : Json = [
    {
      "maximum_token_rows": 16,
      "symbol": "rows16",
      "grid_x": 192,
      "block_x": 512,
      "shared_memory_bytes": 16384,
    },
  ]
  assert_eq(decode_row_variants(legacy, 64)[0].bounded_work_grid_x(), None)
  for grid in [1, 48, 192] {
    let document : Json = [
      {
        "maximum_token_rows": 16,
        "symbol": "rows16",
        "grid_x": 192,
        "block_x": 512,
        "shared_memory_bytes": 16384,
        "bounded_work_grid_x": grid,
      },
    ]
    assert_eq(
      decode_row_variants(document, 64)[0].bounded_work_grid_x(),
      Some(grid),
    )
  }
  let invalid : Array[Json] = [null, false, "48", -1, 0, 193]
  for grid in invalid {
    let document : Json = [
      {
        "maximum_token_rows": 16,
        "symbol": "rows16",
        "grid_x": 192,
        "block_x": 512,
        "shared_memory_bytes": 16384,
        "bounded_work_grid_x": grid,
      },
    ]
    try decode_row_variants(document, 64) catch {
      _ => ()
    } noraise {
      _ => fail("invalid bounded work grid accepted")
    }
  }
}
