///|
test "recurrent output ownership is explicit and defaults are retained" {
  assert_eq(execution_policy({}).recurrent_tile_rows, 16)
  let selected = execution_policy({
    "projection_staging": "synchronous-k64",
    "recurrent_projection_tile_rows": 32,
  })
  assert_eq(selected.recurrent_tile_rows, 32)
  assert_true(selected.staging is SynchronousTiles(inner_tile=64))
  assert_true(
    selected.description().contains("recurrent_projection_tile_rows=32"),
  )
  let invalid : Array[Json] = [
    { "recurrent_projection_tile_rows": 64 },
    {
      "recurrent_projection_tile_rows": 32,
      "projection_staging": "buffered-k32",
    },
  ]
  for j in invalid {
    try execution_policy(j) catch {
      _ => ()
    } noraise {
      _ => fail("unsupported ownership accepted")
    }
  }
}
