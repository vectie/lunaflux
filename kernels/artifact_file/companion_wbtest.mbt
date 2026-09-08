///|
test "companion launch is optional but explicit declarations must be complete" {
  assert_eq(parse_companion_dimensions({}), None)
  let dimensions = decode_companion_dimensions(
    @json.parse(
      "{\"grid_x\":4096,\"grid_y\":1,\"grid_z\":1,\"block_x\":128,\"block_y\":1,\"block_z\":1,\"shared_memory_bytes\":4096}",
    ),
  )
  assert_eq(dimensions.grid_x(), 4096)
  assert_eq(dimensions.block_x(), 128)
  assert_eq(dimensions.shared_memory_bytes(), 4096)
  for
    source in [
      "null", "{\"block_x\":128}", "{\"grid_x\":0,\"grid_y\":1,\"grid_z\":1,\"block_x\":128,\"block_y\":1,\"block_z\":1,\"shared_memory_bytes\":4096}",
      "{\"grid_x\":1,\"grid_y\":1,\"grid_z\":1,\"block_x\":128,\"block_y\":1,\"block_z\":1,\"shared_memory_bytes\":4096,\"extra\":1}",
    ] {
    let rejected = try {
      ignore(decode_companion_dimensions(@json.parse(source)))
      false
    } catch {
      _ => true
    }
    assert_true(rejected)
  }
}
