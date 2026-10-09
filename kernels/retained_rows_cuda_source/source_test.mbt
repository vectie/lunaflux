///|
test "retained rows validate all group positions before bit copies and publication" {
  let source = @retained_rows_cuda_source.retained_rows_source(
    @precision.RetainedRowsPrecision::new(
      word_width=16,
      position_stride=4,
      maximum_input_rows=2,
      maximum_stored_rows=3,
      maximum_state_bytes=124L,
      maximum_workspace_bytes=12L,
    ),
    symbol_prefix="retained_fixture",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("upstream_append[2]"))
  assert_true(text.contains("positions[row] != (base + row) * 4"))
  assert_true(text.contains("live > 3 - base"))
  assert_true(text.contains("cache[destination + col] = input[source + col]"))
  assert_true(text.contains("cache_counts[3] = length"))
  assert_false(text.contains("bfloat"))
  assert_eq(source.launch(0), (1, 1, 1))
  assert_eq(source.launch(1), (2, 1, 256))
  assert_eq(source.launch(2), (1, 1, 1))
  assert_eq(source.symbol(0), "retained_fixture_reserve")
}
