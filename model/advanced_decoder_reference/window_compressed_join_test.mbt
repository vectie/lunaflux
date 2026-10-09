///|
test "window-compressed join preserves prefill padding and literal compressed suffix" {
  let limits = ReferenceLimits::new(
    max_values=64,
    max_width=16,
    max_iterations=4,
  )
  assert_eq(
    deterministic_window_compressed_index_join_v1(
      window_size=4,
      compression_ratio=2,
      query_position=1,
      sequence_length=6,
      decode_mode=false,
      compressed_count=1,
      [6, -1, -1],
      limits~,
    ),
    [0, 1, -1, -1, 6, -1, -1],
  )
}

///|
test "window-compressed join preserves decode ring order" {
  let limits = ReferenceLimits::new(
    max_values=64,
    max_width=16,
    max_iterations=4,
  )
  assert_eq(
    deterministic_window_compressed_index_join_v1(
      window_size=4,
      compression_ratio=2,
      query_position=5,
      sequence_length=1,
      decode_mode=true,
      compressed_count=3,
      [4, 5, 6],
      limits~,
    ),
    [2, 3, 0, 1, 4, 5, 6],
  )
}

///|
test "window-compressed join rejects overlap or reordered compressed indices" {
  try
    deterministic_window_compressed_index_join_v1(
      window_size=4,
      compression_ratio=2,
      query_position=5,
      sequence_length=1,
      decode_mode=true,
      compressed_count=3,
      [4, 6, 5],
      limits=ReferenceLimits::new(max_values=64, max_width=16, max_iterations=4),
    )
  catch {
    InvalidIndexLayout => ()
    _ => fail("unexpected compressed-layout rejection")
  } noraise {
    _ => fail("reordered compressed suffix admitted")
  }
}
