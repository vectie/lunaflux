///|
test "compressed KV prepare preserves prefill and decode segment order" {
  let limits = ReferenceLimits::new(
    max_values=64,
    max_width=16,
    max_iterations=4,
  )
  let prefill = non_overlapping_compressed_kv_prepare_v1(
    [1.0, 2.0, 3.0, 4.0],
    [9.0, 10.0],
    window_size=2,
    compression_ratio=2,
    head_dimension=1,
    sequence_length=4,
    maximum_position_count=8,
    decode_mode=false,
    limits~,
  )
  assert_eq(prefill, [1.0, 2.0, 3.0, 4.0, 9.0, 10.0])
  let decode = non_overlapping_compressed_kv_prepare_v1(
    [3.0, 4.0],
    [9.0, 10.0],
    window_size=2,
    compression_ratio=2,
    head_dimension=1,
    sequence_length=4,
    maximum_position_count=8,
    decode_mode=true,
    limits~,
  )
  assert_eq(decode, [3.0, 4.0, 9.0, 10.0])
}

///|
test "panic compressed KV prepare rejects truncated segments" {
  let limits = ReferenceLimits::new(
    max_values=64,
    max_width=16,
    max_iterations=4,
  )
  try
    non_overlapping_compressed_kv_prepare_v1(
      [1.0],
      [2.0, 3.0],
      window_size=2,
      compression_ratio=2,
      head_dimension=1,
      sequence_length=4,
      maximum_position_count=8,
      decode_mode=false,
      limits~,
    )
  catch {
    InvalidDimension => ()
    _ => fail("wrong truncation failure")
  } noraise {
    _ => fail("truncated ordinary segment was accepted")
  }
}
