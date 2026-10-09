///|
test "compressed KV prepare source pins official segment order" {
  let rendered = non_overlapping_shared_kv_prepare_bf16_source(
    window_size=128,
    compression_ratio=128,
    head_dimension=512,
    maximum_position_count=1024,
    maximum_batches=2,
    function_symbol="lunaflux_decoder_compressed_kv_prepare_bf16_v1",
  )
  assert_eq(rendered.maximum_compressed_slots(), 8)
  assert_eq(rendered.maximum_key_slots(), 1032)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("mode == 0 ? length : 128"))
  assert_true(source.contains("ordinary_slots + slot"))
  assert_true(source.contains("__restrict__ compressed_kv"))
  assert_false(source.contains("softmax"))
}

///|
test "panic compressed KV prepare rejects unsafe geometry and symbols" {
  try
    non_overlapping_shared_kv_prepare_bf16_source(
      window_size=128,
      compression_ratio=1,
      head_dimension=512,
      maximum_position_count=1024,
      maximum_batches=1,
      function_symbol="valid_symbol",
    )
  catch {
    InvalidCompressionRatio => ()
    _ => fail("wrong ratio failure")
  } noraise {
    _ => fail("ratio one was accepted")
  }
  try
    non_overlapping_shared_kv_prepare_bf16_source(
      window_size=128,
      compression_ratio=128,
      head_dimension=512,
      maximum_position_count=1024,
      maximum_batches=1,
      function_symbol="bad-symbol",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong symbol failure")
  } noraise {
    _ => fail("hostile symbol was accepted")
  }
}
