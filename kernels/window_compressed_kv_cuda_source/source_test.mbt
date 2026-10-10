///|
test "joint attention renders one causal union and shares ring publication" {
  let window = @precision.WindowSharedKvPrecision::new(
    query_heads=2,
    head_width=128,
    window_tokens=4,
    maximum_rows=3,
    maximum_positions=16,
    maximum_state_bytes=2580L,
  )
  for read_set in [@precision.CompressedRowReadSet::Learned(top_k=2), AllCausal] {
    let plan = @precision.WindowCompressedKvPrecision::new(
      window,
      compression_ratio=4,
      read_set~,
    )
    let source = @window_compressed_kv_cuda_source.window_compressed_kv_source(
      plan,
      symbol_prefix="joint",
    )
    let text = @utf8.decode(source.source_bytes())
    assert_true(!text.contains("}#include"))
    assert_eq(source.launch(1), (3, 2, 128))
    assert_eq(source.launch(2), (4, 1, 128))
    assert_eq(source.symbol(3), "joint_publish")
    assert_true(text.contains("length + compressed_length"))
    assert_true(text.contains("retained != (base + live) / 4"))
    assert_true(text.contains("expf(__fsub_rn(sink[head], maximum))"))
    assert_true(text.contains("row >= append[1]"))
    assert_true(text.contains("const __nv_bfloat16 *key"))
    assert_true(
      text.contains(
        @utf8.decode(
          @window.window_shared_kv_transfer_source(
            window,
            symbol_prefix="joint",
          ),
        ),
      ),
    )
    match read_set {
      Learned(top_k=_) => assert_true(text.contains("selected_counts[row]"))
      AllCausal =>
        assert_true(text.contains("compressed_length = (position + 1) / 4"))
    }
  }
}

///|
test "joint attention lowering rejects unsupported shared footprint and symbols" {
  let plan = @precision.WindowCompressedKvPrecision::new(
    @precision.WindowSharedKvPrecision::new(
      query_heads=1,
      head_width=128,
      window_tokens=128,
      maximum_rows=1,
      maximum_positions=1048576,
      maximum_state_bytes=40000L,
    ),
    compression_ratio=4,
    read_set=AllCausal,
  )
  try
    @window_compressed_kv_cuda_source.window_compressed_kv_source(
      plan,
      symbol_prefix="joint",
    )
  catch {
    @precision.PrecisionPlanError::NoImplementation => ()
    error => raise error
  } noraise {
    _ => fail("oversized static scores accepted")
  }
  try
    @window_compressed_kv_cuda_source.window_compressed_kv_source(
      plan,
      symbol_prefix="bad-prefix",
    )
  catch {
    @precision.PrecisionPlanError::IllegalPlacement => ()
    error => raise error
  } noraise {
    _ => fail("invalid identifier accepted")
  }
}
