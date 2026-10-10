///|
test "joint window compressed read plans retain causal and learned identity" {
  let window = @precision_ir.WindowSharedKvPrecision::new(
    query_heads=2,
    head_width=128,
    window_tokens=4,
    maximum_rows=3,
    maximum_positions=16,
    maximum_state_bytes=2580L,
  )
  let learned = @precision_ir.WindowCompressedKvPrecision::new(
    window,
    compression_ratio=4,
    read_set=Learned(top_k=2),
  )
  assert_eq(learned.cache_capacity(), 4)
  assert_eq(learned.selected_width(), 2)
  assert_eq(learned.state_bytes(), window.state_bytes())
  assert_eq(learned.effect(), @precision_ir.NumericEffect::CacheCommit)
  assert_eq(learned.score_scale(), 1.0 / 128.0.sqrt())
  let causal = @precision_ir.WindowCompressedKvPrecision::new(
    window,
    compression_ratio=4,
    read_set=AllCausal,
  )
  assert_eq(causal.selected_width(), 4)
  let empty = @precision_ir.WindowCompressedKvPrecision::new(
    window,
    compression_ratio=128,
    read_set=Learned(top_k=2),
  )
  assert_eq(empty.cache_capacity(), 0)
  assert_eq(empty.selected_width(), 0)
}

///|
test "joint attention rejects invalid compression and selection limits" {
  let window = @precision_ir.WindowSharedKvPrecision::new(
    query_heads=2,
    head_width=128,
    window_tokens=4,
    maximum_rows=3,
    maximum_positions=16,
    maximum_state_bytes=2580L,
  )
  for ratio in [0, 1, 1048577] {
    try
      @precision_ir.WindowCompressedKvPrecision::new(
        window,
        compression_ratio=ratio,
        read_set=AllCausal,
      )
    catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement => ()
      error => raise error
    } noraise {
      _ => fail("invalid compression accepted")
    }
  }
  for top_k in [0, 65537] {
    try
      @precision_ir.WindowCompressedKvPrecision::new(
        window,
        compression_ratio=4,
        read_set=Learned(top_k~),
      )
    catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement => ()
      error => raise error
    } noraise {
      _ => fail("invalid selection accepted")
    }
  }
}
