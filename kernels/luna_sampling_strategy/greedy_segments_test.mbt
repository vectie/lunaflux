///|
test "greedy partition covers tails without vocabulary-specific branching" {
  for vocabulary in [1, 4095, 4096, 4097, 151936, 2147483647] {
    let plan = @luna_sampling_strategy.plan_greedy_segments(
      vocabulary_size=vocabulary,
      maximum_rows=32,
      segment_width=4096,
    )
    let (width, count) = plan.partition()
    assert_eq(width, 4096)
    assert_true(
      (count.to_int64() - 1L) * width.to_int64() < vocabulary.to_int64(),
    )
    assert_true(count.to_int64() * width.to_int64() >= vocabulary.to_int64())
    assert_eq(plan.workspace_bytes(), count.to_int64() * 32L * 16L)
  }
}

///|
test "greedy partition rejects invalid and overflowing storage domains" {
  for shape in [(0, 1, 1), (1, 0, 1), (1, 1, 0), (2147483647, 2147483647, 1)] {
    let (vocabulary, rows, width) = shape
    try
      @luna_sampling_strategy.plan_greedy_segments(
        vocabulary_size=vocabulary,
        maximum_rows=rows,
        segment_width=width,
      )
    catch {
      @luna_sampling_strategy.InvalidSamplingStrategy(_) => ()
    } noraise {
      _ => fail("invalid greedy partition accepted")
    }
  }
}
