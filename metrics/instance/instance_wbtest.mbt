///|
test "counter and histogram cells saturate without wrapping" {
  let metrics = LunaInstanceMetrics::new()
  let counter = counter_index(GeneratedTokens)
  metrics.counters[counter] = 0xfffffffffffffffdUL
  metrics.record_counter(GeneratedTokens, amount=10)
  assert_eq(metrics.counter_value(GeneratedTokens), 0xffffffffffffffffUL)
  metrics.record_counter(GeneratedTokens)
  assert_eq(metrics.counter_value(GeneratedTokens), 0xffffffffffffffffUL)
  let bucket = histogram_cell_index(BatchTokens, 0)
  metrics.histogram_buckets[bucket] = 0xffffffffffffffffUL
  metrics.record_histogram(BatchTokens, 1)
  assert_eq(
    metrics.histogram_bucket_value(BatchTokens, 0),
    0xffffffffffffffffUL,
  )
  metrics.set_gauge(KvPagesUsed, 0x7fffffff)
  assert_eq(metrics.gauge_value(KvPagesUsed), 0x7fffffff)
}

///|
test "snapshot exhaustion and forged generations never mutate the slot" {
  let metrics = LunaInstanceMetrics::new()
  metrics.record_counter(Failures, amount=9)
  metrics.snapshot_counters[4] = 77UL
  metrics.snapshot_generation = 0xffffffffffffffffUL
  try metrics.snapshot() catch {
    SnapshotGenerationExhausted => ()
    _ => fail("unexpected snapshot exhaustion error")
  } noraise {
    _ => fail("snapshot generation wrapped")
  }
  assert_eq(metrics.snapshot_generation, 0xffffffffffffffffUL)
  assert_eq(metrics.snapshot_counters[4], 77UL)
  let zero : LunaInstanceMetricsSnapshot = { owner: metrics, generation: 0UL }
  assert_false(zero.is_live())
  try zero.counter_value(Failures) catch {
    StaleSnapshot => ()
    _ => fail("unexpected forged snapshot error")
  } noraise {
    _ => fail("zero-generation snapshot was live")
  }
}

///|
fn instance_metrics_allocation_positive_control() -> FixedArray[Byte] {
  FixedArray::make(19, b'M')
}

///|
test "allocation gate positive control is linked" {
  assert_eq(instance_metrics_allocation_positive_control().length(), 19)
}
